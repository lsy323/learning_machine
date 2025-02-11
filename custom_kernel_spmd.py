import time
from typing import Optional, List, Tuple, Callable, Any
from contextlib import contextmanager
import os
import functools

import torch
import torch_xla
import torch_xla.distributed.spmd as xs
import torch_xla.core.xla_model as xm


from torch_xla.experimental.custom_kernel import (
    FlashAttention,
    trace_pallas,
    jax_import_guard
)

@contextmanager
def _jax_env_context():
  try:
    os.environ['SKIP_MEGASCALE_PJRT_CLIENT'] = 'true'
    yield
  finally:
    os.environ.pop('SKIP_MEGASCALE_PJRT_CLIENT', None)


def requires_jax(func: Callable) -> Callable:
  """Decorator that ensures JAX is safely imported before function execution"""

  @functools.wraps(func)
  def wrapper(*args, **kwargs) -> Any:
    try:
      jax_import_guard()
    except ImportError as e:
      raise ImportError(
          "JAX import guard fail due to PJRT client is unavailable.") from e
    with _jax_env_context():
      return func(*args, **kwargs)

  return wrapper


def generate_ctx_need_grad(*args):
  ctx_need_grad = [False for _ in range(len(args))]
  for i, arg in enumerate(args):
    if arg is not None and isinstance(arg, torch.Tensor) and arg.requires_grad:
      ctx_need_grad[i] = True
  return ctx_need_grad


def get_segment_partition_spec(partition_sepc: Optional[Tuple[str, ...]] = None, ndim: int = 3) -> Optional[Tuple[str, ...]]:
    if partition_sepc is None:
        return None
    
    assert partition_sepc is not None # For typehinting in VSC
    
    batch_spec = partition_sepc[:-3]
    sequence_spec = partition_sepc[-2]
    
    if ndim == 2:
        output = (batch_spec[0], sequence_spec)
    else:
        output = (*batch_spec, sequence_spec)
        
    return output
    

def fa_custom_forward(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool,
    q_segment_ids: torch.Tensor, kv_segment_ids: torch.Tensor, sm_scale: float,
    ab: Optional[torch.Tensor], partition_spec: str, mesh: str,
    ctx_grad: List[bool]
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
           torch.Tensor, torch.Tensor]:

    partition_spec = eval(partition_spec)
    
    has_seq = partition_spec[-2] != None
    
    if has_seq:
        k = xm.all_gather(k, -2)
        v = xm.all_gather(v, -2)
        if kv_segment_ids is not None:
            # Gathers all the kv_segment_ids to all devices
            kv_segment_ids = xm.all_gather(kv_segment_ids, -1)
        
    mesh = xs.get_global_mesh()

    from jax.experimental.pallas.ops.tpu.flash_attention import _flash_attention_impl

    q_full_shape = None

    # Suprisingly, any tensor that is input to the custom_op decorated function will show
    # requires_grad=False. Is this a bug or feature? We have to pass ctx_grad to record the
    # requires_grad for inputs.
    # Original we use save_residuals = q.requires_grad or k.requires_grad or v.requires_grad
    save_residuals = any(ctx_grad[:3])

    # SPMD integration.
    # mark_sharding is in-placed, and therefore save the full q, k, v for the backward.
    # PyTorch tell us clone is necessary:
    full_q = q.clone()
    full_k = k.clone()
    full_v = v.clone()
    if ab is not None:
        full_ab = ab.clone()
    else:
        full_ab = None
        if partition_spec is not None:
            q_full_shape = q.shape
    

    
    kv_spec = partition_spec[:-2] + (None, ) + partition_spec[-1:]
    q = xs.enable_manual_sharding(q, partition_spec, mesh=mesh).global_tensor
    q_mid_shape = q.shape
    q = q.reshape(-1, *q.shape[-3:])
    k = xs.enable_manual_sharding(k, kv_spec, mesh=mesh).global_tensor
    k_mid_shape = k.shape
    k = k.reshape(-1, *k.shape[-3:])
    v = xs.enable_manual_sharding(v, kv_spec, mesh=mesh).global_tensor

    
    v_mid_shape = v.shape
    v = v.reshape(-1, *v.shape[-3:])

    if ab is not None:
        ab = xs.enable_manual_sharding(
            ab, partition_spec, mesh=mesh).global_tensor
        ab_mid_shape = ab.shape
        ab = ab.reshape(-1, *ab_mid_shape[-3:])
    else:
        ab_mid_shape = None
        
    # It computes the shape and type of o, l, m.
    shapes = [q.shape]
    dtypes = [q.dtype]
    if save_residuals:
        res_shape = list(q.shape)
        res_shape[-1] = SPMDFlashAttention.MIN_BLOCK_SIZE
        
        for _ in range(2):
            shapes.append(res_shape)
            dtypes.append(torch.float32)

    with torch.no_grad():
        if partition_spec is not None and q_segment_ids is not None and kv_segment_ids is not None:
            # partition_spec is for q,k,v with shape [batch, num_head, seq_len, head_dim], segment id
            # is of shape [batch, seq_len], hence we need to tweak it a bit
            segment_id_partition_spec = get_segment_partition_spec(partition_spec, q_segment_ids.ndim)
            segment_id_partition_spec_no_seq = segment_id_partition_spec[:-1] + (None, )
            
            q_segment_ids = xs.enable_manual_sharding(
                q_segment_ids, segment_id_partition_spec, mesh=mesh).global_tensor
            q_segment_ids_mid_shape = q_segment_ids.shape
            q_segment_ids = q_segment_ids.reshape(-1, *q_segment_ids_mid_shape[-1:])

            
            kv_segment_ids = xs.enable_manual_sharding(
                kv_segment_ids, segment_id_partition_spec_no_seq, mesh=mesh).global_tensor
            kv_segment_ids_mid_shape = kv_segment_ids.shape
            kv_segment_ids = kv_segment_ids.reshape(-1, *kv_segment_ids_mid_shape[-1:])
            
        else:
            q_segment_ids_mid_shape = None
            kv_segment_ids_mid_shape = None
            
        segment_ids, q_segment_ids_fa, kv_segment_ids_fa = FlashAttention.prepare_segment_ids(
            q_segment_ids, kv_segment_ids)


    shapes_out = [
        q_mid_shape,
        k_mid_shape,
        v_mid_shape,
        ab_mid_shape,
        q_segment_ids_mid_shape,
        kv_segment_ids_mid_shape
    ]

    # We can't directly use flash_attention as we need to override the save_residuals flag which returns
    # l and m that is needed for the backward. Then we lose all the shape checks.
    # TODO: replicate the shape checks on flash_attention.
    # Here we seperate the tracing and execution part just to support SegmentIds.
    
    payload, _ = trace_pallas(
        _flash_attention_impl,
        q,
        k,
        v,
        ab,
        segment_ids,
        save_residuals,
        causal,
        sm_scale,
        min(SPMDFlashAttention.DEFAULT_BLOCK_SIZES["block_b"], q.shape[0]),
        min(SPMDFlashAttention.DEFAULT_BLOCK_SIZES["block_q"], q.shape[2]),
        min(SPMDFlashAttention.DEFAULT_BLOCK_SIZES["block_k_major"], k.shape[2]),
        min(SPMDFlashAttention.DEFAULT_BLOCK_SIZES["block_k"], k.shape[2]),
        False,
        static_argnums=range(5, 13),
        use_cache=True,
    )

    args = [q, k, v]
    if ab is not None:
        args += [ab]
    if segment_ids is not None:
        args += [q_segment_ids_fa, kv_segment_ids_fa]
    o = torch_xla._XLAC._xla_tpu_custom_call(args, payload, shapes, dtypes)


    if not save_residuals:
        shapes_out = shapes_out + [None, ] # LM mid shape
        o = o[0]
        # SPMD integration
        if partition_spec is not None:
            o = o.reshape(q_mid_shape)
            o = xs.disable_manual_sharding(
                o, partition_spec, q_full_shape, mesh=mesh).global_tensor
            # We need to consistently return full_q, full_k, full_v,... even though they are empty to support AOT.
            return tuple([o] + [torch.Tensor() for _ in range(6)] + shapes_out)

    assert isinstance(o, list)
    o, *aux = o
    l, m = (v[..., 0] for v in aux[-2:])

    
    # SPMD integration
    if partition_spec is not None:
        # Should be [batch, num_head, seq_len]
        lm_shape = q_full_shape[:-1]
        lm_spec = partition_spec[:-1]
        lm_mid_shape = q_mid_shape[:-1]
        
        o = o.reshape(q_mid_shape)
        o = xs.disable_manual_sharding(
            o, partition_spec, q_full_shape, mesh=mesh).global_tensor
        
        # m is pre soft-max (q@k.T) matrix, kept for numerical stabilization during backward
        m = m.reshape(*lm_mid_shape)
        m = xs.disable_manual_sharding(
            m, lm_spec, lm_shape, mesh=mesh).global_tensor
        
        # l is the sum of exponents in m, kept for numerical stabilization during backward
        l = l.reshape(*lm_mid_shape)
        l = xs.disable_manual_sharding(
            l, lm_spec, lm_shape, mesh=mesh).global_tensor
        

    # q_segment_ids and kv_segment_ids are sharded here if partition_spec is provided
    # but it should be OK as the backward will use the same partition_spec
    
    outs = [o] + [full_q, full_k, full_v, l, m, full_ab] + shapes_out + [lm_mid_shape, ]

    return tuple(outs)

def fa_custom_backward(
    grad_output: torch.Tensor, q: torch.Tensor, k: torch.Tensor,
    v: torch.Tensor, o: torch.Tensor, l: torch.Tensor, m: torch.Tensor,
    q_segment_ids: Optional[torch.Tensor],
    kv_segment_ids: Optional[torch.Tensor], ab: Optional[torch.Tensor],
    causal: bool, sm_scale: float, partition_spec: str, mesh: str,
    q_full_shape: List[int], kv_full_shape: List[int],
    ab_full_shape: Optional[List[int]], ctx_grad: List[bool],
    q_mid_shape,
    k_mid_shape,
    v_mid_shape,
    ab_mid_shape,
    q_segment_ids_mid_shape,
    kv_segment_ids_mid_shape,
    lm_mid_shape,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    from jax.experimental.pallas.ops.tpu.flash_attention import _flash_attention_bwd_dq, _flash_attention_bwd_dkv
    grad_q = grad_k = grad_v = grad_ab = segment_ids = None

    require_grad_q, require_grad_k, require_grad_v, *rest = ctx_grad
    require_grad_ab = ctx_grad[-3]

    partition_spec = eval(partition_spec)
    
    if partition_spec[-2] != None:
        partition_spec = partition_spec[:-2] + (None, ) + partition_spec[-1:]
    mesh = xs.get_global_mesh()
    assert mesh is not None
    q_full_shape = torch.Size(q_full_shape)
    kv_full_shape = torch.Size(kv_full_shape)
    ab_full_shape = torch.Size(ab_full_shape) if ab_full_shape is not None else None

    l = l.reshape(-1, *l.shape[-2:])
    m = m.reshape(-1, *m.shape[-2:])
    grad_output = grad_output.reshape(-1, *grad_output.shape[-3:])
    o = o.reshape(-1, *o.shape[-3:])

    grad_i = torch.sum(
        o.to(torch.float32) * grad_output.to(torch.float32),
        axis=-1)  # [batch_size, num_heads, q_seq_len]

    expanded_l = l.unsqueeze(-1).expand([-1 for _ in l.shape] +
                                        [SPMDFlashAttention.MIN_BLOCK_SIZE])
    expanded_m = m.unsqueeze(-1).expand([-1 for _ in m.shape] +
                                        [SPMDFlashAttention.MIN_BLOCK_SIZE])
    
    expanded_grad_i = grad_i.unsqueeze(-1).expand([-1 for _ in grad_i.shape] +
                                                    [SPMDFlashAttention.MIN_BLOCK_SIZE])


    # SPMD integration
    if partition_spec is not None:
        if q_segment_ids is not None and kv_segment_ids is not None:
            # partition_spec is for q,k,v with shape [batch, num_head, seq_len, head_dim], segment id
            # is of shape [batch, seq_len], hence we need to tweak it a bit
            segment_id_partition_spec = get_segment_partition_spec(partition_spec, q_segment_ids.ndim)

            has_seq = partition_spec[-2] != None
            if has_seq:
                pass
                #q_segment_ids = xm.all_gather(q_segment_ids, -2)
                #kv_segment_ids = xm.all_gather(kv_segment_ids, -2)
            
            q_segment_ids = xs.enable_manual_sharding(
                q_segment_ids, segment_id_partition_spec, mesh=mesh).global_tensor
            q_seq = q_segment_ids.shape[-1]
            q_segment_ids = q_segment_ids.reshape(-1, q_seq)

            kv_segment_ids = xs.enable_manual_sharding(kv_segment_ids, segment_id_partition_spec, mesh=mesh).global_tensor
            
            kv_seq = kv_segment_ids.shape[-1]
            kv_segment_ids = kv_segment_ids.reshape(-1, kv_seq)


        q = xm.all_gather(q, -2)
        k = xm.all_gather(k, -2)
        v = xm.all_gather(v, -2)
        q = xs.enable_manual_sharding(q, partition_spec, mesh=mesh).global_tensor
        k = xs.enable_manual_sharding(k, partition_spec, mesh=mesh).global_tensor
        v = xs.enable_manual_sharding(v, partition_spec, mesh=mesh).global_tensor
        
        q_seq = q.shape[-2]
        q = q.reshape(-1, *q_mid_shape[-3:-2], q_seq, q_mid_shape[-1])
        kv_seq = k.shape[-2]
        k = k.reshape(-1, *k_mid_shape[-3:-2], kv_seq, k_mid_shape[-1])
        v = v.reshape(-1, *v_mid_shape[-3:-2], kv_seq, v_mid_shape[-1])
        
        if len(partition_spec) == 4:
            used_partition_spec = partition_spec
        else:
            used_partition_spec = (partition_spec[:-3], ) + partition_spec[-3:]
        
        expanded_l = xs.enable_manual_sharding(
            expanded_l, used_partition_spec, mesh=mesh).global_tensor
        expanded_l = expanded_l.reshape(-1, *expanded_l.shape[-3:])
        
        expanded_m = xs.enable_manual_sharding(
            expanded_m, used_partition_spec, mesh=mesh).global_tensor
        expanded_m = expanded_m.reshape(-1, *expanded_m.shape[-3:])
        
        grad_output = xs.enable_manual_sharding(grad_output, used_partition_spec, mesh=mesh).global_tensor
        grad_output = grad_output.reshape(-1, *grad_output.shape[-3:])
    
        expanded_grad_i = xs.enable_manual_sharding(
            expanded_grad_i, used_partition_spec, mesh=mesh).global_tensor
        expanded_grad_i = expanded_grad_i.reshape(-1, *expanded_grad_i.shape[-3:])
        
        if ab is not None:
            ab = xs.enable_manual_sharding(ab, partition_spec, mesh=mesh).global_tensor
            ab = ab.reshape(-1, *ab_mid_shape[-3:])

    if q_segment_ids is not None and kv_segment_ids is not None:
        segment_ids, q_segment_ids_fa, kv_segment_ids_fa = SPMDFlashAttention.prepare_segment_ids(
            q_segment_ids, kv_segment_ids)
        
    if require_grad_q:
        payload, _ = trace_pallas(
            _flash_attention_bwd_dq,
            q,
            k,
            v,
            ab,
            segment_ids,
            l,
            m,
            grad_output,
            grad_i,
            block_q_major=min(SPMDFlashAttention.DEFAULT_BLOCK_SIZES["block_q_dq"],
                            q.shape[2]),
            block_k_major=min(
                SPMDFlashAttention.DEFAULT_BLOCK_SIZES["block_k_major_dq"], k.shape[2]),
            block_k=min(SPMDFlashAttention.DEFAULT_BLOCK_SIZES["block_k_dq"],
                        k.shape[2]),
            sm_scale=sm_scale,
            causal=causal,
            mask_value=SPMDFlashAttention.DEFAULT_MASK_VALUE,
            debug=False,
            static_argnames=[
                "block_q_major", "block_k_major", "block_k", "sm_scale", "causal",
                "mask_value", "debug"
            ],
            use_cache=True,
        )
    args = [q, k, v]
    if ab is not None:
      args += [ab]
    if segment_ids is not None:
      args += [q_segment_ids_fa, kv_segment_ids_fa]
    args += [expanded_l, expanded_m, grad_output, expanded_grad_i]

    outputs = [q]
    if ab is not None:
      outputs += [ab]
    grads = torch_xla._XLAC._xla_tpu_custom_call(args, payload,
                                                 [i.shape for i in outputs],
                                                 [i.dtype for i in outputs])
    if require_grad_q:
      grad_q = grads[0]

    if require_grad_ab:
      grad_ab = grads[1]

    if require_grad_k or require_grad_v:
        payload, _ = trace_pallas(
            _flash_attention_bwd_dkv,
            q,
            k,
            v,
            ab,
            segment_ids,
            l,
            m,
            grad_output,
            grad_i,
            block_q_major=min(
                SPMDFlashAttention.DEFAULT_BLOCK_SIZES["block_q_major_dkv"],
                q.shape[2]),
            block_k_major=min(
                SPMDFlashAttention.DEFAULT_BLOCK_SIZES["block_k_major_dkv"],
                k.shape[2]),
            block_k=min(SPMDFlashAttention.DEFAULT_BLOCK_SIZES["block_k_dkv"],
                        k.shape[2]),
            block_q=min(SPMDFlashAttention.DEFAULT_BLOCK_SIZES["block_q_dkv"],
                        q.shape[2]),
            sm_scale=sm_scale,
            causal=causal,
            mask_value=SPMDFlashAttention.DEFAULT_MASK_VALUE,
            debug=False,
            static_argnames=[
                "block_q_major", "block_k_major", "block_k", "block_q", "sm_scale",
                "causal", "mask_value", "debug"
            ],
            use_cache=True)

    grads = torch_xla._XLAC._xla_tpu_custom_call(args, payload,
                                                 [k.shape, v.shape],
                                                 [k.dtype, v.dtype])

    if require_grad_k:
        grad_k = grads[0]
    if require_grad_v:
        grad_v = grads[1]

    # SPMD integration
    
    if partition_spec is not None:
        q_seq = grad_q.shape[-2]

        grad_q = grad_q.reshape(*q_mid_shape[:-2], q_seq, *q_mid_shape[-1:])
        grad_q = xs.disable_manual_sharding(
            grad_q, partition_spec, q_full_shape, mesh=mesh).global_tensor
        kv_seq_len = k.shape[-2]
        fixed_k_shape = k_mid_shape[:-2] + (kv_seq_len, ) + k_mid_shape[-1:]
        grad_k = grad_k.reshape(fixed_k_shape)
        grad_k = xs.disable_manual_sharding(
            grad_k, partition_spec, kv_full_shape, mesh=mesh).global_tensor
        fixed_v_shape = v_mid_shape[:-2] + (kv_seq_len, ) + v_mid_shape[-1:]
        grad_v = grad_v.reshape(fixed_v_shape)
        grad_v = xs.disable_manual_sharding(
            grad_v, partition_spec, kv_full_shape, mesh=mesh).global_tensor
        if ab is not None:
            grad_ab = grad_ab.reshape(-1, *ab_mid_shape[-3:])
            grad_ab = xs.disable_manual_sharding(
                grad_ab, partition_spec, ab_full_shape, mesh=mesh).global_tensor
    
    return grad_q, grad_k, grad_v, grad_ab

def flash_attention(
    q,  # [batch_size, num_heads, q_seq_len, d_model]
    k,  # [batch_size, num_heads, kv_seq_len, d_model]
    v,  # [batch_size, num_heads, kv_seq_len, d_model]
    causal=False,
    q_segment_ids=None,  # [batch_size, q_seq_len]
    kv_segment_ids=None,  # [batch_size, kv_seq_len]
    sm_scale=1.0,
    *,
    ab=None,  # [batch_size, num_heads, q_seq_len, kv_seq_len]
    partition_spec=None,
    mesh=None,
):
    # TODO: support SPMD and Dynamo with segment_ids.
    return SPMDFlashAttention.apply(
        q,
        k,
        v,
        causal,
        q_segment_ids,
        kv_segment_ids,
        sm_scale,
        ab,
        partition_spec,
        mesh,
    )


class SPMDFlashAttention(FlashAttention):
    """
    This is a simplified wrapper on top of https://github.com/google/jax/blob/b2058d72b7e1693a41303d5411572aabf99b7981/jax/experimental/pallas/ops/tpu/flash_attention.py#L139
    where we only takes q, k, v and causal as input and set block_sizes for the users.
    """

    @staticmethod
    @requires_jax
    def forward(ctx,
        q, k, v,
        causal,
        q_segment_ids, kv_segment_ids,
        sm_scale, ab,
        partition_spec,
        mesh
    ):
        ctx.q_shape = q.shape
        ctx.k_shape = k.shape
        ctx.causal = causal
        ctx.sm_scale = sm_scale
        ctx.partition_spec = partition_spec
        ctx.mesh = mesh
        ctx.q_full_shape = q.shape
        ctx.kv_full_shape = k.shape
        ctx.ab_full_shape = ab.shape if ab is not None else None
        ctx.q_mid_shape = None
        ctx.k_mid_shape = None
        ctx.v_mid_shape = None
        ctx.q_segment_ids_mid_shape = None
        ctx.kv_segment_ids_mid_shape = None
        ctx.ab_mid_shape = None
        partition_spec = str(partition_spec)
        mesh = str(mesh)
        custom_op_arg = [
            q, k, v, causal, q_segment_ids, kv_segment_ids, sm_scale, ab,
            partition_spec, mesh
        ]
        ctx_grads = generate_ctx_need_grad(*custom_op_arg)
        # AOT compatiable funtion only accepts argument types listed https://github.com/pytorch/pytorch/blob/82859f61857ef39898b34a5cdf0ae56ec25704d9/torch/_functorch/_aot_autograd/utils.py#L23-L34, so we serliaze partition_spec and mesh into string.
        outs = fa_custom_forward(*custom_op_arg, ctx_grads)

        o = outs[0]
        (
            full_q, full_k, full_v, 
            l, m, full_ab, 
            q_mid_shape, k_mid_shape, v_mid_shape,
            ab_mid_shape, 
            q_segment_ids_mid_shape, kv_segment_ids_mid_shape,
            lm_mid_shape,
        )  = [x for x in outs[1:]]

        ctx.q_mid_shape = q_mid_shape
        ctx.k_mid_shape = k_mid_shape
        ctx.v_mid_shape = v_mid_shape
        ctx.ab_mid_shape = ab_mid_shape
        ctx.q_segment_ids_mid_shape = q_segment_ids_mid_shape
        ctx.kv_segment_ids_mid_shape = kv_segment_ids_mid_shape
        ctx.lm_mid_shape = lm_mid_shape
        
        # q_segment_ids and kv_segment_ids are sharded here if partition_spec is provided
        # but it should be OK as the backward will use the same partition_spec
        ctx.save_for_backward(
            full_q, full_k, full_v,
            o, l, m, q_segment_ids,
            kv_segment_ids, full_ab
        )
        return o

    @staticmethod
    @requires_jax
    def backward(ctx, grad_output):
        q, k, v, o, l, m, q_segment_ids, kv_segment_ids, ab = ctx.saved_tensors
        causal = ctx.causal
        sm_scale = ctx.sm_scale
        partition_spec = ctx.partition_spec
        mesh = ctx.mesh
        q_full_shape = ctx.q_full_shape
        kv_full_shape = ctx.kv_full_shape
        ab_full_shape = ctx.ab_full_shape

        grad_output, q, k, v, o, l, m = [
            t.contiguous() for t in (grad_output, q, k, v, o, l, m)
        ]

        # this segment_ids only reflects the local shape of segment_ids
        custom_op_arg = [
            grad_output, q, k, v, o, l, m, q_segment_ids, kv_segment_ids, ab,
            causal, sm_scale,
            str(partition_spec),
            str(mesh), q_full_shape, kv_full_shape, ab_full_shape
        ]
  
        ctx_grads = ctx.needs_input_grad
        grad_q, grad_k, grad_v, grad_ab = fa_custom_backward(
            *custom_op_arg, ctx_grads,
            ctx.q_mid_shape, ctx.k_mid_shape, ctx.v_mid_shape,
            ctx.ab_mid_shape, ctx.q_segment_ids_mid_shape, ctx.kv_segment_ids_mid_shape,
            ctx.lm_mid_shape
        )
        return grad_q, grad_k, grad_v, None, None, None, None, grad_ab, None, None



if __name__ == "__main__":
    import jax
    import argparse
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--is-3d', action='store_true', default=False, help='Whether to test 3D attention')
    parser.add_argument('--replicated', action='store_true', default=False, help='Whether to test replicated attention')
    parser.add_argument('--sequence-axis', type=int, default=2, help='Sequence axis')
    parser.add_argument('--model-axis', type=int, default=1, help='Model axis')
    parser.add_argument('--ddp-axis', type=int, default=4, help='DDP axis')
    parser.add_argument('--forward-only', action='store_true', default=False, help='Whether to test forward only')
    parser.add_argument('--no-mask', action='store_true', default=False, help='Whether to test without mask')
    parser.add_argument('--no-rand', action='store_true', default=False, help='Whether to test without random data')
    
    
    args = parser.parse_args()
    
    no_rand = args.no_rand
    
    jax.config.update("jax_default_matmul_precision", "highest")
    mesh, attn_spec = None, None
    import torch_xla.runtime as xr
    import torch
    
    print('Configuring SPMD')
    xr.use_spmd()

    from torch_xla.distributed.spmd import Mesh, ShardingSpec
    import numpy as np
    from termcolor import colored

    print('Configuring Mesh')
    num_devices = xr.global_runtime_device_count()
    model_axis = args.model_axis
    ddp_axis = args.ddp_axis
    sequence_axis = args.sequence_axis
    fsdp_axis = num_devices // sequence_axis // ddp_axis // model_axis
    mesh_shape = (ddp_axis, fsdp_axis, model_axis, sequence_axis)
    device_ids = np.array(range(num_devices))
    mesh = Mesh(device_ids, mesh_shape, ("data","fsdp", "model", "sequence"))
    attn_spec = ("data", "fsdp", "model", "sequence", None)
    if args.replicated:
        attn_spec = (None, None, None, None, None)
    xs.set_global_mesh(mesh)
    per_axis_batch_size = 16
    batch_size = per_axis_batch_size*ddp_axis*fsdp_axis

    q_seq = 1024 * 8
    k_seq = 512 * 8
    depth = 256
    num_heads = 8
    minibatch, batch_size = ddp_axis, batch_size//ddp_axis
    if args.is_3d:
        attn_spec = (
            (attn_spec[0], attn_spec[1]),
            attn_spec[2],
            attn_spec[3],
            attn_spec[4]
        )

    for i in range(3):
        print(f'-======== iteration {i}======')
        print(f'Creating data with q_sec {q_seq} k_seq {k_seq} depth {depth} num_heads {num_heads} batch_size {batch_size}', flush=True)
        
        new_tensor = torch.ones if no_rand else torch.rand
        q = new_tensor(minibatch, batch_size, num_heads, q_seq, depth)
        k = new_tensor(minibatch, batch_size, num_heads, k_seq, depth)
        v = new_tensor(minibatch, batch_size, num_heads, k_seq, depth)
        mask = (new_tensor(minibatch, batch_size, k_seq) > 0.5).to(torch.float32)
        if args.no_mask:
            mask = None
        
        forward_only = args.forward_only
        
        if args.is_3d:
            print('Testing 3D attention instead of 4D')
            q = q.reshape((minibatch*batch_size, *q.shape[-3:]))
            k = k.reshape((minibatch*batch_size, *k.shape[-3:]))
            v = v.reshape((minibatch*batch_size, *v.shape[-3:]))
            if mask is not None:
                mask = mask.reshape((minibatch*batch_size, mask.shape[-1]))
                
            
        batch_spec = attn_spec[0] if len(attn_spec) == 4 else (attn_spec[0], attn_spec[1])
        
        mask_partition_spec = ('data', 'fsdp', 'sequence') if not args.is_3d else (('data', 'fsdp'), 'sequence') 

        mask_sharding_spec = ShardingSpec(mesh, mask_partition_spec)

        print('Moving data to TPU')
        sharding_spec = ShardingSpec(
            mesh,
            attn_spec,
        )
        
        
        device = xm.xla_device()
        q = xm.send_cpu_data_to_device(q, device, sharding_spec)[0]
        k = xm.send_cpu_data_to_device(k, device, sharding_spec)[0]
        v = xm.send_cpu_data_to_device(v, device, sharding_spec)[0]
        if mask is not None:
            mask = mask.to(device)
            xs.mark_sharding(mask, mesh, mask_partition_spec)
        
        sm_scale = (q.shape[-1])**-0.5
        if not forward_only:
            q.requires_grad = True
            k.requires_grad = True
            v.requires_grad = True

        q_segment_indexes = torch.ones(
            (minibatch, batch_size, q_seq), device='cpu', dtype=torch.float32
        )
        if args.is_3d:
            q_segment_indexes = q_segment_indexes.reshape((minibatch*batch_size, q_seq))
        
        q_segment_indexes = q_segment_indexes.to(device)
        xs.mark_sharding(q_segment_indexes, mesh, mask_partition_spec)

        print('Running TPU flash attention')
        if not forward_only:
            q.retain_grad()
            k.retain_grad()
            v.retain_grad()
            
        xm.mark_step(wait=True)
        xm.wait_device_ops()
        start = time.perf_counter()


        o = SPMDFlashAttention.apply(
            q, k, v, False, q_segment_indexes if mask is not None else None, mask, sm_scale, None, attn_spec, mesh
        )

        print(f"created output with shape {o.shape}", flush=True)

        
        if not forward_only:
            loss = o.sum()
            loss.backward()
        xm.mark_step(wait=True)
        xm.wait_device_ops()
        end = time.perf_counter()
        print('total time is ', end - start)

        if not forward_only:
            fa_q_grad = q.grad.cpu()
            fa_k_grad = k.grad.cpu()
            fa_v_grad = v.grad.cpu()

        fa_o = o.cpu()