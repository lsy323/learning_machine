import jax
import jax.numpy as jnp
from jax.experimental.pallas.ops.tpu.paged_attention.paged_attention_kernel import \
    paged_attention


def paged_attn(
    q: jax.Array,
    k_cache: jax.Array,
    v_cache: jax.Array,
    block_tables: jax.Array,
    context_lens: jax.Array,
    pages_per_compute_block: int,
) -> jax.Array:

    output = paged_attention(
        q,
        k_cache,
        v_cache,
        context_lens,
        block_tables,
        pages_per_compute_block=pages_per_compute_block,
    )
    return output


batch_size = 8
num_heads = 32
head_size = 128
num_kv_heads = 32
context_len = 512
num_blocks = 2048
block_size = 128
pages_per_compute_block = 32
MAX_NUM_BLOCKS_PER_SEQ = 512

rng_key = jax.random.PRNGKey(0)
query = jax.random.normal(rng_key, (batch_size, 1, num_heads, head_size),
                          dtype=jnp.bfloat16)
k_cache = jax.random.normal(rng_key,
                            (num_kv_heads, num_blocks * block_size, head_size),
                            dtype=jnp.bfloat16)
v_cache = jax.random.normal(rng_key,
                            (num_kv_heads, num_blocks * block_size, head_size),
                            dtype=jnp.bfloat16)

block_tables = jax.random.randint(rng_key,
                                  (batch_size, MAX_NUM_BLOCKS_PER_SEQ),
                                  0,
                                  num_blocks,
                                  dtype=jnp.int32)
context_lens = jnp.array([context_len] * batch_size, dtype=jnp.int32)

sm_scale = head_size**-0.5
query = query.squeeze(1)
query = query * sm_scale
head_size = query.shape[-1]
num_slots = k_cache.shape[-2]
k_cache = k_cache.reshape(-1, num_slots // block_size, block_size, head_size)
v_cache = v_cache.reshape(-1, num_slots // block_size, block_size, head_size)

args = (query, k_cache, v_cache, block_tables, context_lens,
        pages_per_compute_block)
jitted = jax.jit(paged_attn, static_argnums=(5, ))
breakpoint()
# Pre-opt HLO
pre_opt_hlo = jitted.lower(*args).as_text("hlo")
# Post-opt HLO
post_opt_hlo = jitted.lower(*args).compile().as_text()
# StableHLO
ir = jitted.lower(*args).compiler_ir()
print(ir)
