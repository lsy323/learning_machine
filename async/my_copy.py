import asyncio
import numpy as np
import time



def copy_sync(src: np.ndarray, dst: np.ndarray):
    """
    Synchronously copy data from src to dst.
    """
    assert src.shape == dst.shape, "Source and destination arrays must have the same shape"
    dst[:] = src[:]
    time.sleep(2)
    

async def _copy_start(src: np.ndarray, dst: np.ndarray):
    print("start_copy")
    assert src.shape == dst.shape, "Source and destination arrays must have the same shape"
    dst[:] = src[:]
    await asyncio.sleep(2)
    print("finish_copy")


async def copy_start(src: np.ndarray, dst: np.ndarray):
    """
    Start the transfer of data from src to dst.
    """
    print(asyncio.get_event_loop())
    task = asyncio.create_task(_copy_start(src, dst))
    return task


async def copy_wait(task):
    """
    Wait for the transfer task to complete.
    This is a synchronous function that can be called from non-async code.
    """
    await task
# async def copy_wait(task):
#     """
#     Wait for the transfer task to complete.
#     """
#     await task



