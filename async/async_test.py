import asyncio
import numpy as np
import time


class TransferManager:
    
    def _copy_sync(src: np.ndarray, dst: np.ndarray):
        """
        Synchronously copy data from src to dst.
        """
        assert src.shape == dst.shape, "Source and destination arrays must have the same shape"
        dst[:] = src[:]
        time.sleep(2)
        

    async def _copy_start(src: np.ndarray, dst: np.ndarray):
        assert src.shape == dst.shape, "Source and destination arrays must have the same shape"
        dst[:] = src[:]
        await asyncio.sleep(2)
    
    @staticmethod
    def copy_start(src: np.ndarray, dst: np.ndarray):
        """
        Start the transfer of data from src to dst.
        """
        return asyncio.create_task(TransferManager._copy_start(src, dst))

    @staticmethod
    def copy_wait(task):
        """
        Wait for the transfer task to complete.
        """
        return asyncio.run(task)



