# -*- coding: utf-8 -*-
import asyncio
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from my_copy import copy_sync, copy_start, copy_wait

class TestBasicFunctionality:
    """Test basic functionality."""
    
    def test_sync_copy(self):
        src = np.array([1, 2, 3])
        dst = np.empty_like(src)
        time_start = time.time()
        copy_sync(src, dst)
        time_end = time.time()
        time_s = time_end - time_start
        expected_s = 2
        assert abs(time_s - expected_s) < 0.1, f"Expected {expected_s} seconds, got {time_s:.2f} seconds"
        np.testing.assert_array_equal(src, dst)
    
    def test_async_copy(self):
        src = np.array([1, 2, 3])
        dst1 = np.empty_like(src)
        dst2 = np.empty_like(src)
        time_start = time.time()
        task1 = asyncio.run(copy_start(src, dst1))
        task2 = asyncio.run(copy_start(src, dst2))
        print(f"Task1 ID: {task1}, Task2 ID: {task2}")
        copy_wait(task1)
        copy_wait(task2)
        time_end = time.time()
        time_s = time_end - time_start
        expected_s = 2
        assert abs(time_s - expected_s) < 0.1, f"Expected {expected_s} seconds, got {time_s:.2f} seconds"
        np.testing.assert_array_equal(src, dst1)
        np.testing.assert_array_equal(src, dst2)
