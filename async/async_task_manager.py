import asyncio
import threading
from typing import Dict, Any, Optional, Callable
import uuid
import concurrent.futures

class SimpleTaskManager:
    def __init__(self):
        self.tasks: Dict[str, asyncio.Task] = {}
        self.results: Dict[str, Any] = {}
        self.loop = None
        self.thread = None
        self._start_event_loop()
    
    def _start_event_loop(self):
        """Start the event loop in a separate thread"""
        ready_event = threading.Event()
        
        def run_loop():
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)
            
            # Schedule the ready signal to be called once loop starts
            self.loop.call_soon(ready_event.set)
            self.loop.run_forever()
        
        self.thread = threading.Thread(target=run_loop, daemon=True)
        self.thread.start()
        
        ready_event.wait()  # Wait until the event loop is actually running
    
    def add_task(self, coro_func: Callable, *args, task_id: Optional[str] = None, **kwargs) -> str:
        """Add a task and return its ID. coro_func should be an async function."""
        if task_id is None:
            task_id = str(uuid.uuid4())
        
        # Create the coroutine
        coro = coro_func(*args, **kwargs)
        
        # Schedule the task in the event loop (don't wait for completion)
        asyncio.run_coroutine_threadsafe(
            self._add_task_async(coro, task_id), self.loop
        )
        
        return task_id
    
    async def _add_task_async(self, coro, task_id: str):
        """Internal async method to add task"""
        task = asyncio.create_task(coro)
        self.tasks[task_id] = task
        
        # Set up callback to store result
        def done_callback(t):
            try:
                self.results[task_id] = t.result()
            except Exception as e:
                self.results[task_id] = e
        
        task.add_done_callback(done_callback)
    
    def wait_for_task(self, task_id: str, timeout: Optional[float] = None) -> Any:
        """Wait for a specific task to complete and return its result"""
        if task_id not in self.tasks:
            raise ValueError(f"Task {task_id} not found")
        
        # Wait for the task to complete
        future = asyncio.run_coroutine_threadsafe(
            self._wait_for_task_async(task_id), self.loop
        )
        
        try:
            future.result(timeout=timeout)
            return self.results[task_id]
        except concurrent.futures.TimeoutError:
            raise TimeoutError(f"Task {task_id} did not complete within {timeout} seconds")
    
    async def _wait_for_task_async(self, task_id: str):
        """Internal async method to wait for task"""
        await self.tasks[task_id]
    
    def is_done(self, task_id: str) -> bool:
        """Check if a task is complete"""
        return task_id in self.tasks and self.tasks[task_id].done()
    
    def get_result(self, task_id: str) -> Any:
        """Get result if task is done, otherwise raise ValueError"""
        if not self.is_done(task_id):
            raise ValueError(f"Task {task_id} is not completed yet")
        return self.results[task_id]
    
    def cancel_task(self, task_id: str) -> bool:
        """Cancel a task"""
        if task_id in self.tasks:
            future = asyncio.run_coroutine_threadsafe(
                self._cancel_task_async(task_id), self.loop
            )
            return future.result()
        return False
    
    async def _cancel_task_async(self, task_id: str) -> bool:
        """Internal async method to cancel task"""
        return self.tasks[task_id].cancel()
    
    def shutdown(self):
        """Shutdown the task manager and event loop"""
        if self.loop:
            self.loop.call_soon_threadsafe(self.loop.stop)
            self.thread.join()


# Example usage
async def example_task(name: str, delay: int):
    print(f"Task {name} starting...")
    await asyncio.sleep(delay)
    result = f"Task {name} completed after {delay}s"
    print(result)
    return result

async def fetch_data(url: str):
    # Simulate an HTTP request
    await asyncio.sleep(1)
    return f"Data from {url}"

def main():
    import time
    print("=== Demonstrating Concurrent Task Execution ===\n")
    
    # Create task manager
    tm = SimpleTaskManager()
    
    # Record start time
    start_time = time.time()
    print(f"Starting all tasks at: {start_time:.2f}")
    
    # Add tasks from synchronous code - all start immediately and run concurrently
    task1_id = tm.add_task(example_task, "A (3s)", 3)
    task2_id = tm.add_task(example_task, "B (1s)", 1) 
    task3_id = tm.add_task(example_task, "C (2s)", 2)
    task4_id = tm.add_task(fetch_data, "https://api.example.com")
    
    print("✅ All tasks added instantly and running concurrently in background!")
    print("📊 If sequential: would take 3+1+2+1=7 seconds total")
    print("📊 If concurrent: should take ~3 seconds (longest task)\n")
    
    # Do other synchronous work while tasks run
    print("🔄 Doing other work while tasks run in background...")
    for i in range(3):
        time.sleep(0.5)
        print(f"   Main thread work step {i+1}/3")
    
    print("\n⏱️  Checking task status after 1.5 seconds:")
    print(f"Task A (3s) done: {tm.is_done(task1_id)}")
    print(f"Task B (1s) done: {tm.is_done(task2_id)}")  # Should be done
    print(f"Task C (2s) done: {tm.is_done(task3_id)}")
    print(f"Task D (fetch) done: {tm.is_done(task4_id)}")  # Should be done
    
    # Wait for tasks in order of completion (fastest first)
    print(f"\n🎯 Waiting for fastest task (B - 1s)...")
    result_b = tm.wait_for_task(task2_id)
    elapsed = time.time() - start_time
    print(f"✅ Task B completed in {elapsed:.1f}s: {result_b}")
    
    print(f"\n🎯 Waiting for medium task (C - 2s)...")
    result_c = tm.wait_for_task(task3_id)
    elapsed = time.time() - start_time
    print(f"✅ Task C completed in {elapsed:.1f}s: {result_c}")
    
    print(f"\n🎯 Waiting for fetch task...")
    result_fetch = tm.wait_for_task(task4_id)
    elapsed = time.time() - start_time
    print(f"✅ Fetch task completed in {elapsed:.1f}s: {result_fetch}")
    
    print(f"\n🎯 Waiting for slowest task (A - 3s)...")
    result_a = tm.wait_for_task(task1_id)
    total_elapsed = time.time() - start_time
    print(f"✅ Task A completed in {total_elapsed:.1f}s: {result_a}")
    
    print(f"\n🏁 All tasks completed in {total_elapsed:.1f} seconds total!")
    print("📈 This proves concurrent execution - sequential would take ~7 seconds")
    
    # Final status check
    print(f"\n📊 Final status:")
    print(f"Task A done: {tm.is_done(task1_id)}")
    print(f"Task B done: {tm.is_done(task2_id)}")
    print(f"Task C done: {tm.is_done(task3_id)}")
    print(f"Task D done: {tm.is_done(task4_id)}")
    
    # Clean up
    tm.shutdown()
    print("\n✅ Task manager shut down")

if __name__ == "__main__":
    main()
