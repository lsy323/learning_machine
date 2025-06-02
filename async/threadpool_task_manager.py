import concurrent.futures
from typing import Dict, Any, Optional, Callable
import uuid
import time

class ThreadPoolTaskManager:
    def __init__(self, max_workers: int = 4):
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        self.futures: Dict[str, concurrent.futures.Future] = {}
        self.results: Dict[str, Any] = {}
    
    def add_task(self, func: Callable, *args, task_id: Optional[str] = None, **kwargs) -> str:
        """Add a task and return its ID. func can be any callable (sync or async)."""
        if task_id is None:
            task_id = str(uuid.uuid4())
        
        # Submit the function to thread pool
        future = self.executor.submit(func, *args, **kwargs)
        self.futures[task_id] = future
        
        # Set up callback to store result
        def done_callback(f):
            try:
                self.results[task_id] = f.result()
            except Exception as e:
                self.results[task_id] = e
        
        future.add_done_callback(done_callback)
        return task_id
    
    def wait_for_task(self, task_id: str, timeout: Optional[float] = None) -> Any:
        """Wait for a specific task to complete and return its result"""
        if task_id not in self.futures:
            raise ValueError(f"Task {task_id} not found")
        
        try:
            # Wait for the future to complete
            self.futures[task_id].result(timeout=timeout)
            return self.results[task_id]
        except concurrent.futures.TimeoutError:
            raise TimeoutError(f"Task {task_id} did not complete within {timeout} seconds")
    
    def is_done(self, task_id: str) -> bool:
        """Check if a task is complete"""
        return task_id in self.futures and self.futures[task_id].done()
    
    def get_result(self, task_id: str) -> Any:
        """Get result if task is done, otherwise raise ValueError"""
        if not self.is_done(task_id):
            raise ValueError(f"Task {task_id} is not completed yet")
        return self.results[task_id]
    
    def cancel_task(self, task_id: str) -> bool:
        """Cancel a task (only works if not started yet)"""
        if task_id in self.futures:
            return self.futures[task_id].cancel()
        return False
    
    def wait_for_any(self, task_ids: list, timeout: Optional[float] = None) -> tuple:
        """Wait for any of the specified tasks to complete"""
        futures = [self.futures[tid] for tid in task_ids if tid in self.futures]
        done, pending = concurrent.futures.wait(
            futures, timeout=timeout, return_when=concurrent.futures.FIRST_COMPLETED
        )
        
        # Find which task completed
        for task_id, future in self.futures.items():
            if future in done:
                return task_id, self.results[task_id]
        
        raise TimeoutError("No tasks completed within timeout")
    
    def wait_for_all(self, task_ids: list, timeout: Optional[float] = None) -> Dict[str, Any]:
        """Wait for all specified tasks to complete"""
        futures = [self.futures[tid] for tid in task_ids if tid in self.futures]
        concurrent.futures.wait(futures, timeout=timeout)
        
        return {tid: self.results[tid] for tid in task_ids if tid in self.results}
    
    def shutdown(self, wait: bool = True):
        """Shutdown the thread pool"""
        self.executor.shutdown(wait=wait)


# Example functions that work with threads
def cpu_intensive_task(name: str, iterations: int):
    """Example CPU-bound task"""
    print(f"CPU task {name} starting with {iterations} iterations...")
    total = 0
    for i in range(iterations):
        total += i * i
        # Simulate work
        if i % 100000 == 0:
            print(f"CPU task {name}: {i}/{iterations} iterations done")
    
    result = f"CPU task {name} completed: sum = {total}"
    print(result)
    return result

def io_task(name: str, delay: float):
    """Example I/O-bound task (simulated with sleep)"""
    print(f"I/O task {name} starting (delay: {delay}s)...")
    time.sleep(delay)
    result = f"I/O task {name} completed after {delay}s"
    print(result)
    return result

def fetch_data(url: str):
    """Simulate fetching data from URL"""
    print(f"Fetching data from {url}...")
    time.sleep(1)  # Simulate network delay
    return f"Data fetched from {url}"

def mixed_task(name: str, cpu_work: int, io_work: float):
    """Task that does both CPU and I/O work"""
    print(f"Mixed task {name} starting...")
    
    # Do some CPU work
    total = sum(i * i for i in range(cpu_work))
    print(f"Mixed task {name}: CPU work done (sum={total})")
    
    # Do some I/O work
    time.sleep(io_work)
    
    result = f"Mixed task {name} completed: CPU sum={total}, I/O delay={io_work}s"
    print(result)
    return result


def main():
    print("=== Demonstrating ThreadPool Concurrent Execution ===\n")
    
    # Create task manager with 4 worker threads
    tm = ThreadPoolTaskManager(max_workers=4)
    
    # Record start time
    start_time = time.time()
    print(f"Starting all tasks at: {start_time:.2f}")
    
    # Add different types of tasks - all run concurrently
    task1_id = tm.add_task(cpu_intensive_task, "CPU-A", 500000)
    task2_id = tm.add_task(io_task, "IO-B", 2.0)
    task3_id = tm.add_task(fetch_data, "https://api.example.com")
    task4_id = tm.add_task(mixed_task, "Mixed-C", 200000, 1.5)
    task5_id = tm.add_task(io_task, "IO-D", 1.0)
    
    print("✅ All tasks submitted to thread pool and running concurrently!")
    print("📊 Tasks are running in separate threads, utilizing multiple CPU cores")
    print("📊 I/O tasks can run while CPU tasks are computing\n")
    
    # Do other work while tasks run
    print("🔄 Main thread doing other work...")
    for i in range(3):
        time.sleep(0.7)
        elapsed = time.time() - start_time
        print(f"   Main thread work step {i+1}/3 (elapsed: {elapsed:.1f}s)")
    
    # Check status
    elapsed = time.time() - start_time
    print(f"\n⏱️  Task status after {elapsed:.1f} seconds:")
    print(f"CPU task done: {tm.is_done(task1_id)}")
    print(f"I/O task B (2s) done: {tm.is_done(task2_id)}")
    print(f"Fetch task done: {tm.is_done(task3_id)}")
    print(f"Mixed task done: {tm.is_done(task4_id)}")
    print(f"I/O task D (1s) done: {tm.is_done(task5_id)}")
    
    # Wait for first task to complete
    print(f"\n🎯 Waiting for any task to complete...")
    first_done_id, first_result = tm.wait_for_any([task1_id, task2_id, task3_id, task4_id, task5_id])
    elapsed = time.time() - start_time
    print(f"✅ First completed task {first_done_id} in {elapsed:.1f}s")
    
    # Wait for all tasks
    print(f"\n🎯 Waiting for all tasks to complete...")
    all_results = tm.wait_for_all([task1_id, task2_id, task3_id, task4_id, task5_id])
    total_elapsed = time.time() - start_time
    
    print(f"\n🏁 All tasks completed in {total_elapsed:.1f} seconds!")
    print("📈 This demonstrates true parallelism with multiple threads")
    
    # Show results
    print(f"\n📊 Results:")
    for task_id, result in all_results.items():
        print(f"  {task_id}: {result}")
    
    # Clean up
    tm.shutdown()
    print("\n✅ Thread pool shut down")


if __name__ == "__main__":
    main()
