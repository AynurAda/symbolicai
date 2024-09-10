import concurrent.futures
import threading
import time
import random
from symai import *
from symai.functional import EngineRepository

class TestExpression(Expression):
    def __init__(self, data_point):
        self.data_point = data_point

    def forward(self, executor_callback, *args, **kwargs):
        instance = Symbol(f"{self.data_point}")
        res = instance.query(f"multiply {self.data_point} by 2?", executor_callback=executor_callback)
        cube = res.query(f"What is the cube of this number?", executor_callback=executor_callback)
        if random.choice([True, False]):
            cube = cube.query(f"What is the cube of this number?", executor_callback=executor_callback)
        return [instance.value, res.value, cube.value]

class Publisher:
    def __init__(self, expr, num_workers, engine, dataset, batch_size=5):
        self.num_workers = num_workers
        self.engine = engine
        self.dataset = dataset
        self.results = {}
        self.arguments = []
        self.lock = threading.Lock()
        self.batch_size = max(1, min(num_workers, batch_size))  # Limit batch size
        self.batch_ready = threading.Event()
        self.processing_complete = threading.Event()
        self.results_queue = {}
        self.result_events = {}
        self.pending_tasks = 0
        self.expr = expr

    def process_data(self, data_point):
        expr = self.expr(data_point)
        return expr.forward(executor_callback=self.executor_callback)

    def executor_callback(self, argument):
        with self.lock:
            self.arguments.append(argument)
            result_id = id(argument)
            self.results_queue[result_id] = None
            self.result_events[result_id] = threading.Event()
            if len(self.arguments) == self.batch_size:
                self.batch_ready.set()
        self.result_events[result_id].wait()
        with self.lock:
            result = self.results_queue.pop(result_id)
            del self.result_events[result_id]
        return result

    def execute_queries(self):
        while not self.processing_complete.is_set() or self.pending_tasks > 0:
            self.batch_ready.wait()  # Add timeout
            self.batch_ready.clear()
            with self.lock:
                current_arguments = self.arguments[:self.batch_size]
                self.arguments = self.arguments[self.batch_size:]
                self.pending_tasks -= len(current_arguments)
            if current_arguments:  # Only process if there are arguments
                results = self.engine(current_arguments)
                with self.lock:
                    for arg, result in zip(current_arguments, results):
                        result_id = id(arg)
                        self.results_queue[result_id] = result
                        self.result_events[result_id].set()

    def run(self):
        query_thread = threading.Thread(target=self.execute_queries)
        query_thread.start()

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            future_to_data = {executor.submit(self.process_data, data_point): data_point for data_point in self.dataset}
            with self.lock:
                self.pending_tasks = len(self.dataset)
            for future in concurrent.futures.as_completed(future_to_data):
                data_point = future_to_data[future]
                try:
                    result = future.result()
                    self.results[data_point] = result
                except Exception as exc:
                    print(f'Data point {data_point} generated an exception: {exc}')

        with self.lock:
            if self.arguments:
                self.batch_ready.set()

        self.processing_complete.set()
        query_thread.join(timeout=10)  # Add timeout to join

        return [self.results.get(data_point) for data_point in sorted(self.dataset)]

def main():
    print("Main: Starting")
    dataset = list(range(1, 20))
    num_workers = 10   
    batch_size = 10
    publisher = Publisher(TestExpression, num_workers, engine, dataset, batch_size)
    results = publisher.run()
    
    print("Execution completed. Results:")
    for data_point, result in zip(dataset, results):
        print(f"  Data point {data_point}: {result}")

    print("Main: Exiting.")

if __name__ == "__main__":
    main()
