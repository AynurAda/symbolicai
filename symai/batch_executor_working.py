import concurrent.futures
import threading
import random

class Publisher:
    def __init__(self, expr, num_workers, engine, dataset, batch_size=5):
        self.num_workers = num_workers
        self.engine = engine
        self.dataset = dataset
        self.results = {}
        self.arguments = []
        self.lock = threading.Lock()
        self.batch_size = batch_size
        self.batch_ready = threading.Event()
        self.processing_complete = threading.Event()
        self.llm_responses = {}
        self.llm_response_ready = {}
        self.pending_tasks = 0
        self.expr = expr

    def process_data(self, data_point):
        expr = self.expr(data_point)
        return expr.forward(executor_callback=self.executor_callback)

    def executor_callback(self, argument):
        with self.lock:
            self.arguments.append(argument)
            arg_id = id(argument)
            self.llm_responses[arg_id] = None
            self.llm_response_ready[arg_id] = threading.Event()
            print(len(self.arguments))
            if len(self.arguments) >= self.batch_size:
                self.batch_ready.set()
        self.llm_response_ready[arg_id].wait()
        with self.lock:
            llm_response = self.llm_responses.pop(arg_id)
            del self.llm_response_ready[arg_id]
        return llm_response  # This is the intermediary result from the LLM

    def execute_queries(self):
        while not self.processing_complete.is_set() or self.pending_tasks > 0 or self.arguments:
            self.batch_ready.wait()
            self.batch_ready.clear()
            with self.lock:
                current_arguments = self.arguments[:self.batch_size]
                self.arguments = self.arguments[self.batch_size:]
            if current_arguments:
                llm_batch_responses = self.engine(current_arguments)
                with self.lock:
                    for arg, llm_response in zip(current_arguments, llm_batch_responses):
                        arg_id = id(arg)
                        self.llm_responses[arg_id] = llm_response
                        self.llm_response_ready[arg_id].set()
            if self.arguments:
                self.batch_ready.set()

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
                    final_result = future.result()
                    self.results[data_point] = final_result
                    with self.lock:
                        self.pending_tasks -= 1  # Decrease pending_tasks when a data point is fully processed
                except Exception as exc:
                    print(f'Data point {data_point} generated an exception: {exc}')
                    with self.lock:
                        self.pending_tasks -= 1  # Decrease pending_tasks for failed data points

        self.processing_complete.set()
        self.batch_ready.set()  # Ensure execute_queries can exit its wait state
        query_thread.join()
        return [self.results.get(data_point) for data_point in sorted(self.dataset)]
