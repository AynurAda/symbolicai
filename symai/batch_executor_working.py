import threading
import time
import random
from symai import *
from symai.functional import EngineRepository
engine = EngineRepository.get("neurosymbolic")

class TestExpression(Expression):
    def __init__(self, data_point):
        print("TestExpression: Initializing")
        self.data_point = data_point

    def forward(self, executor_callback, *args, **kwargs):
        print(f"TestExpression: forward called with data_point: {self.data_point}")
        instance = Symbol(f"Processing data point: {self.data_point}")
        res = instance.query(f"write a poem about number {self.data_point}?", executor_callback=executor_callback)
        if random.choice([True, False]):
            cube = instance.query(f"What is the cube of {self.data_point}?", executor_callback=executor_callback)
            res = [res.value, cube.value]
        return res

class Publisher:
    def __init__(self, subscriber_count, engine, dataset):
        print(f"Publisher: Initializing with {subscriber_count} subscribers")
        self.subscribers = []
        self.results = {}
        self.arguments = {}
        self.finished_subscribers = set()
        self.subscriber_count = subscriber_count
        self.engine = engine
        self.lock = threading.Lock()
        self.all_finished_event = threading.Event()
        self.dataset = dataset

    def register(self, subscriber):
        print(f"Publisher: Registering {subscriber.name}")
        self.subscribers.append(subscriber)

    def distribute_data(self):
        chunk_size = len(self.dataset) // len(self.subscribers)
        remainder = len(self.dataset) % len(self.subscribers)
        start = 0
        for i, subscriber in enumerate(self.subscribers):
            end = start + chunk_size + (1 if i < remainder else 0)
            subscriber.set_data(self.dataset[start:end])
            start = end

    def receive_result(self, subscriber_name, result, data_point):
        with self.lock:
            if subscriber_name not in self.results:
                self.results[subscriber_name] = []
            self.results[subscriber_name].append((data_point, result))
            if subscriber_name in self.arguments:
                del self.arguments[subscriber_name]
            if len(self.finished_subscribers) == len(self.subscribers):
                self.all_finished_event.set()
            elif self.arguments:  # Only execute query if there are pending arguments
                self.execute_query()

    def receive_argument(self, subscriber_name, argument):
        with self.lock:
            if subscriber_name not in self.finished_subscribers:
                self.arguments[subscriber_name] = argument
                if len(self.arguments) == len(self.subscribers) - len(self.finished_subscribers):
                    self.execute_query()

    def execute_query(self):
        arguments = list(self.arguments.values())
        results = _execute_query_batch(self.engine, arguments)
        for subscriber_name, result in zip(self.arguments.keys(), results):
            subscriber = next((s for s in self.subscribers if s.name == subscriber_name), None)
            if subscriber:
                subscriber.receive_result(result)
        self.arguments.clear()

    def subscriber_finished(self, subscriber_name):
        with self.lock:
            self.finished_subscribers.add(subscriber_name)
            if subscriber_name in self.arguments:
                del self.arguments[subscriber_name]
            if len(self.finished_subscribers) == len(self.subscribers):
                self.all_finished_event.set()
            elif self.arguments:  # Execute query if there are any pending arguments
                self.execute_query()

    def get_final_results(self):
        self.all_finished_event.wait()
        return self.results

class Subscriber(threading.Thread):
    def __init__(self, name, publisher, expression_class):
        threading.Thread.__init__(self)
        print(f"Subscriber {name}: Initializing")
        self.name = name
        self.publisher = publisher
        self.expression_class = expression_class
        self.result_event = threading.Event()
        self.llm_result = None
        self.data = None
        self.is_finished = False

    def set_data(self, data):
        self.data = data

    def run(self):
        print(f"{self.name}: Starting run method")
        for data_point in self.data:
            expr = self.expression_class(data_point)
            res = expr.forward(executor_callback=self.executor_callback)
            self.publisher.receive_result(self.name, res, data_point)
        self.is_finished = True
        self.publisher.subscriber_finished(self.name)
        print(f"{self.name}: Finished processing all data points")

    def executor_callback(self, argument):
        if not self.is_finished:
            self.publisher.receive_argument(self.name, argument)
            self.result_event.wait()
            self.result_event.clear()
            return self.llm_result
        else:
            print(f"{self.name}: Ignoring argument submission as processing is finished")
            return None

    def receive_result(self, result):
        self.llm_result = result
        self.result_event.set()

class BatchExecutor:
    def __init__(self, data, batch_size, expression_class):
        self.data = data
        self.batch_size = batch_size
        self.expression_class = expression_class
        self.engine = EngineRepository.get("neurosymbolic")

    def execute(self):
        publisher = Publisher(self.batch_size, self.engine, self.data)
        subscribers = []
        for i in range(self.batch_size):
            subscriber = Subscriber(f"Subscriber-{i+1}", publisher, self.expression_class)
            publisher.register(subscriber)
            subscribers.append(subscriber)

        publisher.distribute_data()

        for subscriber in subscribers:
            subscriber.start()

        final_results = publisher.get_final_results()
        ordered_results = {data_point: None for data_point in self.data}
        for results in final_results.values():
            for data_point, result in results:
                ordered_results[data_point] = result

        return [ordered_results[data_point] for data_point in self.data]

def _execute_query_batch(engine, arguments):
    results = []
    for argument in arguments:
        if argument.prop.preview:
            results.append(engine.preview(argument))
        else:
            outputs = engine(argument)
            results.append(outputs)   
    print(f"results: {results}")
    return results

def main():
    print("Main: Starting")
    dataset = list(range(1, 21))  
    batch_executor = BatchExecutor(dataset, batch_size=5, expression_class=TestExpression)
    results = batch_executor.execute()
    
    print("Execution completed. Results:")
    for data_point, result in zip(dataset, results):
        print(f"  Data point {data_point}: {result}")

    print("Main: Exiting.")

if __name__ == "__main__":
    main()