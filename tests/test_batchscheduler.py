import random
import time

import pytest

from symai import Expression
from symai.backend.base import Engine
from symai.batch_executor_working import BatchScheduler


class TestExpression(Expression):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def forward(self, input, **kwargs):
        print(f"Processing: {input}")
        return f"Processed: {input}"

class NestedExpression(Expression):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.nested_expr = TestExpression()
    
    def forward(self, input, **kwargs):
        nested_result = self.nested_expr(input, **kwargs)  # Pass kwargs here
        return f"Nested result: {nested_result}"

class DoubleNestedExpression(Expression):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.nested_expr1 = TestExpression()
        self.nested_expr2 = NestedExpression()
    
    def forward(self, input, **kwargs):
        result1 = self.nested_expr1(input, **kwargs)   
        result2 = self.nested_expr2(input, **kwargs)  
        return f"Double nested results: {result1} and {result2}"

class ConditionalExpression(Expression):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.nested_expr = NestedExpression()
    
    def forward(self, input, **kwargs):
        print(input)
        print("hohohoho:", input)
        if len(input) > 10:
            return f"Conditional: {self.nested_expr(input, **kwargs)}"   
        else:
            return f"Conditional: Input too short, not processing further"
        
class SlowExpression(Expression):
    def __init__(self, delay=5, **kwargs):
        super().__init__(**kwargs)
        self.delay = delay
    
    def forward(self, input, **kwargs):
        time.sleep(self.delay)   
        return f"Slow processed (after {self.delay} seconds): {input}"

class DoubleNestedExpressionSlow(Expression):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.nested_expr1 = TestExpression()
        self.nested_expr2 = NestedExpression()
        self.slow_expr = SlowExpression()
    
    def forward(self, input, **kwargs):
        result1 = self.nested_expr1(input, **kwargs)
        result2 = self.nested_expr2(input, **kwargs)
        slow_result = self.slow_expr(input, **kwargs)
        return f"Double nested results: {result1}, {result2}, and {slow_result}"

class ConditionalSlowExpression(Expression):
    def __init__(self, delay=5, threshold=10, **kwargs):
        super().__init__(**kwargs)
        self.delay = delay
        self.threshold = threshold
        self.slow_expr = SlowExpression(delay=delay)
    
    def forward(self, input, **kwargs):
        if len(input) > self.threshold:
            return f"Conditional Slow: {self.slow_expr(input, **kwargs)}"
        else:
            return f"Conditional Slow: Input too short, processed quickly: {input}"
        

class MockGPTXChatEngine(Engine):
    def __init__(self):
        super().__init__()
        self.response_template = "This is a mock response with random number: {}"
        self.model = "gpt-3.5-turbo"
        self.max_tokens = 1000
        self.allows_batching = True

    def forward(self, arguments):
        results = []
        for argument in arguments:
            input_data = argument.prop.processed_input
            random_number = random.randint(1, 1000)
            mock_response = self.response_template.format(random_number)
            mock_response += f" Input: {input_data}"
            
            delay = random.uniform(0.1, 0.5)
            time.sleep(delay)

            results.append((mock_response, {"usage": {"total_tokens": random.randint(50, 200)}}))
        
        return results

    def prepare(self, argument):
        # Hardcode attributes similar to GPTXChatEngine
        argument.prop.prepared_input = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "This is a mock user input."}
        ]
        argument.prop.raw_input = False
        argument.prop.processed_input = "This is a mock processed input."
        argument.prop.suppress_verbose_output = True
        argument.prop.response_format = None
        argument.prop.payload = None
        argument.prop.examples = []
        argument.prop.prompt = "This is a mock prompt."
        argument.prop.template_suffix = None
        argument.prop.parse_system_instructions = False

        # Add some mock kwargs
        argument.kwargs = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": 0.7,
            "top_p": 1,
            "frequency_penalty": 0,
            "presence_penalty": 0,
        }

    def compute_remaining_tokens(self, prompts: list) -> int:
        return self.max_tokens

@pytest.fixture
def mock_engine():
    return MockGPTXChatEngine()

def test_simple_batch(mock_engine):
    expr = TestExpression()
    inputs = ["test1", "test2", "test3"]
    scheduler = BatchScheduler(expr, num_workers=2, engine=mock_engine, dataset=inputs)
    results = scheduler.run()
    assert len(results) == 3
    assert results[0] == "Processed: test1"
    assert results[1] == "Processed: test2"
    assert results[2] == "Processed: test3"

def test_nested_batch(mock_engine):
    expr = NestedExpression()
    inputs = ["nested1", "nested2"]
    scheduler = BatchScheduler(expr, num_workers=2, engine=mock_engine, dataset=inputs)
    results = scheduler.run()
    assert len(results) == 2
    assert results[0] == "Nested result: Processed: nested1"
    assert results[1] == "Nested result: Processed: nested2"

def test_conditional_batch(mock_engine):
    expr = ConditionalExpression()
    inputs = ["short", "this is a long input"]
    scheduler = BatchScheduler(expr, num_workers=2, engine=mock_engine, dataset=inputs)
    results = scheduler.run()
    assert len(results) == 2
    assert results[0] == "Conditional: Input too short, not processing further"
    assert results[1] == "Conditional: Nested result: Processed: this is a long input"

def test_slow_batch(mock_engine):
    expr = SlowExpression(delay=1)   
    inputs = ["slow1", "slow2"]
    scheduler = BatchScheduler(expr, num_workers=2, engine=mock_engine, dataset=inputs)
    results = scheduler.run()
    assert len(results) == 2
    assert results[0] == "Slow processed (after 1 seconds): slow1"
    assert results[1] == "Slow processed (after 1 seconds): slow2"

def test_double_nested_slow_batch(mock_engine):
    expr = DoubleNestedExpressionSlow()
    inputs = ["input1", "input2"]
    scheduler = BatchScheduler(expr, num_workers=2, engine=mock_engine, dataset=inputs)
    results = scheduler.run()
    assert len(results) == 2
    expected_output = (
        "Double nested results: Processed: input1, "
        "Nested result: Processed: input1, and "
        "Slow processed (after 5 seconds): input1"
    )
    assert results[0] == expected_output.replace("input1", "input1")
    assert results[1] == expected_output.replace("input1", "input2")

def test_simple_batch_variations(mock_engine):
    expr = TestExpression()
    inputs = ["test1", "test2", "test3", "test4", "test5", "test6"]
    
    # Test with batch_size=2 and num_workers=3
    scheduler = BatchScheduler(expr, num_workers=3, batch_size=2, engine=mock_engine, dataset=inputs)
    results = scheduler.run()
    assert len(results) == 6
    assert all(result.startswith("Processed: test") for result in results)
    
    # Test with batch_size=3 and num_workers=2
    scheduler = BatchScheduler(expr, num_workers=2, batch_size=3, engine=mock_engine, dataset=inputs)
    results = scheduler.run()
    assert len(results) == 6
    assert all(result.startswith("Processed: test") for result in results)

def test_nested_batch_variations(mock_engine):
    expr = NestedExpression()
    inputs = ["nested1", "nested2", "nested3", "nested4"]
    
    # Test with batch_size=1 and num_workers=4
    scheduler = BatchScheduler(expr, num_workers=4, batch_size=1, engine=mock_engine, dataset=inputs)
    results = scheduler.run()
    assert len(results) == 4
    assert all(result.startswith("Nested result: Processed: nested") for result in results)
    
    # Test with batch_size=4 and num_workers=1
    scheduler = BatchScheduler(expr, num_workers=1, batch_size=4, engine=mock_engine, dataset=inputs)
    results = scheduler.run()
    assert len(results) == 4
    assert all(result.startswith("Nested result: Processed: nested") for result in results)

def test_conditional_batch_variations(mock_engine):
    expr = ConditionalExpression()
    inputs = ["short", "this is a long input", "short+", "yet another long input"]

    # Test with batch_size=2 and num_workers=2
    scheduler = BatchScheduler(expr, num_workers=2, batch_size=2, engine=mock_engine, dataset=inputs)
    results = scheduler.run()
    assert len(results) == 4
    
    print("Results:", results)  # Add this line to see all results
    
    # Check each result individually
    assert results[0] == "Conditional: Input too short, not processing further", f"Unexpected result for 'short': {results[0]}"
    assert "Conditional: Nested result: Processed: this is a long input" in results[1], f"Unexpected result for 'this is a long input': {results[1]}"
    assert results[2] == "Conditional: Input too short, not processing further", f"Unexpected result for 'another short': {results[2]}"
    assert "Conditional: Nested result: Processed: yet another long input" in results[3], f"Unexpected result for 'yet another long input': {results[3]}"

def test_slow_batch_variations(mock_engine):
    expr = SlowExpression(delay=0.5)  # Reduced delay for faster testing
    inputs = ["slow1", "slow2", "slow3", "slow4", "slow5"]
    
    # Test with batch_size=2 and num_workers=3
    scheduler = BatchScheduler(expr, num_workers=3, batch_size=2, engine=mock_engine, dataset=inputs)
    results = scheduler.run()
    assert len(results) == 5
    assert all(result.startswith("Slow processed (after 0.5 seconds): slow") for result in results)
    
    # Test with batch_size=5 and num_workers=1
    scheduler = BatchScheduler(expr, num_workers=1, batch_size=5, engine=mock_engine, dataset=inputs)
    results = scheduler.run()
    assert len(results) == 5
    assert all(result.startswith("Slow processed (after 0.5 seconds): slow") for result in results)

def test_double_nested_slow_batch_variations(mock_engine):
    expr = DoubleNestedExpressionSlow()
    inputs = ["input1", "input2", "input3"]
    
    # Test with batch_size=1 and num_workers=3
    scheduler = BatchScheduler(expr, num_workers=3, batch_size=1, engine=mock_engine, dataset=inputs)
    results = scheduler.run()
    assert len(results) == 3
    for i, result in enumerate(results, 1):
        assert result == (
            f"Double nested results: Processed: input{i}, "
            f"Nested result: Processed: input{i}, and "
            f"Slow processed (after 5 seconds): input{i}"
        )
    
    # Test with batch_size=3 and num_workers=1
    scheduler = BatchScheduler(expr, num_workers=1, batch_size=3, engine=mock_engine, dataset=inputs)
    results = scheduler.run()
    assert len(results) == 3
    assert all(result.startswith(f"Double nested results: Processed: input") for result in results)

    for i, result in enumerate(results, 1):
        expected = (
            f"Double nested results: Processed: input{i}, "
            f"Nested result: Processed: input{i}, and "
            f"Slow processed (after 5 seconds): input{i}"
        )
        assert result == expected, f"Mismatch for input{i}"

class RandomErrorExpression(Expression):
    def __init__(self, error_rate=0.5, **kwargs):
        super().__init__(**kwargs)
        self.error_rate = error_rate
    
    def forward(self, input, **kwargs):
        if random.random() < self.error_rate:
            raise ValueError("Simulated expression error")
        return f"Processed: {input}"

class MockRandomErrorEngine(MockGPTXChatEngine):
    def __init__(self, error_rate=0.5):
        super().__init__()
        self.error_rate = error_rate

    def forward(self, arguments):
        results = []
        for argument in arguments:
            if random.random() < self.error_rate:
                raise ValueError("Simulated engine error")
            input_data = argument.prop.processed_input
            random_number = random.randint(1, 1000)
            mock_response = self.response_template.format(random_number)
            mock_response += f" Input: {input_data}"
            
            delay = random.uniform(0.1, 0.5)
            time.sleep(delay)

            results.append((mock_response, {"usage": {"total_tokens": random.randint(50, 200)}}))
        
        return results

@pytest.fixture
def mock_random_error_engine():
    return MockRandomErrorEngine()

def test_expression_error_handling(mock_engine):
    expr = RandomErrorExpression(error_rate=0.5)
    inputs = ["test1", "test2", "test3", "test4", "test5"]
    scheduler = BatchScheduler(expr, num_workers=2, engine=mock_engine, dataset=inputs)
    results = scheduler.run()
    
    assert len(results) == 5
    for result in results:
        assert isinstance(result, ValueError) or result.startswith("Processed: test")

def test_engine_error_handling(mock_random_error_engine):
    expr = TestExpression()
    inputs = ["test1", "test2", "test3", "test4", "test5"]
    scheduler = BatchScheduler(expr, num_workers=2, engine=mock_random_error_engine, dataset=inputs)
    results = scheduler.run()
    
    assert len(results) == 5
    for result in results:
        assert isinstance(result, ValueError) or result.startswith("Processed: test")

def test_mixed_error_handling(mock_random_error_engine):
    expr = RandomErrorExpression(error_rate=0.3)
    inputs = ["test1", "test2", "test3", "test4", "test5", "test6", "test7"]
    scheduler = BatchScheduler(expr, num_workers=3, batch_size=2, engine=mock_random_error_engine, dataset=inputs)
    results = scheduler.run()
    
    assert len(results) == 7
    for result in results:
        assert isinstance(result, ValueError) or result.startswith("Processed: test")
