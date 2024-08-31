import pytest
import sys, os
sys.path.append('/Users/aynur/aynur/projects/')
from symbolicai.symai.functional import _apply_preprocessors
from symbolicai.symai.pre_processors import PreProcessor
from types import SimpleNamespace

class MockArgument:
    def __init__(self, raw_input=False, args=None):
        self.prop = SimpleNamespace(raw_input=raw_input)
        self.args = args or []

class MockPreProcessor(PreProcessor):
    def __call__(self, argument):
        return f"Processed: {argument.prop.instance}"

@pytest.mark.parametrize("raw_input,args,pre_processors,instance,expected", [
    (False, [], [MockPreProcessor()], "test_instance", "Processed: test_instance"),
    (True, ["arg1", "arg2"], None, "test_instance", "test_instance"),
    (False, [], None, "test_instance", ""),
    (False, ["arg1", "arg2"], None, "test_instance", "test_instance"),
    (False, [], [], "test_instance", "test_instance"),  # Empty list of pre-processors
])
def test_apply_preprocessors(raw_input, args, pre_processors, instance, expected):
    argument = MockArgument(raw_input=raw_input, args=args)
    result = _apply_preprocessors(argument, instance, pre_processors)
    assert result == expected

def test_apply_preprocessors_multiple():
    class AnotherMockPreProcessor(PreProcessor):
        def __call__(self, argument):
            return f"Also processed: {argument.prop.instance}"

    argument = MockArgument(raw_input=False)
    pre_processors = [MockPreProcessor(), AnotherMockPreProcessor()]
    result = _apply_preprocessors(argument, "test_instance", pre_processors)
    assert result == "Processed: test_instanceAlso processed: test_instance"

def test_apply_preprocessors_none_return():
    class NoneReturnPreProcessor(PreProcessor):
        def __call__(self, argument):
            return None

    argument = MockArgument(raw_input=False)
    pre_processors = [NoneReturnPreProcessor(), MockPreProcessor()]
    result = _apply_preprocessors(argument, "test_instance", pre_processors)
    assert result == "Processed: test_instance"

def test_apply_preprocessors_modifying():
    class ModifyingPreProcessor(PreProcessor):
        def __call__(self, argument):
            argument.prop.instance = "modified_" + argument.prop.instance
            return argument.prop.instance

    argument = MockArgument(raw_input=False)
    pre_processors = [ModifyingPreProcessor(), MockPreProcessor()]
    result = _apply_preprocessors(argument, "test_instance", pre_processors)
    assert result == "Processed: modified_test_instance"

def test_apply_preprocessors_order():
    class FirstPreProcessor(PreProcessor):
        def __call__(self, argument):
            return f"First: {argument.prop.instance}"

    class SecondPreProcessor(PreProcessor):
        def __call__(self, argument):
            return f"Second: {argument.prop.instance}"

    argument = MockArgument(raw_input=False)
    pre_processors = [FirstPreProcessor(), SecondPreProcessor()]
    result = _apply_preprocessors(argument, "test_instance", pre_processors)
    assert result == "Second: First: test_instance"

def test_apply_preprocessors_exception():
    class ExceptionPreProcessor(PreProcessor):
        def __call__(self, argument):
            raise ValueError("Test exception")

    argument = MockArgument(raw_input=False)
    pre_processors = [ExceptionPreProcessor(), MockPreProcessor()]
    with pytest.raises(ValueError, match="Test exception"):
        _apply_preprocessors(argument, "test_instance", pre_processors)

def test_apply_preprocessors_performance():
    class NoOpPreProcessor(PreProcessor):
        def __call__(self, argument):
            return argument.prop.instance

    argument = MockArgument(raw_input=False)
    pre_processors = [NoOpPreProcessor() for _ in range(1000)]
    result = _apply_preprocessors(argument, "test_instance", pre_processors)
    assert result == "test_instance"