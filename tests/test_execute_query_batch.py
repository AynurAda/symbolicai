import pytest
from unittest.mock import Mock, patch
import sys, os
sys.path.append('/Users/aynur/aynur/projects/')
from symbolicai.symai.functional import _execute_query_batch, ProbabilisticBooleanMode, ConstraintViolationException

@pytest.fixture
def mock_engine():
    return Mock()

@pytest.fixture
def mock_argument():
    argument = Mock()
    argument.prop = Mock()
    argument.prop.preview = False
    argument.prop.raw_output = False
    return argument

def test_execute_query_batch_normal(mock_engine, mock_argument):
    mock_engine.return_value = ([["response1", "response2"]], {"metadata": "test"})
    post_processors = [Mock(side_effect=lambda x, _: x.upper())]
    
    result, metadata = _execute_query_batch(mock_engine, post_processors, str, mock_argument)
    
    assert result == ["RESPONSE1", "RESPONSE2"]
    assert metadata == {"metadata": "test"}
    mock_engine.prepare.assert_called_once_with(mock_argument)

def test_execute_query_batch_preview(mock_engine, mock_argument):
    mock_argument.prop.preview = True
    mock_engine.preview.return_value = "preview_result"
    
    result, metadata = _execute_query_batch(mock_engine, None, str, mock_argument)
    
    assert result == "preview_result"
    assert metadata == {}
    mock_engine.preview.assert_called_once_with(mock_argument)

def test_execute_query_batch_raw_output(mock_engine, mock_argument):
    mock_argument.prop.raw_output = True
    mock_engine.return_value = (None, {"raw_output": "raw_result"})
    
    result = _execute_query_batch(mock_engine, None, str, mock_argument)
    
    assert result == "raw_result"

def test_execute_query_batch_constraint_violation(mock_engine, mock_argument):
    mock_engine.return_value = ([["response"]], {})
    constraint = Mock(return_value=False)
    mock_argument.prop.constraints = [constraint]
    
    with pytest.raises(ConstraintViolationException):
        _execute_query_batch(mock_engine, None, str, mock_argument)

@patch('symai.functional._cast_return_type')
def test_execute_query_batch_return_type_casting(mock_cast, mock_engine, mock_argument):
    mock_engine.return_value = ([["response"]], {})
    mock_cast.return_value = 42
    
    result, _ = _execute_query_batch(mock_engine, None, int, mock_argument)
    
    assert result == [42]
    mock_cast.assert_called_once_with("response", int, ProbabilisticBooleanMode.MEDIUM)

def test_execute_query_batch_multiple_post_processors(mock_engine, mock_argument):
    mock_engine.return_value = ([["response"]], {})
    post_processors = [
        Mock(side_effect=lambda x, _: x.upper()),
        Mock(side_effect=lambda x, _: x + "!")
    ]
    
    result, _ = _execute_query_batch(mock_engine, post_processors, str, mock_argument)
    
    assert result == ["RESPONSE!"]

def test_execute_query_batch_empty_response(mock_engine, mock_argument):
    mock_engine.return_value = ([[]], {})
    
    result, metadata = _execute_query_batch(mock_engine, None, str, mock_argument)
    
    assert result == []
    assert metadata == {}