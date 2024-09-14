import pytest
from unittest.mock import Mock, patch
from symai.functional import _process_query_single, ConstraintViolationException

@pytest.fixture
def mock_setup():
    mock_engine = Mock()
    mock_instance = Mock()
    mock_func = Mock()
    mock_argument = Mock()
    mock_argument.prop = Mock()
    mock_argument.kwargs = {}
    return mock_engine, mock_instance, mock_func, mock_argument

@patch('symai.functional._prepare_argument')
@patch('symai.functional._apply_preprocessors')
@patch('symai.functional._postprocess_response')
@patch('symai.functional._limit_number_results')
def test_successful_query(mock_limit, mock_postprocess, mock_preprocess, mock_prepare, mock_setup):
    mock_engine, mock_instance, mock_func, mock_argument = mock_setup
    
    # Setup
    mock_prepare.return_value = mock_argument
    mock_preprocess.return_value = "preprocessed input"
    mock_postprocess.return_value = ("result", {"metadata": "value"})
    mock_limit.return_value = "limited result"
    mock_argument.kwargs['executor_callback'] = Mock(return_value=("raw_result", {"metadata": "value"}))

    # Execute
    result = _process_query_single(
        mock_engine, mock_instance, mock_func,
        constraints=[], default=None, limit=1, trials=1,
        pre_processors=None, post_processors=None, argument=mock_argument
    )

    # Assert
    assert result == "limited result"
    mock_prepare.assert_called_once()
    mock_preprocess.assert_called_once()
    mock_engine.prepare.assert_called_once_with(mock_argument)
    mock_postprocess.assert_called_once()
    mock_limit.assert_called_once()

@patch('symai.functional._prepare_argument')
@patch('symai.functional._apply_preprocessors')
@patch('symai.functional._execute_query_fallback')
def test_query_with_fallback(mock_fallback, mock_preprocess, mock_prepare, mock_setup):
    mock_engine, mock_instance, mock_func, mock_argument = mock_setup
    
    # Setup
    mock_prepare.return_value = mock_argument
    mock_preprocess.return_value = "preprocessed input"
    mock_argument.kwargs['executor_callback'] = Mock(side_effect=Exception("Query failed"))
    mock_fallback.return_value = "fallback result"

    # Execute
    result = _process_query_single(
        mock_engine, mock_instance, mock_func,
        constraints=[], default="default", limit=1, trials=1,
        pre_processors=None, post_processors=None, argument=mock_argument
    )

    # Assert
    assert result == "fallback result"
    mock_fallback.assert_called_once_with(mock_func, mock_instance, mock_argument, "default")

@patch('symai.functional._prepare_argument')
@patch('symai.functional._apply_preprocessors')
def test_query_failure_with_no_fallback(mock_preprocess, mock_prepare, mock_setup):
    mock_engine, mock_instance, mock_func, mock_argument = mock_setup
    
    # Setup
    mock_prepare.return_value = mock_argument
    mock_preprocess.return_value = "preprocessed input"
    mock_argument.kwargs['executor_callback'] = Mock(side_effect=Exception("Query failed"))

    # Execute and Assert
    with pytest.raises(Exception):
        _process_query_single(
            mock_engine, mock_instance, mock_func,
            constraints=[], default=None, limit=1, trials=1,
            pre_processors=None, post_processors=None, argument=mock_argument
        )