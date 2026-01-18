import pytest
import pandas as pd
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '../olscheck')
import ols_assumptions_check

@pytest.fixture
def setup_data_ols_assumptions():
    data = pd.DataFrame({
        'y_true': [1, 2, 2, 4, 5],
        'y_pred': [1, 2, 3, 4, 5],
        'x1': [2, 0, 1, 1, 1], 
        'x2': [-1.0, 2.0, 0.5, 2.5, 3.0]
    })
    return data

def test_ols_assumptions(mocker, setup_data_ols_assumptions):
    # Mock print, plt.show, and plt.style.context
    mocker.patch('builtins.print')
    mocker.patch('matplotlib.pyplot.show')
    mock_style_context = mocker.patch('matplotlib.pyplot.style.context')
    
    ols_checker = ols_assumptions_check.OlsCheck()

    # Call the main function
    ols_checker.ols_assumptions(setup_data_ols_assumptions, 'y_true', 'y_pred', ['x1', 'x2'])

    # Check if the plotting context manager was used
    expected_style = ols_assumptions_check._resolve_style('seaborn-paper')
    mock_style_context.assert_called_once_with(expected_style)

    # Check if there are 4 subplots
    axes = plt.gcf().axes  # gcf() gets the current figure, and .axes gives a list of its axes (subplots)
    assert len(axes) == 4, f"Expected 4 subplots, but found {len(axes)}"


