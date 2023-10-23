import pytest
import matplotlib.pyplot as plt
import pandas as pd
import sys
sys.path.insert(0, '../olscheck')
import ols_assumptions_check

# Create a fixture for sample data
@pytest.fixture
def setup_data_residual_plot():
    data = pd.DataFrame({
        'residuals': [0, 0, -1, 0, 0],
        'y_pred': [1, 2, 3, 4, 5]
    })
    return data

def test_residual_plot(setup_data_residual_plot):
    ols_checker = ols_assumptions_check.OlsCheck()
    ax = ols_checker.residual_plot(setup_data_residual_plot, 'y_pred')

    # Check title and labels
    assert ax.get_title() == 'Residuals vs Fitted', "Title does not match expectation"
    assert ax.get_xlabel() == 'Fitted values', "X label does not match expectation"
    assert ax.get_ylabel() == 'Residuals', "Y label does not match expectation"

    # There's 1 scatter plot (residuals vs fitted) and 1 line (lowess)
    assert len(ax.collections) == 1, "Expected 1 scatter plot"
    assert len(ax.lines) == 2, "Expected 1 lowess line and 1 horizontal line at zero"

    # Check if the method returns the expected type
    assert isinstance(ax, plt.Axes)
