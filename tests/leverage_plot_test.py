import pandas as pd
import pytest
import sys
from statsmodels.tools.tools import add_constant
import matplotlib.pyplot as plt
sys.path.insert(0, '../olscheck')
import ols_assumptions_check

@pytest.fixture
def setup_data_leverage_plot():
    data = pd.DataFrame({
                'x1': [2, 0, 1, 1, 1],
                'x2': [-1, 2, .5, 2.5, 3],
                'residuals': [0, 0, -1, 0, 0]
    })
    return data

def test_leverage_plot_return_type(setup_data_leverage_plot):
    ols_checker = ols_assumptions_check.OlsCheck()
    ax = ols_checker.leverage_plot(setup_data_leverage_plot, ['x1','x2'])
    assert isinstance(ax, plt.Axes)

def test_leverage_plot_elements(setup_data_leverage_plot):
    ols_checker = ols_assumptions_check.OlsCheck()
    ax = ols_checker.leverage_plot(setup_data_leverage_plot, ['x1','x2'])
    
    # Check title, labels, and legend
    assert ax.get_title() == 'Residuals vs Leverage', "Title does not match expectation"
    assert ax.get_xlabel() == 'Leverage', "X label does not match expectation"
    assert ax.get_ylabel() == 'Standardized Residuals', "Y label does not match expectation"
    assert ax.get_legend() is not None, "Legend is missing"

    # Assuming 1 scatter plot, 1 regplot line, and 2 Cook's distance lines
    assert len(ax.collections) == 1, "Expected 1 scatter plot"
    assert len(ax.lines) == 4, "Expected 3 lines (regplot + 2 Cook's distance lines + horizontal line)"

