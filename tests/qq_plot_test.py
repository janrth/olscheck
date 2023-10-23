import pytest
import pandas as pd
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '../olscheck')
import ols_assumptions_check

@pytest.fixture
def setup_data_qq_plot():
    data = pd.DataFrame({
        'residuals': [0.1, -0.2, 0.15, -0.3, 0.05],
        'x1': [2, 0, 1, 1, 1], 
        'x2': [-1.0, 2.0, 0.5, 2.5, 3.0]
    })
    return data

def test_qq_plot(setup_data_qq_plot):
    ols_checker = ols_assumptions_check.OlsCheck()
    ax = ols_checker.qq_plot(setup_data_qq_plot, ['x1', 'x2'])

    # Check title and labels
    assert ax.get_title() == 'Normal Q-Q', "Title does not match expectation"
    assert ax.get_xlabel() == 'Theoretical Quantiles', "X label does not match expectation"
    assert ax.get_ylabel() == 'Standardized Residuals', "Y label does not match expectation"
    
    # Assuming that ax.lines captures the 45-degree line
    assert len(ax.lines) == 2, "Expected 1 line (45-degree line)"

