import pytest
import pandas as pd
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '../olscheck')
import ols_assumptions_check

@pytest.fixture
def setup_data_scale_location_plot():
    return pd.DataFrame({
        'y_pred': [1, 2, 3, 4, 5],
        'residuals': [0.1, -0.2, 0.15, -0.3, 0.05],
        'x1': [2, 0, 1, 1, 1], 
        'x2': [-1.0, 2.0, 0.5, 2.5, 3.0]
    })

def test_scale_location_plot(setup_data_scale_location_plot):
    ols_checker = ols_assumptions_check.OlsCheck()
    ax = ols_checker.scale_location_plot(setup_data_scale_location_plot, 'y_pred', ['x1', 'x2'])

    # Check title and labels
    assert ax.get_title() == 'Scale-Location', "Title does not match expectation"
    assert ax.get_xlabel() == 'Fitted values', "X label does not match expectation"
    assert ax.get_ylabel() == r'$\sqrt{|\mathrm{Standardized\ Residuals}|}$', "Y label does not match expectation"
    
    # There's 1 scatter plot and 1 lowess line (also considering a possible horizontal line at zero)
    assert len(ax.collections) == 1, "Expected 1 scatter plot"
    assert len(ax.lines) >= 1, "Expected at least 1 line (lowess)"
