import pandas as pd
import numpy as np
import pytest
import sys
sys.path.insert(0, '../olscheck')
import ols_assumptions_check

@pytest.fixture
def setup_data_residuals():
    data = pd.DataFrame({
        'y_true': [1, 2, 2, 4, 5],
        'y_pred': [1, 2, 3, 4, 5],
    })
    return data

def test_compute_residuals(setup_data_residuals):
    ols_checker = ols_assumptions_check.OlsCheck()
    result = ols_checker._compute_residuals(setup_data_residuals, 'y_true', 'y_pred')
    expected_residuals = np.array([0, 0, -1, 0, 0])
    # Check if residuals match the expected values
    assert np.array_equal(result['residuals'].to_numpy(), expected_residuals)


    