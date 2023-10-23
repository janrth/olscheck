import pandas as pd
import numpy as np
import pytest
import sys
from statsmodels.tools.tools import add_constant
sys.path.insert(0, '../olscheck')
import ols_assumptions_check

@pytest.fixture
def setup_data_cooks_distance():
    data = pd.DataFrame({
                'x1': [2, 0, 1, 1, 1],
                'x2': [-1, 2, .5, 2.5, 3],
                'residuals': [0, 0, -1, 0, 0]
    })
    return data

leverage = np.array([0.82352941, 0.21960784, 0.14313725, 0.34705882, 0.46666667])

def test_cooks_distance(setup_data_cooks_distance):
    ols_checker = ols_assumptions_check.OlsCheck()
    cooks_distance = ols_checker._cooks_distance(setup_data_cooks_distance, leverage, ['x1','x2'])
    expected_cooks_distance = np.array([0. , 0. , 0.29242965, 0. , 0. ])
    # Check if residuals match the expected values
    assert np.allclose(cooks_distance.values, expected_cooks_distance, atol=1e-8)