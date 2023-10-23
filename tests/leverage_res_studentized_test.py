import pandas as pd
import numpy as np
import pytest
import sys
from statsmodels.tools.tools import add_constant
sys.path.insert(0, '../olscheck')
import ols_assumptions_check

@pytest.fixture
def setup_data_residuals_studentized():
    data = pd.DataFrame({
        'x1': [2, 0, 1, 1, 1],
        'x2': [-1, 2, .5, 2.5, 3],
        'residuals': [0, 0, -1, 0, 0]
    })
    return data

# Define your expected values based on constant value
expected_values = {
    True: (np.array([0.83064516, 0.83064516, 0.33064516, 0.39516129, 0.61290323]), 
           np.array([0. ,  0. , -2.23606798, 0. ,  0.])),
    False: (np.array([0.82352941, 0.21960784, 0.14313725, 0.34705882, 0.46666667]), 
            np.array([0. ,  0. , -2.23606798, 0. ,  0.]))
}

# Parametrizes the test to run with both constant=True and constant=False
@pytest.mark.parametrize("constant", [True, False])
def test_residuals_studentized(setup_data_residuals_studentized, constant):
    ols_checker = ols_assumptions_check.OlsCheck()
    leverage, res_studentized_internal = ols_checker._residuals_studentized_internal(setup_data_residuals_studentized, 
                                                         ['x1', 'x2'],
                                                         constant=constant)
    expected_leverage, expected_res_studentized_internal = expected_values[constant]

    assert np.allclose(leverage, expected_leverage, atol=1e-8)
    assert np.allclose(res_studentized_internal.to_numpy(), expected_res_studentized_internal, atol=1e-8)
