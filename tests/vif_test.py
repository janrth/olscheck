import pandas as pd
import numpy as np
import pytest
import sys
sys.path.insert(0, '../olscheck')
import ols_assumptions_check

@pytest.fixture
def setup_data_vif():
    data = pd.DataFrame({
        'feature1': [1, 2, 2, 4, 5],
        'feature2': [5, 1, 3, 2, 1],
    })
    return data

def test_vif_test(setup_data_vif):
    ols_checker = ols_assumptions_check.OlsCheck()
    result = ols_checker.vif_test(setup_data_vif)

    assert isinstance(result, pd.DataFrame)
    assert 'variables' in result.columns
    assert 'vif' in result.columns
    assert np.round(result.vif.iloc[1],3) == 1.914
