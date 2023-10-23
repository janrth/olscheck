import pandas as pd
import pytest
import sys
sys.path.insert(0, '../olscheck')
import ols_assumptions_check

def test_histogram_residuals_mock(mocker):
    ols_checker = ols_assumptions_check.OlsCheck()
    data = pd.DataFrame({
        'residuals': [1, 2, 3, 4, 5]
    })

    mock_histplot = mocker.patch("seaborn.histplot")
    ols_checker.histogram_residuals(data)
    assert mock_histplot.call_args[0][0].equals(data['residuals'])
