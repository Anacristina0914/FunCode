import pytest
import pandas as pd
from pathlib import Path
from FunCode.matching import convert_num_toexcel_col, find_non_identical_columns, sex_from_pn, apply_sex_from_pn

# Mock data for testing
@pytest.fixture
def sample_dataframe():
    data = {
        'A': [1, 2, 3],
        'B': [4, 5, 6],
        'C': [7, 8, 9]
    }
    return pd.DataFrame(data)

def test_convert_num_toexcel_col():
    assert convert_num_toexcel_col(0) == 'A'
    assert convert_num_toexcel_col(25) == 'Z'
    assert convert_num_toexcel_col(26) == 'AA'

#def test_find_non_identical_columns(sample_dataframe):
#    df = sample_dataframe
#    result = find_non_identical_columns(0, 1, df)
#    assert isinstance(result, pd.DataFrame)
#    assert result.shape == (3, 5)  

def test_sex_from_pn():
    assert sex_from_pn(1234567860) == 'Kvinna'
    assert sex_from_pn(1234567891) == 'Man'

def test_apply_sex_from_pn(sample_dataframe):
    sample_dataframe['nummer'] = [1234567860, 1234567891, 1234567802]
    result = apply_sex_from_pn(sample_dataframe, 'nummer')
    assert isinstance(result, pd.Series)
    assert result.tolist() == ['Kvinna', 'Man', 'Kvinna']  


