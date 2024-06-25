import pytest
import pandas as pd
from pathlib import Path
from FunCode.matching import convert_num_toexcel_col, find_non_identical_columns, sex_from_pn, apply_row_by_row, hamming_distance, find_similar_identifiers

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

@pytest.mark.parametrize("row1, row2, reduced, output_len", [
    (1, 2, True, 3),
    (1, 2, False, 5),
])
def test_find_non_identical_columns(row1, row2, reduced, output_len, sample_dataframe):
    df = sample_dataframe
    result = find_non_identical_columns(df = df, num_row1 = row1, num_row2 = row2, reduced = reduced)
    print(result)
    assert isinstance(result, pd.DataFrame)
    assert result.shape[1] == output_len  

@pytest.mark.parametrize("num, expected", [
    (1234567860, "Kvinna"),
    (1234567891, "Man"),
    (9818246141, "Kvinna"),
])
def test_sex_from_pn(num, expected):
    assert sex_from_pn(num) == expected

def test_sex_from_pn_invalid():
    with pytest.raises(AssertionError, match = "8 must contain at least 1 digit"):
        sex_from_pn(8)

def test_apply_row_by_rpw(sample_dataframe):
    sample_dataframe['nummer'] = [1234567860, 1234567891, 1234567802]
    result = apply_row_by_row(sample_dataframe, 'nummer', sex_from_pn)
    assert isinstance(result, pd.Series)
    assert result.tolist() == ['Kvinna', 'Man', 'Kvinna']  

@pytest.mark.parametrize("s1, s2, dist", [
    ("ACCT", "AGCT", 1),
    ("AATT", "AGTA", 2),
])
def test_hamming_distance(s1, s2, dist):
    hdist = hamming_distance(s1, s2)
    assert hdist == dist

@pytest.mark.parametrize("id, id_list, hdist, exp_len", [
    ("882964", ["882864", "998855"], 1, 1),
    ("998021", ["888021", "218021"], 2, 2),
    ("802121", ["912121", "882120"], 1, 0),
])

def test_find_similar_identifiers(id, id_list, hdist, exp_len):
    output = find_similar_identifiers(id, id_list, hdist)
    assert isinstance(output, list)
    assert len(output) == exp_len

#TODO write tests for map_str_to_value, rename_columns