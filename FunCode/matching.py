import pandas as pd
from pathlib import Path
import os
import itertools
import string
from itertools import product
import numpy as np
from pykeepass import PyKeePass
import io
import msoffcrypto
import openpyxl
from typing import Union

def make_excel_combinations():
    combinations = []

    # Generate combinations for lengths 1 to 4
    for length in range(1, 5):
        # Generate combinations for the current length
        for combination in itertools.product(list(string.ascii_uppercase), repeat=length):
            combinations.append(''.join(combination))
    return(combinations)

def convert_num_toexcel_col(num, combinations=make_excel_combinations()):
    if num < 0:
        raise ValueError("Number must be > 0")
    return(combinations[num])

def find_non_identical_columns(num_row1:int, num_row2:int, df:pd.DataFrame, reduced: bool=False, combinations: list=make_excel_combinations()):
    """Given two row numbers in a dataframe it iterates over all columns and returns columns that are different between the two rows, and where at least one of them is 
    a non-NaN value.

    Args:
        num_row1 (int): Index of row number 1.
        num_row2 (int): Index of row number 2.
        df (pd.DataFrame): Dataframe that contains information for both rows.
        reduced (bool): If True, dataset returned doesn't show row values, but only col index (in original df), col name, and excel position.
        combinations (list, optional): Alphabetical combination for excel sheets.

    Returns:
        - pd.DataFrame: A pandas dataframe with 5 columns containing the column index, name, excel position, value for first row, and value for second row.
        - pd.DataFrame: A pandas datadrame with 3 columns containing the column index of the original df, column name and excel position.
    """
    non_identical_columns = []
    row1=df.iloc[num_row1]
    row2=df.iloc[num_row2]
    for i, (col1, col2) in enumerate(zip(row1, row2)):
        if col1 != col2:
            col_name=df.columns[i]
            excel_col=convert_num_toexcel_col(i, combinations)
            non_identical_columns.append([i, col_name, excel_col, col1, col2])

    non_identical_columns=pd.DataFrame(non_identical_columns, columns=["col_num", "col_name", "excel_col", f"valueRow{num_row1}", f"ValueRow{num_row2}"])
    non_identical_columns=non_identical_columns[
        ~non_identical_columns[f"valueRow{num_row1}"].isna() |
        ~non_identical_columns[f"ValueRow{num_row2}"].isna()
    ]
    if reduced:
        non_identical_columns=non_identical_columns.loc[:, non_identical_columns.columns.isin(["col_num", "col_name", "excel_col"])]
        
    return non_identical_columns

def hamming_distance(s1, s2):
    """Calculate the Hamming distance between two strings."""
    return sum(c1 != c2 for c1, c2 in zip(s1, s2))

def find_similar_identifiers(id1, id2_list, dist = 1):
    """Find identifiers in id2_list similar to id1."""
    similar_identifiers = []
    for id2 in id2_list:
        if hamming_distance(id1, id2) == dist:
            similar_identifiers.append(id2)
    return similar_identifiers

def find_non_identical_columns(num_row1:int, num_row2:int, df:pd.DataFrame, reduced: bool=False, combinations: list=make_excel_combinations()):
    """Given two row numbers in a dataframe it iterates over all columns and returns columns that are different between the two rows, and where at least one of them is 
    a non-NaN value.

    Args:
        num_row1 (int): Index of row number 1.
        num_row2 (int): Index of row number 2.
        df (pd.DataFrame): Dataframe that contains information for both rows.
        reduced (bool): If True, dataset returned doesn't show row values, but only col index (in original df), col name, and excel position.
        combinations (list, optional): Alphabetical combination for excel sheets.

    Returns:
        pd.DataFrame: A pandas dataframe with 5 columns containing the column index, name, excel position, value for first row, and value for second row.
    """
    non_identical_columns = []
    row1=df.iloc[num_row1]
    row2=df.iloc[num_row2]
    for i, (col1, col2) in enumerate(zip(row1, row2)):
        if col1 != col2:
            col_name=df.columns[i]
            excel_col=convert_num_toexcel_col(i, combinations)
            non_identical_columns.append([i, col_name, excel_col, col1, col2])

    non_identical_columns=pd.DataFrame(non_identical_columns, columns=["col_num", "col_name", "excel_col", f"valueRow{num_row1}", f"ValueRow{num_row2}"])
    non_identical_columns=non_identical_columns[
        ~non_identical_columns[f"valueRow{num_row1}"].isna() |
        ~non_identical_columns[f"ValueRow{num_row2}"].isna()
    ]
    if reduced:
        non_identical_columns=non_identical_columns.loc[:, non_identical_columns.columns.isin(["col_num", "col_name", "excel_col"])]
        
    return non_identical_columns

def get_keypass_password(pwd_db_path:Path, pw_file:Path, key_file_path:Path, entry_name:str) ->  str:
    """Function allows to access passwords in a KeyPass database.

    Args:
        pwd_db_path (Path): Path to KeyPass database.
        pw_file (Path): Path to file containing password to open KeyPass database.
        key_file_path (Path): Path to key file to open KeyPass databse.
        entry_name (str): Name of entry, normally corresponds to name of the file.

    Returns:
        str: Password to the entry.
    """
    with open(pw_file, "r") as file:
        db_psswd=file.read().strip()
    kp=PyKeePass(pwd_db_path, password=db_psswd, keyfile=key_file_path)
    entry=kp.find_entries(title=entry_name, first=True)
    if entry:
        return entry.password
    else:
        print(f"No entry found for {entry_name} in this db")
        return None


def read_encrypted_excel(pwd_db_path:Path, pw_file:Path, key_file_path:Path, file_path:str, file_name:str) -> pd.DataFrame:
    """
    Function read excel files that are encryped with password

    Args:
        pwd_db_path (Path): Path to KeyPass database.
        pw_file (Path): Path to file containing password to open KeyPass database.
        key_file_path (Path): Path to key file to open KeyPass databse.
        file_path (str): Path to encryped file containing db.
        file_name (str): Name of encryped file. It must be the same name as the entry in pwd_db
    
    Returns:
        pd.DataFrame: Pandas data frame of encrypted file

    """

    file_pw=get_keypass_password(pwd_db_path=pwd_db_path, pw_file=pw_file, key_file_path=key_file_path, entry_name=file_name)

    decrypted_file = io.BytesIO()
    
    # Decrypt file
    with open(os.path.join(file_path, file_name), "rb") as enc_file:
        file=msoffcrypto.OfficeFile(enc_file)
        file.load_key(password=file_pw)
        file.decrypt(decrypted_file)

    decrypted_db=pd.read_excel(decrypted_file)

    return decrypted_db

def map_str_to_value(df: pd.DataFrame, col_names:list, eq_dict: dict) -> pd.DataFrame:
    """Function takes a pandas data frame, the names of some columns in the dataframe and maps
    values in the columns to values present in the eq_dict.

    Args:
        df (pd.DataFrame): Pandas dataframe must contain column names.
        col_names(list): list of column names to be modified.
        eq_dict (dict): equivalence of values in columns to be modified. key-value pairs must correspond to:
        old-value: new-value for each value present in the columns.

    Returns:
        pd.DataFrame: Data frame with mapped values.
    """
    for col in col_names:
        df.loc[:, col] = df[col].map(eq_dict)
    return df

def rename_columns(df: pd.DataFrame, eq_dic:dict) -> pd.DataFrame:
    """_summary_

    Args:
        df (pd.DataFrame): Pandas dataframe, must contain column names.
        eq_dic (dict): equivalence of columns to be renames in the format

    Returns:
        pd.DataFrame: Pandas data frame with renamed column names.
    """
    df.rename(columns=eq_dic, inplace=True)
    return df

def sex_from_pn(number:Union[str,int]) -> str:
    """Function takes a personnummer and extracts the second to last number
    to infer biological sex.

    Args:
        number (Union[str,int]): A number that contains at least 1 digit.

    Returns:
        str: Returns "Kvinna" if digit is even, and "Man" if digit is odd.
    """
    assert len(str(number)) > 1, f"{number} must contain at least 1 digit"
    second_last_digit=str(number)[-2]
    if int(second_last_digit) %2 == 0:
        return "Kvinna"
    else:
        return "Man"

def apply_row_by_row(df: pd.DataFrame, col_name: str, function) -> pd.DataFrame:
    """Function used to apply a function row-by-row in a dataframe.

    Args:
        df (pd.DataFrame): Data Frame where function is to be applied.
        col_name (str): column name where function is to be applied.
        function (function): Function name to be applied.

    Returns:
        pd.DataFrame: Dataframe where function has been applied in all rows of col_name.
    """
    return df[col_name].apply(function)


