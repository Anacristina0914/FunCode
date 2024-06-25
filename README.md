# FunCode

This package provides various utilities for file manipulation and plotting simple statistics. It includes functions to handle encrypted Excel files, manipulate DataFrames, and generate plots.

## Table of Contents

- [FunCode](#funcode)
  - [Table of Contents](#table-of-contents)
  - [Installation](#installation)
  - [Usage](#usage)
  - [modules](#modules)
    - [plotting](#plotting)
    - [matching](#matching)
  - [Examples](#examples)

## Installation

To install the package, you can clone the repository:

```bash
git clone https://github.com/Anacristina0914/FunCode.git
cd FunCode
pip install -e .
```

## Usage
```python
from FunCode import matching, plotting
matching.function_name(args*)
plotting.function_name(args*)
```

## modules
Functions on each module.
### plotting
- plot_plate_positions
- make_barplot_sample_dist
- make_barplot_sample_dist_twogroups

### matching
- make_excel_combinations
- convert_num_toexcel_col
- find_non_identical_columns
- hamming_distance
- find_similar_identifiers
- find_non_identical_columns
- get_keypass_password
- read_encrypted_excel
- map_str_to_value
- rename_columns
- sex_from_pn
- apply_sex_from_pn

 
## Examples
```python
from FunCode import matching, plotting

# See all arguments for a function
help(matching.function_name)

# Read encrypted excel file using pw database and a key file
encrypted_excel = matching.read_encrypted_excel(pwd_db_path=pw_db, pw_file=pw_file, key_file_path=key_path,
                                                           file_path=database_dir, file_name=sle_fulldb_name)
# Plot sample dist using a column in a df
plotting.make_barplot_sample_dist(db, group_col="col_name", plot_title="sample dist col_name", 
                                    y_axis_title="counts", x_axis_title="")
```

