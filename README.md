# FunCode

This package provides various utilities for file manipulation and plotting simple statistics. It includes functions to handle encrypted Excel files, manipulate DataFrames, and generate plots.

## Table of Contents

- [FunCode](#funcode)
  - [Table of Contents](#table-of-contents)
  - [Installation](#installation)
  - [Usage](#usage)
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

## Examples
```python
from FunCode import matching, plotting
# Read encrypted excel file using pw database and a key file
encrypted_excel = matching.read_encrypted_excel(pwd_db_path=pw_db, pw_file=pw_file, key_file_path=key_path,
                                                           file_path=database_dir, file_name=sle_fulldb_name)
# Plot sample dist using a column in a df
plotting.make_barplot_sample_dist(db, group_col="col_name", plot_title="sample dist col_name", 
                                    y_axis_title="counts", x_axis_title="")
```
