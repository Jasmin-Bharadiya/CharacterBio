
# CureData Project

The `CureData` project is a Python-based solution for generating summary statistics, performing regression analysis, and validating data for demographic, phenotype, and PRS datasets. It provides various utility functions to process and analyze data, and integrates with libraries like `pandas`, `pyarrow`, `statsmodels`, `seaborn`, and others. This project is also designed to version the data processing pipeline and include automated testing for validating key functionalities.

## Features

- **Versioning System**: Increment and manage the version of the analysis pipeline with automated version updates.
- **Data Loading**: Load demographic, phenotype, and PRS data in various formats (`.tsv`, `.parquet`).
- **Summary Statistics**: Generate summary statistics like mean, median, standard deviation, missing rate, and average age for different phenotypes.
- **Regression Analysis**: Perform regression analysis to investigate the relationship between PRS scores and phenotype values.
- **Data Validation**: Validate the consistency of summary statistics using Pydantic data models.
- **Plotting**: Visualize significant regression results using bar plots.
- **Testing**: Unit tests to validate the correctness of each module and function.

## Project Structure

```
CureData/
│
├── cureData.py              # Main script containing data processing and analysis functions
├── test_cureData.py         # Unit tests for validating functionality
├── version.txt              # Version file for the project
│
├── requirements.txt         # List of Python dependencies
└── README.md                # This file
```

## Installation

To get started, clone the repository and install the required dependencies.

1. Clone the repository:
    ```bash
    git clone <repository_url>
    cd CureData
    ```

2. Create and activate a virtual environment (optional but recommended):
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows use venv\Scripts\activate
    ```

3. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### Command-line Interface (CLI)

You can run the main script from the command line to generate summary statistics, perform regression analysis, and save results.

```bash
python cureData.py --demographics <path_to_demographics_tsv> --phenotypes <path_to_phenotypes_parquet> --prs <path_to_prs_tsv> --output <path_to_output_parquet_summary> --output_regression_path <path_to_output_parquet_regression> --plot_path <path_to_save_plot>
```

### Example

```bash
python cureData.py --demographics demographics.tsv --phenotypes phenotypes.parquet --prs prs.tsv --output summary.parquet --output_regression_path regression_results.parquet --plot_path regression_plot.png
```

### Output

- **Summary Statistics**: A Parquet file with summary statistics for each phenotype.
- **Regression Analysis**: A Parquet file with regression results, including the coefficients, p-values, and R-squared values.
- **Plot**: A bar plot of PRS coefficients for significant phenotypes.

### Testing

Unit tests are provided using the `unittest` framework. To run the tests:

```bash
python -m unittest test_cureData.py
```
OR

```bash
pytest
```

The tests include validation for:
- Correct data loading
- Generation of summary statistics
- Regression analysis
- Plot generation
- Versioning functionality

## Requirements

The following Python packages are required:

- pandas
- pyarrow
- seaborn
- matplotlib
- numpy
- statsmodels
- pydantic
- scikit-learn

To install the dependencies, run:

```bash
pip install -r requirements.txt
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.