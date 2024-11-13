import unittest
import pandas as pd
import os
from io import StringIO
import sys
from cureData import *
from pydantic import ValidationError

VERSION = "1.0.0"

def get_version():
    return VERSION

def increment_version():
    global VERSION
    major, minor, patch = map(int, VERSION.split('.'))
    patch += 1
    new_version = f"{major}.{minor}.{patch}"
    VERSION = new_version
    return VERSION

class TestCureDataModule(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.demographics_data_path = 'demographics.tsv'
        cls.phenotypes_data_path = 'phenotypes.parquet'
        cls.prs_data_path = 'prs.tsv'

    def test_load_data(self):
        demographics = pd.read_csv(self.demographics_data_path, sep='\t')
        phenotypes = pd.read_parquet(self.phenotypes_data_path)
        prs = pd.read_csv(self.prs_data_path, sep='\t')

        self.assertEqual(demographics.shape[1], 4)
        self.assertTrue(demographics.shape[0] > 0)
        self.assertEqual(phenotypes.shape[1], 3)
        self.assertEqual(prs.shape[1], 2)
        self.assertTrue(demographics['uuid'].dtype == 'O')

    def test_generate_summary_stats(self):
        demographics = pd.read_csv(self.demographics_data_path, sep='\t')
        phenotypes = pd.read_parquet(self.phenotypes_data_path)

        summary_df = generate_summary_stats(demographics, phenotypes)

        expected_columns = ['phenotype', 'num_patients', 'missing_rate', 'mean', 'median', 'std_dev', 'avg_age']
        self.assertEqual(summary_df.shape[1], 7)
        self.assertTrue(all(col in summary_df.columns for col in expected_columns))
        self.assertTrue(summary_df.shape[0] > 0)

    def test_regression_analysis(self):
        demographics = pd.read_csv(self.demographics_data_path, sep='\t')
        phenotypes = pd.read_parquet(self.phenotypes_data_path)
        prs = pd.read_csv(self.prs_data_path, sep='\t')

        regression_df = regression_analysis(demographics, phenotypes, prs)

        self.assertEqual(regression_df.shape[1], 4)
        self.assertTrue(regression_df.shape[0] > 0)
        self.assertTrue(all(col in regression_df.columns for col in ['phenotype_id', 'coef_prs_score', 'p_value_prs_score', 'r_squared']))

    def test_plot_regression_results(self):
        regression_df = pd.DataFrame({
            'phenotype_id': ['pheno_1', 'pheno_2'],
            'coef_prs_score': [0.1, -0.2],
            'p_value_prs_score': [0.04, 0.06],
            'r_squared': [0.5, 0.6]
        })

        plot_path = "test_plot.png"
        plot_regression_results(regression_df, plot_path)
        self.assertTrue(os.path.exists(plot_path))

    def test_version_functions(self):
        initial_version = get_version()
        new_version = increment_version()

        self.assertNotEqual(initial_version, new_version)
        self.assertTrue(self.version_compare(new_version, initial_version))

    @staticmethod
    def version_compare(version1, version2):
        # Helper function to compare version strings
        v1_parts = list(map(int, version1.split('.')))
        v2_parts = list(map(int, version2.split('.')))
        return v1_parts > v2_parts

    def test_validate_summary_stats_valid_data(self):
        valid_data = pd.DataFrame({
            "phenotype": [1, 2],
            "num_patients": [100, 200],
            "missing_rate": [0.1, 0.2],
            "mean": [5.2, 6.5],
            "median": [5, 6],
            "std_dev": [1.2, 1.5],
            "avg_age": [65, 70]
        })

        captured_output = StringIO()
        sys.stdout = captured_output
        validate_summary_stats(valid_data)
        self.assertEqual(captured_output.getvalue(), "")
        sys.stdout = sys.__stdout__

    def test_validate_summary_stats_invalid_data(self):
        invalid_data = pd.DataFrame({
            "phenotype": [1, 2],
            "num_patients": [100, 200],
            "missing_rate": [0.1, 0.2],
            "mean": [5.2, 6.5],
            "std_dev": [1.2, 1.5],
            "avg_age": [65, 70]
            # Missing 'median' field here
        })

        captured_output = StringIO()
        sys.stdout = captured_output
        validate_summary_stats(invalid_data)
        self.assertIn("Validation error", captured_output.getvalue())
        sys.stdout = sys.__stdout__

    def test_validate_summary_stats_empty_data(self):
        empty_data = pd.DataFrame()

        captured_output = StringIO()
        sys.stdout = captured_output
        validate_summary_stats(empty_data)
        self.assertEqual(captured_output.getvalue(), "")
        sys.stdout = sys.__stdout__

if __name__ == "__main__":
    unittest.main()
