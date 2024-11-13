import argparse
import pandas as pd
import pyarrow as arrow
import pyarrow.parquet as pq
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
from pydantic import BaseModel, Field, ValidationError
import os
from sklearn.preprocessing import OneHotEncoder

# Ignore timezone warnings from pyarrow
os.environ['PYARROW_IGNORE_TIMEZONE'] = '1'

def read_version(file_path="version.txt"):
    """Read the version from a version file."""
    try:
        with open(file_path, 'r') as f:
            version = f.read().strip()
    except FileNotFoundError:
        version = "1.0.0"  # Default version if file is not found
    return version

def write_version(version, file_path="version.txt"):
    """Write the new version to the version file."""
    with open(file_path, 'w') as f:
        f.write(version)

def increment_version(version):
    """Increment the patch version."""
    major, minor, patch = map(int, version.split('.'))
    patch += 1
    new_version = f"{major}.{minor}.{patch}"
    return new_version

class SummaryStatsModel(BaseModel):
    phenotype: int
    num_patients: int
    missing_rate: float
    mean: float
    median: float
    std_dev: float
    avg_age: float

    class Config:
        extra = "ignore"

def load_data(demographics_path, phenotypes_path, prs_path):
    demographics = pd.read_csv(demographics_path, sep='\t')
    demographics['uuid'] = demographics['uuid'].astype(str)
    phenotypes = pd.read_parquet(phenotypes_path)
    phenotypes['uuid'] = phenotypes['uuid'].astype(str)
    prs = pd.read_csv(prs_path, sep='\t')
    prs['uuid'] = prs['uuid'].astype(str)
    return demographics, phenotypes, prs

def validate_summary_stats(summary_df):
    for idx, row in summary_df.iterrows():
        try:
            SummaryStatsModel.parse_obj(row.to_dict())
        except ValidationError as e:
            print(f"Validation error at row {idx}: {e}")
            continue

def generate_summary_stats(demographics, phenotypes):
    demographics['uuid'] = demographics['uuid'].astype(str)
    phenotypes['uuid'] = phenotypes['uuid'].astype(str)
    merged_data = pd.merge(demographics, phenotypes, on="uuid", how="inner")
    
    summary_stats = []
    for phenotype_id, group in merged_data.groupby("phenotype_id"):
        total_patients = demographics.shape[0]
        missing_rate = 1 - (group['value'].notnull().sum() / total_patients)
        data = group[['age_at_progression_enrollment', 'value']].dropna(subset=['value'])
        stats = {
            "phenotype": phenotype_id,
            "num_patients": data['value'].notnull().sum(),
            "missing_rate": missing_rate,
            "mean": data['value'].mean(),
            "median": data['value'].median(),
            "std_dev": data['value'].std(),
            "avg_age": data['age_at_progression_enrollment'].mean()
        }
        summary_stats.append(stats)
    summary_df = pd.DataFrame(summary_stats)
    
    validate_summary_stats(summary_df)
    
    return summary_df

def regression_analysis(demographics, phenotypes, prs):
    demographics['uuid'] = demographics['uuid'].astype(str)
    phenotypes['uuid'] = phenotypes['uuid'].astype(str)
    prs['uuid'] = prs['uuid'].astype(str)
    merged_data = demographics.merge(phenotypes, on="uuid").merge(prs, on="uuid")
    
    regression_results = []
    one_hot_encoder = OneHotEncoder(handle_unknown='ignore')
    smoking_dummies = one_hot_encoder.fit_transform(merged_data[['smoking_status']]).toarray()
    smoking_columns = one_hot_encoder.get_feature_names_out(['smoking_status'])
    smoking_df = pd.DataFrame(smoking_dummies, columns=smoking_columns, index=merged_data.index)
    merged_data = pd.concat([merged_data, smoking_df], axis=1).drop(columns=['smoking_status'])
    
    for phenotype_id, group in merged_data.groupby("phenotype_id"):
        phenotype_data = group.dropna(subset=['value', 'prs'])
        
        X = phenotype_data[['age_at_progression_enrollment', 'sexM', 'prs'] + list(smoking_columns)]
        y = phenotype_data['value']
        
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.dropna()
        y = y.loc[X.index]
        
        if y.isnull().any() or X.isnull().any().any():
            print(f"Skipping phenotype_id: {phenotype_id} due to missing data.")
            continue
        
        X = sm.add_constant(X)
        model = sm.OLS(y, X).fit()
        
        result = {
            "phenotype_id": phenotype_id,
            "coef_prs_score": model.params['prs'],
            "p_value_prs_score": model.pvalues['prs'],
            "r_squared": model.rsquared
        }
        regression_results.append(result)
    
    return pd.DataFrame(regression_results)

def plot_regression_results(regression_df, plot_path="regression_plot.png"):
    significant_results = regression_df[regression_df['p_value_prs_score'] < 0.05]
    significant_results = significant_results.sort_values(by='coef_prs_score', ascending=False)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(data=significant_results, x='coef_prs_score', y='phenotype_id', palette='viridis')
    plt.title('PRS Coefficients for Significant Phenotypes')
    plt.xlabel('Coefficient of PRS Score')
    plt.ylabel('Phenotype ID')
    plt.axvline(0, color='gray', linestyle='--')
    plt.tight_layout()
    
    plt.savefig(plot_path)
    print(f"Plot saved to {plot_path}")

def main(demographics_path, phenotypes_path, prs_path, output_path, output_regression_path, plot_path="regression_plot.png"):
    version = read_version()  # Read current version from file
    print(f"Running version {version}")
    
    demographics, phenotypes, prs = load_data(demographics_path, phenotypes_path, prs_path)
    
    summary_df = generate_summary_stats(demographics, phenotypes)
    
    table = arrow.Table.from_pandas(summary_df)
    pq.write_table(table, output_path)
    print(f"Summary statistics saved to {output_path}")
    
    regression_df = regression_analysis(demographics, phenotypes, prs)
    
    regression_df.to_parquet(output_regression_path)
    
    plot_regression_results(regression_df, plot_path)
    
    # Increment version and write back to file
    new_version = increment_version(version)
    write_version(new_version)
    print(f"Version incremented to {new_version}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate summary statistics and regression analysis for phenotypes.")
    parser.add_argument("--demographics", type=str, required=True, help="Path to demographics.tsv file")
    parser.add_argument("--phenotypes", type=str, required=True, help="Path to phenotypes.parquet file")
    parser.add_argument("--prs", type=str, required=True, help="Path to prs.tsv file")
    parser.add_argument("--output", type=str, required=True, help="Path to output Parquet file for summary statistics")
    parser.add_argument("--output_regression_path", type=str, required=True, help="Path to output Parquet file for regression analysis")
    parser.add_argument("--plot_path", type=str, default="regression_plot.png", help="Path to save the regression plot")
    
    args = parser.parse_args()
    main(args.demographics, args.phenotypes, args.prs, args.output, args.output_regression_path, args.plot_path)
