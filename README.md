# gu-cancer-analysis
#!/usr/bin/env python3
"""
Genitourinary Cancer Survival Analysis Tool
Author: Your Name | Institution: Najran University
Usage: python analyze_gu_cancer.py --input data.xlsx --output-dir results/
"""

import argparse
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter, CoxPHFitter, WeibullFitter
from lifelines.statistics import logrank_test
import seaborn as sns
from sklearn.impute import IterativeImputer

def parse_arguments():
    parser = argparse.ArgumentParser(description='GU Cancer Survival Analysis')
    parser.add_argument('--input', required=True, help='Path to Excel file')
    parser.add_argument('--output-dir', default='results', help='Output directory')
    parser.add_argument('--dpi', type=int, default=300, help='Figure resolution')
    return parser.parse_args()

def load_data(filepath):
    """Load and preprocess the Excel data"""
    df = pd.read_excel(filepath, sheet_name="Form responses 1")
    
    # Clean survival time
    df['survival_months'] = (
        df['Survival time ( from initial diagnosis to death)']
        .replace({'Not applicable': np.nan, '> 5 years': 60,
                 '6 months or less': 6, '> 6 months < 12': 9,
                 '12-18 months': 15, '19-24 months': 21,
                 '25-36 months': 30})
        .astype(float)
    )
    
    # Convert status
    status_map = {'dead': 1, 'متوفى': 1, 'alive': 0, 'على قيد الحياة': 0}
    df['status'] = df['status'].map(status_map).fillna(0)
    
    return df

def run_analysis(df, output_dir, dpi=300):
    """Main analysis pipeline"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Kaplan-Meier Plots by Cancer Type
    plt.figure(figsize=(10,6))
    ax = plt.subplot(111)
    cancer_types = df['Pathology ( if any of the following continue your survey )'].unique()
    
    for cancer in cancer_types:
        subset = df[df['Pathology ( if any of the following continue your survey )'] == cancer]
        kmf = KaplanMeierFitter()
        kmf.fit(subset['survival_months'], subset['status'], label=f"{cancer} (n={len(subset)})")
        kmf.plot(ax=ax, ci_show=True)
    
    plt.title('Overall Survival by Cancer Type')
    plt.xlabel('Time (Months)')
    plt.ylabel('Survival Probability')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/survival_curves.png', dpi=dpi)
    plt.close()
    
    # 2. Generate Summary Statistics Table
    results = []
    for cancer in cancer_types:
        subset = df[df['Pathology ( if any of the following continue your survey )'] == cancer]
        kmf = KaplanMeierFitter()
        kmf.fit(subset['survival_months'], subset['status'])
        
        results.append({
            'Cancer Type': cancer,
            'N': len(subset),
            'Deaths': subset['status'].sum(),
            'Median Survival (95% CI)': f"{kmf.median_survival_time_:.1f} ({kmf.confidence_interval_median_survival_time_.iloc[0,0]:.1f}-{kmf.confidence_interval_median_survival_time_.iloc[0,1]:.1f})",
            '1-Year Survival': f"{kmf.predict(12):.1%}",
            '5-Year Survival': f"{kmf.predict(60):.1%}"
        })
    
    pd.DataFrame(results).to_csv(f'{output_dir}/survival_summary.csv', index=False)
    
    # 3. Additional analyses can be added here...
    
    print(f"Analysis complete! Results saved to {output_dir}")

if __name__ == "__main__":
    args = parse_arguments()
    df = load_data(args.input)
    run_analysis(df, args.output_dir, args.dpi)
    [Uploading Pattern of Genitourinary cancer in Najran  (Responses) (1).xlsx…]()
