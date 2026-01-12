"""
Data validation module.

This module checks the quality of raw Facebook Ads data before cleaning.
"""

import pandas as pd
import numpy as np


def validate_data(df):
    """
    Validate Facebook Ads data for quality issues.

    Checks:
    - Required columns are present
    - No completely empty DataFrame
    - Data types are correct
    - Basic logical consistency

    Args:
        df: Raw Facebook Ads DataFrame

    Returns:
        dict with validation results and issues found
    """

    issues = []

    print("=" * 50)
    print("STEP 2: VALIDATING DATA")
    print("=" * 50)

    # Check 1: Dataframe is not empty
    if df.empty:
        issues.append("ERROR: DataFrame is empty")
        return {'valid': False, 'issues': issues}

    print(f"  DataFrame has {len(df)} rows")

    # Check 2: Required_columns exist
    required_cols = ['date', 'campaign_name', 'spend', 'impressions', 'clicks', 'conversions', 'revenue']
    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        issues.append(f"ERROR: Missing required columns: {missing_cols}")
    else:
        print(f"  All required columns present: {required_cols}")

    # Check 3: Check for completely null columns
    null_cols = df.columns[df.isnull().all()].tolist()
    if null_cols:
        issues.append(f"WARNING: Completely null columns: {null_cols}")

    # Check 4: Check datatypes
    numeric_cols = ['spend', 'impressions', 'clicks', 'conversions', 'revenue']
    for col in numeric_cols:
        if col in df.columns:
            if not pd.api.types.is_numeric_dtype(df[col]):
                issues.append(f"WARNING: Column '{col}' is not a numeric (type: {df[col].dtype})")
            else:
                print(f"  Column '{col}' is numeric.")

    # Check 5: Date column is datetime:
    if 'date' in df.columns:
        if not pd.api.types.is_datetime64_any_dtype(df['date']):
            issues.append(f"WARNING: 'date' column is not datetime (type: {df['date'].dtype})")
        else:
            print(f"  Date column is of type datetime.")

    # Check 6: Check for negative values (should not exist)
    for col in ['spend', 'impressions', 'clicks', 'conversions', 'revenue']:
        if col in df.columns:
            neg_count = (df[col] < 0).sum()
            if neg_count > 0:
                issues.append(f"ERROR: {neg_count} negative values in '{col}'")

    if not any('negative' in issue.lower() for issue in issues):
        print("  No negative values found.")

    # Check 7: Logical consistency checks
    # Clicks should not exceed impressions
    if 'clicks' in df.columns and 'impressions' in df.columns:
        invalid = (df['clicks'] > df['impressions']).sum()
        if invalid > 0:
            issues.append(f"WARNING: {invalid} rows where clicks > impressions")
        else:
            print("  Clicks <= Impressions (logical)")

    # Conversions should not exceed clicks
    if 'conversions' in df.columns and 'clicks' in df.columns:
        invalid = (df['conversions'] > df['clicks']).sum()
        if invalid > 0:
            issues.append(f"WARNING: {invalid} rows where conversions > clicks")
        else:
            print("  Conversions <= Clicks (logical)")

    # Check 8: Check for missing values summary
    missing_summary = df.isnull().sum()
    if missing_summary.sum() > 0:
        print("\nMissing values found:")
        for col, count in missing_summary[missing_summary > 0].items():
            pct = (count / len(df)) * 100
            print(f"  - {col}: {count} ({pct:.1f}%)")
            issues.append(f"INFO: {col} has {count} missing values ({pct:.1f}%)")
    else:
        print("  No missing values")

    # Determine if validation passed
    critical_issues = [i for i in issues if i.startswith('ERROR')]
    valid = len(critical_issues) == 0

    print("\n" + "=" * 50)
    if valid:
        print("  VALIDATION PASSED")
    else:
        print("  VALIDATION FAILED")
    print(f"Total issues found: {len(issues)}")
    print("=" * 50 + "\n")

    return {
        'valid': valid,
        'issues': issues,
        'row_count': len(df),
        'column_count': len(df.columns)
    }


if __name__ == "__main__":
    # Test with sample data
    from fetch_data import generate_realistic_facebook_ads_data

    df = generate_realistic_facebook_ads_data(n_days=30)
    results = validate_data(df)

    print("\nValidation results:")
    for key, value in results.items():
        print(f"{key}: {value}")