"""
Data cleaning module.

This module handles missing values, outliers, duplicates, and other data quality issues.
"""

import pandas as pd
import numpy as np


def clean_data(df):
    """
    Clean Facebook Ads data.

    Tasks:
    - Handle missing values
    - Remove/fix duplicates
    - Handle outliers
    - Fix data type issues
    - Basic data corrections

    Args:
        df: Validated Facebook Ads DataFrame

    Returns:
        Cleaned DataFrame
    """

    print("=" * 50)
    print("STEP 3: CLEANING DATA")
    print("=" * 50)

    df_clean = df.copy()
    original_rows = len(df_clean)

    # 1. Ensure date is datetime
    if not pd.api.types.is_datetime64_any_dtype(df_clean['date']):
        print("Converting date column to datetime...")
        df_clean['date'] = pd.to_datetime(df_clean['date'])
        print("  Date column converted")

    # 2. Handle missing values in numeric columns
    numeric_cols = ['spend', 'impressions', 'clicks', 'conversions', 'revenue']

    for col in numeric_cols:
        missing_count = df_clean[col].isnull().sum()
        if missing_count > 0:
            print(f"\nHandling {missing_count} missing values in '{col}':")

            # Strategy: Fill with 0 (assuming missing means no activity)
            df_clean[col] = df_clean[col].fillna(0)
            print(f"  - Filled with 0")

    # 3. Handle missing campaign names
    if df_clean['campaign_name'].isnull().any():
        missing_count = df_clean['campaign_name'].isnull().sum()
        print(f"\nHandling {missing_count} missing campaign names:")
        df_clean['campaign_name'] = df_clean['campaign_name'].fillna('Unknown_Campaign')
        print(f"   - Filled with 'Unknown_Campaign'")

    # 4. Remove duplicates (same date + campaign)
    duplicates = df_clean.duplicated(subset=['date', 'campaign_name'], keep='first')
    dup_count = duplicates.sum()
    if dup_count > 0:
        print(f"\nRemoving {dup_count} duplicate rows...")
        df_clean = df_clean[~duplicates]
        print("  Duplicates removed")
    else:
        print("\n  No duplicates found")

    # 5. Handle negative values (replace with 0)
    for col in numeric_cols:
        neg_count = (df_clean[col] < 0).sum()
        if neg_count > 0:
            print(f"\nFixing {neg_count} negative values in '{col}':")
            df_clean.loc[df_clean[col] < 0, col] = 0
            print(f"   - Replaced with 0")

    # 6. Handle outliers (using IQR method)
    print("\nChecking for outliers...")
    for col in ['spend', 'revenue']:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - 3 * IQR  # Using 3*IQR for more conservative outlier detection
        upper_bound = Q3 + 3 * IQR

        outliers = ((df_clean[col] < lower_bound) | (df_clean[col] > upper_bound)).sum()

        if outliers > 0:
            print(f"  - {col}: {outliers} outliers detected (range: {lower_bound:.2f} to {upper_bound:.2f})")
            # Strategy: Cap outliers rather than remove them
            df_clean[col] = df_clean[col].clip(lower=lower_bound, upper=upper_bound)
            print(f"     - Capped to bounds")
        else:
            print(f"    {col}: No outliers")

    # 7. Fix logical inconsistencies
    print("\nFixing logical inconsistencies...")

    # Clicks cannot exceed impressions
    invalid_clicks = df_clean['clicks'] > df_clean['impressions']
    if invalid_clicks.any():
        count = invalid_clicks.sum()
        print(f"  - Fixing {count} rows where clicks > impressions")
        df_clean.loc[invalid_clicks, 'clicks'] = df_clean.loc[invalid_clicks, 'impressions']

    # Conversions cannot exceed clicks
    invalid_conversions = df_clean['conversions'] > df_clean['clicks']
    if invalid_conversions.any():
        count = invalid_conversions.sum()
        print(f"  - Fixing {count} rows where conversions > clicks")
        df_clean.loc[invalid_conversions, 'conversions'] = df_clean.loc[invalid_conversions, 'clicks']

    print("    Logical consistency enforced")

    # 8. Ensure correct data types
    print("\nEnsuring correct data types...")
    df_clean['impressions'] = df_clean['impressions'].astype(int)
    df_clean['clicks'] = df_clean['clicks'].astype(int)
    df_clean['conversions'] = df_clean['conversions'].astype(int)
    df_clean['spend'] = df_clean['spend'].astype(float)
    df_clean['revenue'] = df_clean['revenue'].astype(float)
    print("  Data types corrected")

    # 9. Sort by date
    df_clean = df_clean.sort_values('date').reset_index(drop=True)
    print("  Data sorted by date")

    # Summary
    rows_removed = original_rows - len(df_clean)
    print("\n" + "=" * 50)
    print(f"  CLEANING COMPLETE")
    print(f"Original rows: {original_rows}")
    print(f"Cleaned rows: {len(df_clean)}")
    print(f"Rows removed: {rows_removed}")
    print("=" * 50 + "\n")

    return df_clean


if __name__ == "__main__":
    # Test with sample data
    from fetch_data import generate_realistic_facebook_ads_data
    from validate_data import validate_data

    # Generate and validate data
    df = generate_realistic_facebook_ads_data(n_days=30)
    validation_results = validate_data(df)

    # Clean data
    df_clean = clean_data(df)

    print("\nCleaned data sample:")
    print(df_clean.head())
    print("\nCleaned data info:")
    print(df_clean.info())