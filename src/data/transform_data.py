"""
Data transformation module.

This module creates features needed for marketing mix modeling and analysis.
"""

import pandas as pd
import numpy as np
from datetime import datetime


def transform_data(df):
    """
    Transform cleaned data by creating features.

    Creates:
    - Time-based features (day of week, weekend, holidays, etc.)
    - Derived metrics (CTR, CPC, CVR, CPA, ROAS)
    - Aggregated spend across all campaigns
    - Lagged features
    - Rolling averages

    Args:
        df: Cleaned Facebook Ads DataFrame

    Returns:
        Transformed DataFrame with new features
    """

    print("=" * 50)
    print("STEP 4: TRANSFORMING DATA (Feature Engineering)")
    print("=" * 50)

    df_transform = df.copy()

    # ===========================
    # 1. TIME-BASED FEATURES
    # ===========================
    print("\n1. Creating time-based features...")

    df_transform['day_of_week'] = df_transform['date'].dt.dayofweek  # 0=Monday, 6=Sunday
    df_transform['day_name'] = df_transform['date'].dt.day_name()
    df_transform['is_weekend'] = df_transform['day_of_week'].isin([5, 6]).astype(int)
    df_transform['month'] = df_transform['date'].dt.month
    df_transform['quarter'] = df_transform['date'].dt.quarter
    df_transform['year'] = df_transform['date'].dt.year

    print(f"    Added: day_of_week, day_name, is_weekend, month, quarter, year")

    # Simple holiday indicator (US major holidays)
    # In production, you'd use a holiday calendar library
    df_transform['is_holiday'] = 0

    # Mark common holiday periods (simplified)
    for idx, row in df_transform.iterrows():
        date = row['date']
        # Christmas season (Dec 20-31)
        if date.month == 12 and date.day >= 20:
            df_transform.loc[idx, 'is_holiday'] = 1
        # Thanksgiving week (4th Thursday of Nov)
        elif date.month == 11 and date.day >= 22:
            df_transform.loc[idx, 'is_holiday'] = 1
        # Black Friday / Cyber Monday
        elif date.month == 11 and date.day in [23, 24, 25, 26, 27]:
            df_transform.loc[idx, 'is_holiday'] = 1

    holiday_count = df_transform['is_holiday'].sum()
    print(f"    Added: is_holiday ({holiday_count} holiday days marked)")

    # ===========================
    # 2. DERIVED METRICS
    # ===========================
    print("\n2. Creating derived metrics...")

    # CTR (Click-Through Rate)
    df_transform['ctr'] = np.where(
        df_transform['impressions'] > 0,
        df_transform['clicks'] / df_transform['impressions'],
        0
    )

    # CPC (Cost Per Click)
    df_transform['cpc'] = np.where(
        df_transform['clicks'] > 0,
        df_transform['spend'] / df_transform['clicks'],
        0
    )

    # CVR (Conversion Rate)
    df_transform['cvr'] = np.where(
        df_transform['clicks'] > 0,
        df_transform['conversions'] / df_transform['clicks'],
        0
    )

    # CPA (Cost Per Acquisition/Conversion)
    df_transform['cpa'] = np.where(
        df_transform['conversions'] > 0,
        df_transform['spend'] / df_transform['conversions'],
        0
    )

    # ROAS (Return on Ad Spend)
    df_transform['roas'] = np.where(
        df_transform['spend'] > 0,
        df_transform['revenue'] / df_transform['spend'],
        0
    )

    print(f"    Added: ctr, cpc, cvr, cpa, roas")

    # ===========================
    # 3. DAILY AGGREGATES (Total spend across all campaigns per day)
    # ===========================
    print("\n3. Creating daily aggregates...")

    # Group by date to get total daily metrics
    daily_totals = df_transform.groupby('date').agg({
        'spend': 'sum',
        'impressions': 'sum',
        'clicks': 'sum',
        'conversions': 'sum',
        'revenue': 'sum'
    }).add_suffix('_daily_total').reset_index()

    # Merge back to main dataframe
    df_transform = df_transform.merge(daily_totals, on='date', how='left')

    print(f"    Added: *_daily_total columns (5 metrics)")

    # ===========================
    # 4. LAGGED FEATURES (Previous day's spend)
    # ===========================
    print("\n4. Creating lagged features...")

    # For each campaign, create lagged spend
    # First sort by campaign and date
    df_transform = df_transform.sort_values(['campaign_name', 'date'])

    # Create lag features within each campaign
    for lag in [1, 7]:  # 1 day ago, 7 days ago
        df_transform[f'spend_lag_{lag}'] = df_transform.groupby('campaign_name')['spend'].shift(lag)
        df_transform[f'spend_lag_{lag}'] = df_transform[f'spend_lag_{lag}'].fillna(0)

    print(f"    Added: spend_lag_1, spend_lag_7")

    # ===========================
    # 5. ROLLING AVERAGES (7-day moving average)
    # ===========================
    print("\n5. Creating rolling averages...")

    # 7-day rolling average of spend (within each campaign)
    df_transform['spend_rolling_7d'] = df_transform.groupby('campaign_name')['spend'].transform(
        lambda x: x.rolling(window=7, min_periods=1).mean()
    )

    # 7-day rolling sum of conversions
    df_transform['conversions_rolling_7d'] = df_transform.groupby('campaign_name')['conversions'].transform(
        lambda x: x.rolling(window=7, min_periods=1).sum()
    )

    print(f"    Added: spend_rolling_7d, conversions_rolling_7d")

    # ===========================
    # 6. CAMPAIGN-LEVEL FEATURES
    # ===========================
    print("\n6. Creating campaign-level features...")

    # Days since campaign start (for each campaign)
    campaign_start_dates = df_transform.groupby('campaign_name')['date'].min()
    df_transform['campaign_start_date'] = df_transform['campaign_name'].map(campaign_start_dates)
    df_transform['days_since_campaign_start'] = (
            df_transform['date'] - df_transform['campaign_start_date']
    ).dt.days

    # Drop the helper column
    df_transform = df_transform.drop('campaign_start_date', axis=1)

    print(f"    Added: days_since_campaign_start")

    # ===========================
    # 7. CATEGORICAL ENCODING (If needed for modeling later)
    # ===========================
    print("\n7. Preparing categorical variables...")

    # Create campaign type categories (extract from campaign name)
    df_transform['campaign_type'] = df_transform['campaign_name'].apply(
        lambda x: x.split('_')[0] if '_' in x else 'Other'
    )

    print(f"    Added: campaign_type")
    print(f"  Campaign types: {df_transform['campaign_type'].unique().tolist()}")

    # Reset index and sort by date
    df_transform = df_transform.sort_values('date').reset_index(drop=True)

    # Summary
    original_cols = len(df.columns)
    new_cols = len(df_transform.columns)
    added_cols = new_cols - original_cols

    print("\n" + "=" * 50)
    print(f"  TRANSFORMATION COMPLETE")
    print(f"Original columns: {original_cols}")
    print(f"New columns: {new_cols}")
    print(f"Features added: {added_cols}")
    print("=" * 50 + "\n")

    return df_transform


if __name__ == "__main__":
    # Test with sample data
    from fetch_data import generate_realistic_facebook_ads_data
    from validate_data import validate_data
    from clean_data import clean_data

    # Run full pipeline
    print("Running full data pipeline...\n")

    # Step 1: Fetch (generate sample data)
    df_raw = generate_realistic_facebook_ads_data(n_days=90)

    # Step 2: Validate
    validation_results = validate_data(df_raw)

    # Step 3: Clean
    df_clean = clean_data(df_raw)

    # Step 4: Transform
    df_final = transform_data(df_clean)

    print("\nFinal transformed data sample:")
    print(df_final.head())

    print("\nAll columns in final dataset:")
    print(df_final.columns.tolist())

    print("\nFinal data shape:", df_final.shape)
    print("\nBasic statistics:")
    print(df_final[['spend', 'revenue', 'roas', 'ctr', 'cvr']].describe())