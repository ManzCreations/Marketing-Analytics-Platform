"""
Data fetching module for Facebook Ads sample data.

This module creates sample Facebook Ads data that mimics the
structure you'd get from the Facebook Marketing API.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def generate_facebook_ads_sample(n_days:int=90):
    """
    Generate sample Facebook Ads data.

    This simulates what you'd get from the Facebook Marketing API.
    In production, this would be replaced with actual API calls.

    Args:
        n_days: Number of days of data to generate

    Returns:
        DataFrame with Facebook Ads data
    """

    # Set random seed for reproducibility
    np.random.seed(42)

    # Generate date range
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=n_days-1)
    dates = pd.date_range(start=start_date, end=end_date, freq='D')

    # Define campaigns
    campaigns = [
        'Brand_Awareness_Q1',
        'Product_Launch_Spring',
        'Retargeting_Warm_Audience',
        'Lead_Generation_B2B'
    ]

    data = []
    for date in dates:
        for campaign in campaigns:
            # Generate realistic spend amounts
            if 'Brand' in campaign:
                base_spend = 500
            elif 'Product' in campaign:
                base_spend = 1000
            elif 'Retargeting' in campaign:
                base_spend = 300
            else:
                base_spend = 700

            # Add some randomness and weekday effects
            weekday_multiplier = 1.3 if date.dayofweek < 5 else 0.7  # Higher on weekdays
            spend = base_spend * weekday_multiplier * np.random.uniform(0.8, 1.2)

            # Generate impressions (roughly proportional to spend)
            impressions = int(spend * np.random.uniform(80, 120))

            # Generate clicks (realistic CTR between 1-3%)
            ctr = np.random.uniform(0.01, 0.03)
            clicks = int(impressions * ctr)

            # Generate conversions (realistic CVR between 5-15% of clicks)
            cvr = np.random.uniform(0.05, 0.15)
            conversions = int(clicks * cvr)

            # Generate revenue (average order value between $50-$150)
            aov = np.random.uniform(50, 150)
            revenue = conversions * aov

            data.append({
                'date': date,
                'campaign_name': campaign,
                'spend': round(spend, 2),
                'impressions': impressions,
                'clicks': clicks,
                'conversions': conversions,
                'revenue': round(revenue, 2)
            })

    df = pd.DataFrame(data)

    print(f"Generated {len(df)} rows of Facebook Ads data")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"Campaigns: {df['campaign_name'].unique().tolist()}")

    return df

if __name__ == "__main__":
    df = generate_facebook_ads_sample(n_days=90)
    print("\nFirst few rows:")
    print(df.head(10))
    print("\nBasic stats:")
    print(df.describe())

