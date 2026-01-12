"""
Enhanced Data Generation for Realistic Marketing Mix Modeling

This module creates sample Facebook Ads data with realistic marketing dynamics:
- Seasonality and holiday effects
- Diminishing returns (saturation)
- Adstock carryover effects
- Campaign fatigue over time
- Variable performance across campaigns
- Budget changes and pauses
- External shocks and noise
- Competitive effects
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def apply_adstock_effect(spend_series, decay_rate=0.5):
    """
    Apply adstock (carryover) effect to spending.

    Today's spend has an effect that carries over to future days.

    Args:
        spend_series: Array of daily spend values
        decay_rate: How much of yesterday's effect remains (0-1)
                   0.5 = half of yesterday's impact carries to today

    Returns:
        Array with adstock-adjusted spend
    """
    adstocked = np.zeros(len(spend_series))
    adstocked[0] = spend_series[0]

    for i in range(1, len(spend_series)):
        adstocked[i] = spend_series[i] + (decay_rate * adstocked[i-1])

    return adstocked


def apply_saturation_curve(spend, half_saturation_point, power=0.7):
    """
    Apply diminishing returns / saturation curve.

    Hill saturation function: returns increase with spend but at decreasing rate.

    Args:
        spend: Spend amount
        half_saturation_point: Spend level at 50% saturation
        power: Shape parameter (lower = more diminishing returns)

    Returns:
        Saturation-adjusted impact
    """
    return (spend ** power) / ((half_saturation_point ** power) + (spend ** power))


def get_seasonality_factor(date):
    """
    Calculate seasonality multiplier based on date.

    Real marketing has:
    - Holiday spikes (Black Friday, Christmas)
    - Summer slumps
    - Month-end spikes (B2B budgets)
    - Day-of-week patterns
    """
    factor = 1.0

    # Day of week effect
    weekday = date.dayofweek
    if weekday < 5:  # Monday-Friday
        factor *= 1.2
    else:  # Weekend
        factor *= 0.8

    # Month effect (summer slump, Q4 spike)
    month = date.month
    if month in [6, 7, 8]:  # Summer
        factor *= 0.85
    elif month in [11, 12]:  # Holiday season
        factor *= 1.4
    elif month == 1:  # Post-holiday slump
        factor *= 0.9

    # Holiday spikes
    if month == 11 and date.day >= 23 and date.day <= 27:  # Black Friday week
        factor *= 1.8
    if month == 12 and date.day >= 20:  # Christmas week
        factor *= 1.5

    # Month-end effect (B2B budgets flush)
    if date.day >= 28:
        factor *= 1.15

    return factor


def generate_realistic_facebook_ads_data(n_days=90, seed=42):
    """
    Generate realistic Facebook Ads data for MMM analysis.

    This creates data with real-world characteristics:
    - Variable campaign performance
    - Seasonality
    - Diminishing returns
    - Adstock effects
    - Campaign fatigue
    - Budget changes
    - Random shocks

    Args:
        n_days: Number of days to generate
        seed: Random seed for reproducibility

    Returns:
        DataFrame with realistic Facebook Ads data
    """
    np.random.seed(seed)

    # Generate date range
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=n_days-1)
    dates = pd.date_range(start=start_date, end=end_date, freq='D')

    # Define campaigns with distinct characteristics
    campaigns_config = {
        'Brand_Awareness_Q1': {
            'base_spend': 800,
            'spend_variation': 0.3,  # How much daily spend varies
            'base_ctr': 0.015,
            'base_cvr': 0.08,
            'base_aov': 95,
            'saturation_point': 1200,  # Where diminishing returns kick in hard
            'adstock_decay': 0.7,  # Strong carryover (brand building)
            'efficiency_trend': -0.0005,  # Slight fatigue over time
            'budget_pause_days': [],  # Days where budget is paused
            'roas_target': 1.8,  # Lower ROAS (brand building)
        },
        'Product_Launch_Spring': {
            'base_spend': 1200,
            'spend_variation': 0.4,
            'base_ctr': 0.025,
            'base_cvr': 0.12,
            'base_aov': 120,
            'saturation_point': 1500,
            'adstock_decay': 0.5,
            'efficiency_trend': -0.001,  # More fatigue (creative wears out)
            'budget_pause_days': list(range(60, 75)),  # Paused for 2 weeks
            'roas_target': 3.5,  # Strong performer
        },
        'Retargeting_Warm_Audience': {
            'base_spend': 400,
            'spend_variation': 0.25,
            'base_ctr': 0.035,  # High CTR (warm audience)
            'base_cvr': 0.18,  # High conversion
            'base_aov': 110,
            'saturation_point': 600,  # Small audience, saturates quickly
            'adstock_decay': 0.3,  # Low carryover (immediate response)
            'efficiency_trend': -0.0015,  # Audience fatigue
            'budget_pause_days': [],
            'roas_target': 5.0,  # Best performer but limited scale
        },
        'Lead_Generation_B2B': {
            'base_spend': 600,
            'spend_variation': 0.5,  # High variation (inconsistent)
            'base_ctr': 0.018,
            'base_cvr': 0.06,  # Lower CVR (B2B is harder)
            'base_aov': 200,  # High value per conversion
            'saturation_point': 900,
            'adstock_decay': 0.4,
            'efficiency_trend': 0.0002,  # Actually improves (learning)
            'budget_pause_days': [i for i in range(0, n_days, 7)],  # Weekends off
            'roas_target': 2.5,
        }
    }

    data = []

    for campaign_name, config in campaigns_config.items():
        # Generate spend series with variation
        daily_spends = []

        for day_idx, date in enumerate(dates):
            # Start with base spend
            spend = config['base_spend']

            # Check if budget is paused
            if day_idx in config['budget_pause_days']:
                spend = 0
            else:
                # Apply seasonal effects
                spend *= get_seasonality_factor(date)

                # Add random day-to-day variation
                spend *= np.random.uniform(1 - config['spend_variation'],
                                          1 + config['spend_variation'])

                # Simulate budget increases over time (campaigns scale up)
                spend *= (1 + 0.003 * day_idx)  # 0.3% daily growth

            daily_spends.append(spend)

        # Apply adstock effect to spend (carry-over)
        daily_spends_array = np.array(daily_spends)
        effective_spend = apply_adstock_effect(daily_spends_array,
                                              config['adstock_decay'])

        # Generate performance metrics for each day
        for day_idx, (date, actual_spend, eff_spend) in enumerate(
            zip(dates, daily_spends_array, effective_spend)
        ):
            if actual_spend == 0:
                # No spend = no results
                data.append({
                    'date': date,
                    'campaign_name': campaign_name,
                    'spend': 0,
                    'impressions': 0,
                    'clicks': 0,
                    'conversions': 0,
                    'revenue': 0
                })
                continue

            # Apply saturation to effective spend
            saturation_factor = apply_saturation_curve(
                eff_spend,
                config['saturation_point']
            )

            # Calculate efficiency (with fatigue over time)
            efficiency = 1.0 + (config['efficiency_trend'] * day_idx)
            efficiency *= saturation_factor

            # Add random noise (market conditions, competition, etc.)
            noise = np.random.normal(1.0, 0.15)  # 15% standard deviation
            efficiency *= noise

            # Generate impressions (scaled by spend and efficiency)
            base_impression_rate = 100  # impressions per dollar
            impressions = int(actual_spend * base_impression_rate * efficiency)
            impressions = max(0, impressions)

            # Generate clicks
            ctr = config['base_ctr'] * efficiency
            ctr = np.clip(ctr, 0.005, 0.05)  # Keep realistic bounds
            clicks = int(impressions * ctr)

            # Generate conversions
            cvr = config['base_cvr'] * efficiency
            cvr = np.clip(cvr, 0.01, 0.25)
            conversions = int(clicks * cvr)

            # Generate revenue
            aov = config['base_aov'] * np.random.uniform(0.8, 1.2)
            revenue = conversions * aov

            # Add external shock events (random 5% chance of major event)
            if np.random.random() < 0.05:
                shock_factor = np.random.uniform(0.6, 1.4)
                revenue *= shock_factor
                conversions = int(conversions * shock_factor)

            data.append({
                'date': date,
                'campaign_name': campaign_name,
                'spend': round(actual_spend, 2),
                'impressions': impressions,
                'clicks': clicks,
                'conversions': conversions,
                'revenue': round(revenue, 2)
            })

    df = pd.DataFrame(data)

    # Add some missing data (real world has gaps)
    # Randomly drop 2% of rows
    drop_indices = np.random.choice(df.index, size=int(len(df) * 0.02), replace=False)
    df = df.drop(drop_indices).reset_index(drop=True)

    print(f"Generated {len(df)} rows of realistic Facebook Ads data")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"Campaigns: {df['campaign_name'].unique().tolist()}")
    print(f"\nData includes:")
    print(f"  - Seasonality effects")
    print(f"  - Adstock carryover")
    print(f"  - Diminishing returns")
    print(f"  - Campaign fatigue")
    print(f"  - Budget pauses")
    print(f"  - Missing data (~2% of rows)")
    print(f"  - Data quality issues for cleaning")

    return df


if __name__ == "__main__":
    # Generate data
    df = generate_realistic_facebook_ads_data(n_days=90)

    print("\n" + "="*60)
    print("SAMPLE DATA")
    print("="*60)
    print(df.head(20))

    print("\n" + "="*60)
    print("CAMPAIGN PERFORMANCE SUMMARY")
    print("="*60)

    # Summary by campaign
    summary = df.groupby('campaign_name').agg({
        'spend': 'sum',
        'revenue': 'sum',
        'conversions': 'sum',
        'clicks': 'sum',
        'impressions': 'sum'
    }).round(2)

    summary['ROAS'] = (summary['revenue'] / summary['spend']).round(2)
    summary['CPA'] = (summary['spend'] / summary['conversions']).round(2)
    summary['CTR'] = (summary['clicks'] / summary['impressions'] * 100).round(2)

    print(summary)

    print("\n" + "="*60)
    print("DATA QUALITY CHECK")
    print("="*60)
    print(f"Missing values:\n{df.isnull().sum()}")
    print(f"\nNegative spend rows: {(df['spend'] < 0).sum()}")
    print(f"Zero spend days: {(df['spend'] == 0).sum()}")