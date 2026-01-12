"""
Feature Engineering for Marketing Mix Modeling (MMM)

This module transforms raw marketing data into features suitable for MMM:
1. Adstock transformation - Captures advertising carryover effects
2. Saturation transformation - Models diminishing returns
3. Time-based features - Seasonality, trends, holidays
4. Interaction features - Campaign combinations (optional)
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')


class MMMFeatureEngineer:
    """
    Feature engineering pipeline for Marketing Mix Modeling.

    This class handles all transformations needed to prepare marketing data
    for regression-based attribution modeling.

    Key Capabilities:
    - Adstock transformation with configurable decay rates
    - Saturation (log) transformation for diminishing returns
    - Seasonality and trend features
    - Train/test splitting with time-based logic

    Example Usage:
        engineer = MMMFeatureEngineer()
        df_transformed = engineer.fit_transform(df_clean)
        X_train, X_test, y_train, y_test = engineer.train_test_split(df_transformed)
    """

    def __init__(
        self,
        adstock_params: Optional[Dict[str, float]] = None,
        apply_saturation: bool = True,
        include_time_features: bool = True
    ):
        """
        Initialize the feature engineer.

        Args:
            adstock_params: Dict mapping campaign names to decay rates (0-1)
                           If None, uses default 0.5 for all campaigns
            apply_saturation: Whether to apply log saturation transformation
            include_time_features: Whether to include day of week, holidays, etc.
        """
        self.adstock_params = adstock_params or {}
        self.use_saturation = apply_saturation
        self.include_time_features = include_time_features

        # Will store feature names after transformation
        self.feature_columns_ = None
        self.campaign_names_ = None

    def apply_adstock(
        self,
        spend_series: pd.Series,
        decay_rate: float = 0.5
    ) -> pd.Series:
        """
        Apply adstock (advertising carryover) transformation.

        Mathematical formula:
            Adstock_t = Spend_t + (decay_rate × Adstock_t-1)

        Interpretation:
            - decay_rate = 0.0: No carryover (immediate effect only)
            - decay_rate = 0.5: Half of yesterday's effect remains today
            - decay_rate = 0.9: Strong carryover (brand building)

        Args:
            spend_series: Daily spend values (pd.Series)
            decay_rate: Decay rate between 0 and 1
                       Higher = stronger carryover effect

        Returns:
            pd.Series with adstocked values

        Example:
            If you spend $100 on Monday with decay_rate=0.5:
            - Monday effect: $100
            - Tuesday effect from Monday: $50 (even if Tuesday spend = $0)
            - Wednesday effect from Monday: $25
        """
        if not 0 <= decay_rate <= 1:
            raise ValueError(f"decay_rate must be between 0 and 1, got {decay_rate}")

        adstocked = np.zeros(len(spend_series))
        spend_array = spend_series.values

        # First value has no carryover
        adstocked[0] = spend_array[0]

        # Apply recursive formula
        for t in range(1, len(spend_array)):
            adstocked[t] = spend_array[t] + (decay_rate * adstocked[t-1])

        return pd.Series(adstocked, index=spend_series.index)

    def apply_saturation(
        self,
        spend_series: pd.Series,
        transformation: str = 'log'
    ) -> pd.Series:
        """
        Apply saturation transformation to model diminishing returns.

        Why we need this:
            - First $100 of spend has more impact than the 10,000th $100
            - Linear models assume constant returns (not realistic)
            - Log transformation captures this concave relationship

        Mathematical formula (log):
            Saturated_Spend = log(Spend + 1)

        The +1 ensures we can handle zero spend days.

        Args:
            spend_series: Daily spend values
            transformation: Type of transformation ('log' supported for now)

        Returns:
            pd.Series with saturated values

        Example:
            Original: [0, 100, 1000, 10000]
            Log:      [0, 4.6, 6.9, 9.2]

            Notice: Doubling from 100→1000 gives +2.3 impact
                   But doubling from 1000→10000 gives only +2.3 impact
                   (Same increase, diminishing marginal returns)
        """
        if transformation == 'log':
            # Add 1 to handle zero spend, then log transform
            saturated = np.log1p(spend_series)  # log1p(x) = log(1 + x)
            return saturated
        else:
            raise ValueError(f"Transformation '{transformation}' not supported. Use 'log'.")

    def create_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create time-based features for seasonality and trends.

        Features created:
        1. Day of week (0=Monday, 6=Sunday)
        2. Is weekend (binary)
        3. Month (1-12)
        4. Is holiday period (binary)
        5. Time trend (0, 1, 2, ... for each day)

        Why we need these:
            - Control for natural seasonality (weekends, holidays)
            - Separate time effects from marketing effects
            - Avoid attributing seasonal lift to campaigns

        Args:
            df: DataFrame with 'date' column

        Returns:
            DataFrame with additional time features
        """
        df = df.copy()

        # Ensure date is datetime
        if not pd.api.types.is_datetime64_any_dtype(df['date']):
            df['date'] = pd.to_datetime(df['date'])

        # 1. Day of week (as categorical dummy variables)
        df['day_of_week'] = df['date'].dt.dayofweek

        # Create dummy variables for each day (drop Sunday as reference)
        day_dummies = pd.get_dummies(df['day_of_week'], prefix='dow', drop_first=True)
        df = pd.concat([df, day_dummies], axis=1)

        # 2. Weekend indicator
        df['is_weekend'] = (df['day_of_week'].isin([5, 6])).astype(int)

        # 3. Month dummies (drop December as reference)
        df['month'] = df['date'].dt.month
        month_dummies = pd.get_dummies(df['month'], prefix='month', drop_first=True)
        df = pd.concat([df, month_dummies], axis=1)

        # 4. Holiday indicator (simplified - major US holidays)
        df['is_holiday'] = 0

        # Black Friday week (late November)
        df.loc[(df['month'] == 11) & (df['date'].dt.day >= 23) &
               (df['date'].dt.day <= 27), 'is_holiday'] = 1

        # Christmas week (late December)
        df.loc[(df['month'] == 12) & (df['date'].dt.day >= 20), 'is_holiday'] = 1

        # 5. Time trend (linear trend variable)
        df = df.sort_values('date').reset_index(drop=True)
        df['time_trend'] = range(len(df))

        return df

    def create_campaign_features(
        self,
        df: pd.DataFrame,
        campaign_col: str = 'campaign_name'
    ) -> pd.DataFrame:
        """
        Transform campaign spend data into adstocked and saturated features.

        Process:
        1. Pivot data so each campaign becomes a column
        2. Apply adstock transformation to each campaign
        3. Apply saturation (log) transformation
        4. Handle missing values (fill with 0)

        Args:
            df: DataFrame with date, campaign_name, spend columns
            campaign_col: Name of campaign column

        Returns:
            DataFrame with one row per date and columns for each campaign's
            transformed spend (e.g., 'Brand_Awareness_Q1_adstock_log')
        """
        df = df.copy()

        # Ensure date is datetime
        df['date'] = pd.to_datetime(df['date'])

        # Get unique campaigns
        campaigns = df[campaign_col].unique()
        self.campaign_names_ = campaigns

        print(f"\n{'='*60}")
        print("CREATING CAMPAIGN FEATURES")
        print(f"{'='*60}")
        print(f"Campaigns found: {len(campaigns)}")
        for camp in campaigns:
            print(f"  - {camp}")

        # Pivot to wide format (one row per date, one column per campaign)
        df_pivot = df.pivot_table(
            index='date',
            columns=campaign_col,
            values='spend',
            aggfunc='sum',
            fill_value=0
        )

        print(f"\nPivoted shape: {df_pivot.shape}")
        print(f"Date range: {df_pivot.index.min()} to {df_pivot.index.max()}")

        # Apply transformations to each campaign
        transformed_features = {'date': df_pivot.index}

        for campaign in campaigns:
            print(f"\nTransforming: {campaign}")

            # Get spend series for this campaign
            spend = df_pivot[campaign]

            # Get campaign-specific decay rate, or use default
            decay_rate = self.adstock_params.get(campaign, 0.5)
            print(f"  - Applying adstock (decay={decay_rate})")

            # Step 1: Apply adstock
            adstocked = self.apply_adstock(spend, decay_rate=decay_rate)

            # Step 2: Apply saturation (log)
            if self.use_saturation:
                print(f"  - Applying saturation (log transformation)")
                saturated = self.apply_saturation(adstocked)
                feature_name = f"{campaign}_adstock_log"
            else:
                saturated = adstocked
                feature_name = f"{campaign}_adstock"

            transformed_features[feature_name] = saturated.values

            # Print summary statistics
            print(f"  - Original spend range: ${spend.min():.0f} to ${spend.max():.0f}")
            print(f"  - Transformed range: {saturated.min():.2f} to {saturated.max():.2f}")

        # Create DataFrame from transformed features
        df_features = pd.DataFrame(transformed_features)

        print(f"\n{'='*60}")
        print(f"Campaign features created: {len(campaigns)} features")
        print(f"{'='*60}\n")

        return df_features

    def fit_transform(
        self,
        df: pd.DataFrame,
        target_col: str = 'revenue'
    ) -> pd.DataFrame:
        """
        Complete feature engineering pipeline.

        Steps:
        1. Create campaign features (adstock + saturation)
        2. Aggregate target variable by date
        3. Add time-based features
        4. Merge everything together

        Args:
            df: Cleaned data with columns: date, campaign_name, spend, revenue
            target_col: Name of target variable (usually 'revenue')

        Returns:
            DataFrame ready for modeling with:
            - Date
            - Transformed campaign spend features
            - Time features (if enabled)
            - Target variable (revenue)
        """
        print("\n" + "="*70)
        print(" "*20 + "MMM FEATURE ENGINEERING")
        print("="*70)

        # Step 1: Create campaign features
        df_campaigns = self.create_campaign_features(df)

        # Step 2: Aggregate target variable by date
        df_target = df.groupby('date')[target_col].sum().reset_index()
        df_target.columns = ['date', 'revenue']

        print(f"Aggregated revenue by date: {len(df_target)} days")

        # Step 3: Merge campaign features with target
        df_final = df_campaigns.merge(df_target, on='date', how='left')

        # Step 4: Add time features
        if self.include_time_features:
            print("\nAdding time-based features...")
            df_final = self.create_time_features(df_final)
            print("  - Day of week dummies")
            print("  - Weekend indicator")
            print("  - Month dummies")
            print("  - Holiday indicator")
            print("  - Time trend")

        # Store feature column names (excluding date and target)
        self.feature_columns_ = [col for col in df_final.columns
                                if col not in ['date', 'revenue', 'day_of_week', 'month']]

        print(f"\n{'='*70}")
        print(f"FEATURE ENGINEERING COMPLETE")
        print(f"Final dataset shape: {df_final.shape}")
        print(f"Number of features: {len(self.feature_columns_)}")
        print(f"Target variable: revenue")
        print(f"{'='*70}\n")

        return df_final

    def train_test_split(
        self,
        df: pd.DataFrame,
        test_size: float = 0.2,
        target_col: str = 'revenue'
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Create time-based train/test split.

        CRITICAL: For time series data, we MUST NOT use random splits!
        - Random splits leak future information into training
        - We want to predict the future, so test set must be chronologically last

        Strategy:
            Train: First (1 - test_size) of data
            Test: Last test_size of data

        Args:
            df: DataFrame with features and target
            test_size: Proportion of data for testing (default 0.2 = last 20%)
            target_col: Name of target variable

        Returns:
            X_train, X_test, y_train, y_test

        Example:
            If you have 90 days and test_size=0.2:
            - Train: Days 0-71 (72 days, 80%)
            - Test: Days 72-89 (18 days, 20%)
        """
        # Sort by date to ensure chronological order
        df = df.sort_values('date').reset_index(drop=True)

        # Calculate split point
        n_samples = len(df)
        split_idx = int(n_samples * (1 - test_size))

        # Split into train and test
        train_df = df.iloc[:split_idx].copy()
        test_df = df.iloc[split_idx:].copy()

        # Separate features and target
        X_train = train_df[self.feature_columns_]
        X_test = test_df[self.feature_columns_]
        y_train = train_df[target_col]
        y_test = test_df[target_col]

        print("\n" + "="*70)
        print("TIME-BASED TRAIN/TEST SPLIT")
        print("="*70)
        print(f"Total samples: {n_samples}")
        print(f"Train samples: {len(train_df)} ({len(train_df)/n_samples*100:.1f}%)")
        print(f"Test samples: {len(test_df)} ({len(test_df)/n_samples*100:.1f}%)")
        print(f"\nTrain date range: {train_df['date'].min()} to {train_df['date'].max()}")
        print(f"Test date range: {test_df['date'].min()} to {test_df['date'].max()}")
        print(f"\nFeatures: {len(self.feature_columns_)}")
        print(f"Target: {target_col}")
        print("="*70 + "\n")

        return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    """
    Test the feature engineering pipeline.
    """
    # This would normally import from your data pipeline
    # For now, create simple test data

    print("Testing MMM Feature Engineering Pipeline")
    print("="*70)

    # Create sample data
    dates = pd.date_range('2024-01-01', periods=90, freq='D')
    campaigns = ['Campaign_A', 'Campaign_B', 'Campaign_C']

    data = []
    for date in dates:
        for campaign in campaigns:
            data.append({
                'date': date,
                'campaign_name': campaign,
                'spend': np.random.uniform(100, 1000),
                'revenue': np.random.uniform(200, 2000)
            })

    df_test = pd.DataFrame(data)

    print(f"\nTest data shape: {df_test.shape}")
    print(f"Date range: {df_test['date'].min()} to {df_test['date'].max()}")

    # Initialize feature engineer with custom adstock parameters
    adstock_params = {
        'Campaign_A': 0.7,  # Strong carryover (brand building)
        'Campaign_B': 0.5,  # Medium carryover
        'Campaign_C': 0.3   # Low carryover (direct response)
    }

    engineer = MMMFeatureEngineer(
        adstock_params=adstock_params,
        apply_saturation=True,
        include_time_features=True
    )

    # Transform data
    df_transformed = engineer.fit_transform(df_test)

    print("\nTransformed data sample:")
    print(df_transformed.head())

    print("\nAll features:")
    print(engineer.feature_columns_)

    # Create train/test split
    X_train, X_test, y_train, y_test = engineer.train_test_split(df_transformed)

    print("\n✅ Feature engineering pipeline test completed successfully!")