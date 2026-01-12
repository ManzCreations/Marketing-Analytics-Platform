"""
Marketing Mix Model (MMM) Implementation

This module implements a regression-based Marketing Mix Model using Ridge Regression.
The model attributes revenue to marketing channels while controlling for seasonality
and other external factors.

Key Features:
- Ridge Regression (handles multicollinearity)
- Hyperparameter tuning with cross-validation
- Channel contribution analysis
- ROI/ROAS calculation by channel
- Model diagnostics and evaluation
- Scenario analysis capabilities

"""

import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge, ElasticNet, Lasso
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from typing import Dict, List, Tuple, Optional
import warnings

warnings.filterwarnings('ignore')


class MarketingMixModel:
    """
    Marketing Mix Model using regularized regression.

    This class implements a complete MMM pipeline:
    1. Model training with hyperparameter optimization
    2. Performance evaluation
    3. Channel contribution decomposition
    4. ROI/ROAS calculation
    5. Scenario analysis

    Attributes:
        model: Trained scikit-learn regression model
        scaler: StandardScaler for feature normalization
        feature_names: List of feature names
        coefficients: Model coefficients (one per feature)
        baseline_revenue: Intercept (revenue with $0 spend)

    Example:
        mmm = MarketingMixModel(model_type='ridge')
        mmm.fit(X_train, y_train, campaign_features=campaign_cols)
        mmm.evaluate(X_test, y_test)
        contributions = mmm.get_channel_contributions(X_test)
    """

    def __init__(
            self,
            model_type: str = 'ridge',
            alpha: Optional[float] = None,
            scale_features: bool = True
    ):
        """
        Initialize the MMM.

        Args:
            model_type: Type of regression ('ridge', 'lasso', 'elasticnet')
            alpha: Regularization strength (if None, will be tuned)
            scale_features: Whether to standardize features before modeling
        """
        self.model_type = model_type.lower()
        self.alpha = alpha
        self.scale_features = scale_features

        # Will be set during training
        self.model = None
        self.scaler = StandardScaler() if scale_features else None
        self.feature_names = None
        self.campaign_features = None
        self.best_params_ = None
        self.cv_results_ = None

        # Performance metrics
        self.metrics_ = {}

    def _initialize_model(self, alpha: float = 1.0):
        """
        Create the regression model.

        Args:
            alpha: Regularization strength

        Returns:
            Initialized model
        """
        if self.model_type == 'ridge':
            return Ridge(alpha=alpha, random_state=42)
        elif self.model_type == 'lasso':
            return Lasso(alpha=alpha, random_state=42, max_iter=10000)
        elif self.model_type == 'elasticnet':
            return ElasticNet(alpha=alpha, l1_ratio=0.5, random_state=42, max_iter=10000)
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")

    def fit(
            self,
            X_train: pd.DataFrame,
            y_train: pd.Series,
            campaign_features: Optional[List[str]] = None,
            tune_hyperparameters: bool = True,
            cv_folds: int = 5
    ):
        """
        Train the MMM.

        Process:
        1. Store feature names
        2. Scale features (if enabled)
        3. Tune hyperparameters (if enabled)
        4. Train final model

        Args:
            X_train: Training features
            y_train: Training target (revenue)
            campaign_features: List of campaign feature names 
                             (e.g., ['Brand_Awareness_adstock_log'])
            tune_hyperparameters: Whether to perform grid search
            cv_folds: Number of cross-validation folds
        """
        print("\n" + "=" * 70)
        print(" " * 20 + "TRAINING MMM MODEL")
        print("=" * 70)

        # Store feature information
        self.feature_names = list(X_train.columns)
        self.campaign_features = campaign_features

        if self.campaign_features is None:
            # Auto-detect campaign features (those with 'adstock' in name)
            self.campaign_features = [col for col in self.feature_names
                                      if 'adstock' in col.lower()]
            print(f"\nAuto-detected {len(self.campaign_features)} campaign features")

        print(f"\nModel Configuration:")
        print(f"  Model type: {self.model_type.upper()}")
        print(f"  Training samples: {len(X_train)}")
        print(f"  Features: {len(self.feature_names)}")
        print(f"  Campaign features: {len(self.campaign_features)}")
        print(f"  Scale features: {self.scale_features}")

        # Scale features if enabled
        X_train_transformed = X_train.copy()
        if self.scale_features:
            print(f"\nScaling features...")
            X_train_transformed = pd.DataFrame(
                self.scaler.fit_transform(X_train),
                columns=X_train.columns,
                index=X_train.index
            )
            print("  Features standardized (mean=0, std=1)")

        # Hyperparameter tuning
        if tune_hyperparameters and self.alpha is None:
            print(f"\n{'=' * 70}")
            print("HYPERPARAMETER TUNING")
            print(f"{'=' * 70}")

            # Use TimeSeriesSplit for cross-validation (respects temporal order)
            tscv = TimeSeriesSplit(n_splits=cv_folds)

            # Define parameter grid
            if self.model_type == 'ridge':
                param_grid = {
                    'alpha': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
                }
            elif self.model_type == 'lasso':
                param_grid = {
                    'alpha': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
                }
            elif self.model_type == 'elasticnet':
                param_grid = {
                    'alpha': [0.001, 0.01, 0.1, 1.0, 10.0],
                    'l1_ratio': [0.3, 0.5, 0.7, 0.9]
                }

            print(f"\nSearching over {len(param_grid)} parameters...")
            print(f"Using {cv_folds}-fold Time Series Cross-Validation")

            # Initialize base model
            base_model = self._initialize_model()

            # Grid search
            grid_search = GridSearchCV(
                base_model,
                param_grid,
                cv=tscv,
                scoring='neg_mean_absolute_error',
                n_jobs=-1,
                verbose=0
            )

            grid_search.fit(X_train_transformed, y_train)

            # Store results
            self.best_params_ = grid_search.best_params_
            self.cv_results_ = pd.DataFrame(grid_search.cv_results_)

            print(f"\n Best parameters found:")
            for param, value in self.best_params_.items():
                print(f"    {param}: {value}")
            print(f"  Cross-validation MAE: ${-grid_search.best_score_:,.2f}")

            # Use best model
            self.model = grid_search.best_estimator_
            self.alpha = self.best_params_['alpha']

        else:
            # Train with specified alpha
            alpha_to_use = self.alpha if self.alpha is not None else 1.0
            print(f"\nTraining with alpha={alpha_to_use}")

            self.model = self._initialize_model(alpha=alpha_to_use)
            self.model.fit(X_train_transformed, y_train)
            self.alpha = alpha_to_use

        # Extract coefficients
        print(f"\n{'=' * 70}")
        print("MODEL TRAINING COMPLETE")
        print(f"{'=' * 70}")
        print(f"\nModel coefficients extracted:")
        print(f"  Intercept (baseline revenue): ${self.model.intercept_:,.2f}")
        print(f"  Feature coefficients: {len(self.model.coef_)}")

        # Show campaign coefficients
        print(f"\n  Campaign Coefficients (impact on revenue):")
        for feat in self.campaign_features:
            idx = self.feature_names.index(feat)
            coef = self.model.coef_[idx]
            print(f"    {feat:45s}: {coef:+,.2f}")

        print(f"\n{'=' * 70}\n")

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions.

        Args:
            X: Features

        Returns:
            Predicted revenue
        """
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")

        # Scale if needed
        X_transformed = X.copy()
        if self.scale_features:
            X_transformed = pd.DataFrame(
                self.scaler.transform(X),
                columns=X.columns,
                index=X.index
            )

        return self.model.predict(X_transformed)

    def evaluate(
            self,
            X_test: pd.DataFrame,
            y_test: pd.Series,
            print_results: bool = True
    ) -> Dict[str, float]:
        """
        Evaluate model performance on test set.

        Metrics calculated:
        - R² (coefficient of determination)
        - MAE (Mean Absolute Error)
        - RMSE (Root Mean Squared Error)
        - MAPE (Mean Absolute Percentage Error)

        Args:
            X_test: Test features
            y_test: True revenue values
            print_results: Whether to print results

        Returns:
            Dictionary of metrics
        """
        # Make predictions
        y_pred = self.predict(X_test)

        # Calculate metrics
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mape = np.mean(np.abs((y_test - y_pred) / y_test))

        # Store metrics
        self.metrics_ = {
            'r2': r2,
            'mae': mae,
            'rmse': rmse,
            'mape': mape,
            'n_samples': len(y_test)
        }

        if print_results:
            print("\n" + "=" * 70)
            print(" " * 20 + "MODEL EVALUATION")
            print("=" * 70)
            print(f"\nTest Set Performance:")
            print(f"  Samples: {len(y_test)}")
            print(f"\n  R² Score:  {r2:.4f}")
            print(f"    → Model explains {r2 * 100:.1f}% of revenue variance")
            print(f"\n  MAE:  ${mae:,.2f}")
            print(f"    → Average prediction error")
            print(f"\n  RMSE: ${rmse:,.2f}")
            print(f"    → Root Mean Squared Error")
            print(f"\n  MAPE: {mape:.2f}%")
            print(f"    → Mean Absolute Percentage Error")

            # Actual vs Predicted summary
            print(f"\n  Actual Revenue Range:    ${y_test.min():,.2f} to ${y_test.max():,.2f}")
            print(f"  Predicted Revenue Range: ${y_pred.min():,.2f} to ${y_pred.max():,.2f}")
            print(f"  Mean Actual:   ${y_test.mean():,.2f}")
            print(f"  Mean Predicted: ${y_pred.mean():,.2f}")

            print(f"\n{'=' * 70}\n")

        return self.metrics_

    def get_channel_contributions(
            self,
            X: pd.DataFrame,
            original_spend_data: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Calculate how much each marketing channel contributed to revenue.

        This is the core MMM insight: attributing revenue to channels.

        Formula:
            Channel Contribution = Coefficient × Feature Value

        Args:
            X: Features (should include campaign features)
            original_spend_data: Optional DataFrame with original spend by campaign
                               (for ROI calculation)

        Returns:
            DataFrame with channel contributions, percentages, and ROI
        """
        print("\n" + "=" * 70)
        print(" " * 15 + "CHANNEL CONTRIBUTION ANALYSIS")
        print("=" * 70)

        # Scale features if needed
        X_transformed = X.copy()
        if self.scale_features:
            X_transformed = pd.DataFrame(
                self.scaler.transform(X),
                columns=X.columns,
                index=X.index
            )

        # Calculate baseline (intercept)
        baseline_revenue = self.model.intercept_ * len(X)

        # Calculate contribution from each feature
        contributions = {}

        for feat in self.feature_names:
            idx = self.feature_names.index(feat)
            coef = self.model.coef_[idx]
            feature_values = X_transformed[feat].values

            # Contribution = coefficient × feature value, summed over all samples
            contribution = coef * feature_values.sum()
            contributions[feat] = contribution

        # Create DataFrame
        df_contrib = pd.DataFrame([contributions]).T
        df_contrib.columns = ['Total_Contribution']
        df_contrib['Feature'] = df_contrib.index
        df_contrib = df_contrib.reset_index(drop=True)

        # Calculate percentages
        total_revenue = df_contrib['Total_Contribution'].sum() + baseline_revenue
        df_contrib['Percentage'] = (df_contrib['Total_Contribution'] / total_revenue)

        # Filter to campaign features only
        df_campaigns = df_contrib[df_contrib['Feature'].isin(self.campaign_features)].copy()

        # Calculate ROI if original spend data provided
        if original_spend_data is not None:
            print("\nCalculating ROI with original spend data...")

            # Map campaign features to spend columns
            df_campaigns['Campaign'] = df_campaigns['Feature'].apply(
                lambda x: x.replace('_adstock_log', '').replace('_adstock', '')
            )

            # Sum spend by campaign
            spend_by_campaign = {}
            for campaign in df_campaigns['Campaign']:
                if campaign in original_spend_data.columns:
                    spend_by_campaign[campaign] = original_spend_data[campaign].sum()

            df_campaigns['Total_Spend'] = df_campaigns['Campaign'].map(spend_by_campaign)
            df_campaigns['ROI'] = df_campaigns['Total_Contribution'] / df_campaigns['Total_Spend']
            df_campaigns['ROAS'] = df_campaigns['ROI']  # Same thing

        # Sort by contribution
        df_campaigns = df_campaigns.sort_values('Total_Contribution', ascending=False)

        # Print results
        print(f"\n{'=' * 70}")
        print("CAMPAIGN CONTRIBUTIONS")
        print(f"{'=' * 70}\n")
        print(f"Baseline Revenue (intercept): ${baseline_revenue:,.2f}")
        print(f"  → Revenue with $0 marketing spend")
        print(f"\nTotal Revenue: ${total_revenue:,.2f}")
        print(f"\nMarketing-Driven Revenue: ${df_campaigns['Total_Contribution'].sum():,.2f}")
        print(f"  → {df_campaigns['Total_Contribution'].sum() / total_revenue * 100:.1f}% of total")

        print(f"\n{'=' * 70}")
        print("BY CAMPAIGN:")
        print(f"{'=' * 70}\n")

        for _, row in df_campaigns.iterrows():
            print(f"{row['Feature']:45s}")
            print(f"  Contribution: ${row['Total_Contribution']:,.2f} ({row['Percentage']:.1f}%)")
            if 'Total_Spend' in row:
                print(f"  Spend:        ${row['Total_Spend']:,.2f}")
                print(f"  ROI/ROAS:     {row['ROI']:.2f}x")
            print()

        print(f"{'=' * 70}\n")

        return df_campaigns

    def get_coefficients_df(self) -> pd.DataFrame:
        """
        Get model coefficients in a nice DataFrame.

        Returns:
            DataFrame with features and their coefficients
        """
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")

        df_coef = pd.DataFrame({
            'Feature': self.feature_names,
            'Coefficient': self.model.coef_
        })

        df_coef['Abs_Coefficient'] = np.abs(df_coef['Coefficient'])
        df_coef = df_coef.sort_values('Abs_Coefficient', ascending=False)
        df_coef = df_coef.drop('Abs_Coefficient', axis=1)

        return df_coef

    def predict_scenario(
            self,
            X_base: pd.DataFrame,
            spend_changes: Dict[str, float]
    ) -> Dict[str, float]:
        """
        What-if scenario analysis: What happens if we change spend?

        Args:
            X_base: Base features (e.g., from recent period)
            spend_changes: Dict mapping campaign names to spend multipliers
                         e.g., {'Brand_Awareness_Q1': 1.2} = increase by 20%

        Returns:
            Dictionary with scenario results
        """
        print("\n" + "=" * 70)
        print(" " * 20 + "SCENARIO ANALYSIS")
        print("=" * 70)

        # Make base prediction
        base_revenue = self.predict(X_base).sum()

        print(f"\nBase scenario:")
        print(f"  Total revenue: ${base_revenue:,.2f}")

        # Create modified scenario
        X_scenario = X_base.copy()

        print(f"\nApplying changes:")
        for campaign, multiplier in spend_changes.items():
            # Find the feature name
            feature_name = None
            for feat in self.campaign_features:
                if campaign in feat:
                    feature_name = feat
                    break

            if feature_name:
                original_values = X_scenario[feature_name].copy()
                # Scale the feature by multiplier
                X_scenario[feature_name] = original_values * multiplier
                print(f"  {campaign}: {multiplier:.0%} of original")

        # Predict with new scenario
        scenario_revenue = self.predict(X_scenario).sum()

        revenue_change = scenario_revenue - base_revenue
        pct_change = (revenue_change / base_revenue) * 100

        print(f"\nScenario results:")
        print(f"  New revenue: ${scenario_revenue:,.2f}")
        print(f"  Change: ${revenue_change:+,.2f} ({pct_change:+.1f}%)")

        print(f"\n{'=' * 70}\n")

        return {
            'base_revenue': base_revenue,
            'scenario_revenue': scenario_revenue,
            'revenue_change': revenue_change,
            'pct_change': pct_change
        }


if __name__ == "__main__":
    """
    Test the MMM with sample data.
    """
    print("=" * 80)
    print(" " * 25 + "TESTING MMM MODEL")
    print("=" * 80)

    # Create sample data
    np.random.seed(42)
    n_samples = 100

    # Features
    X = pd.DataFrame({
        'Campaign_A_adstock_log': np.random.uniform(5, 8, n_samples),
        'Campaign_B_adstock_log': np.random.uniform(4, 7, n_samples),
        'Campaign_C_adstock_log': np.random.uniform(3, 6, n_samples),
        'is_weekend': np.random.randint(0, 2, n_samples),
        'time_trend': range(n_samples)
    })

    # Target (revenue) - influenced by campaigns
    y = (1000 +  # baseline
         500 * X['Campaign_A_adstock_log'] +
         300 * X['Campaign_B_adstock_log'] +
         200 * X['Campaign_C_adstock_log'] +
         -500 * X['is_weekend'] +
         10 * X['time_trend'] +
         np.random.normal(0, 1000, n_samples))

    # Train/test split
    train_size = int(0.8 * n_samples)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # Train model
    mmm = MarketingMixModel(model_type='ridge', scale_features=True)

    campaign_features = [
        'Campaign_A_adstock_log',
        'Campaign_B_adstock_log',
        'Campaign_C_adstock_log'
    ]

    mmm.fit(X_train, y_train, campaign_features=campaign_features, tune_hyperparameters=True)

    # Evaluate
    metrics = mmm.evaluate(X_test, y_test)

    # Get contributions
    contributions = mmm.get_channel_contributions(X_test)

    print("\n MMM model test completed successfully!")