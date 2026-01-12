"""
Budget Optimization Module for Marketing Mix Modeling

This module uses MMM results to optimize budget allocation across channels.
It accounts for diminishing returns and provides actionable recommendations.

Key Features:
- Optimal budget allocation given total budget constraint
- Respects channel min/max spend limits
- Accounts for saturation (diminishing returns)
- Provides scenario comparisons
- Calculates expected ROI improvements
"""

import pandas as pd
import numpy as np
from scipy.optimize import minimize, Bounds, LinearConstraint
from typing import Dict, List, Tuple, Optional
import warnings

warnings.filterwarnings('ignore')


class BudgetOptimizer:
    """
    Optimize marketing budget allocation using MMM results.

    Given:
    - Current spend by channel
    - Model coefficients (marginal impact per channel)
    - Total budget constraint
    - Channel min/max constraints

    Finds:
    - Optimal spend allocation that maximizes predicted revenue

    Example:
        optimizer = BudgetOptimizer(mmm_model)
        optimal_budget = optimizer.optimize(
            total_budget=100000,
            current_spend={'Campaign_A': 30000, 'Campaign_B': 20000}
        )
    """

    def __init__(
            self,
            mmm_model,
            feature_engineer=None
    ):
        """
        Initialize the optimizer.

        Args:
            mmm_model: Trained MarketingMixModel instance
            feature_engineer: MMMFeatureEngineer instance (for transformations)
        """
        self.mmm_model = mmm_model
        self.feature_engineer = feature_engineer

        # Extract model information
        self.coefficients = {}
        self.campaign_features = mmm_model.campaign_features

        # Map campaign features to coefficients
        for feat in self.campaign_features:
            idx = mmm_model.feature_names.index(feat)
            coef = mmm_model.model.coef_[idx]
            self.coefficients[feat] = coef

    def _apply_transformations(
            self,
            spend: float,
            campaign_feature: str
    ) -> float:
        """
        Apply adstock and saturation transformations to raw spend.

        This simulates what the model sees after feature engineering.

        Args:
            spend: Raw spend amount
            campaign_feature: Name of campaign feature

        Returns:
            Transformed spend value
        """
        # For optimization, we use simplified transformation
        # In practice, we'd need to account for full adstock history

        # Apply saturation (log transformation)
        transformed = np.log1p(spend)

        return transformed

    def _predict_revenue_from_spend(
            self,
            spend_allocation: Dict[str, float],
            baseline_features: Optional[pd.DataFrame] = None
    ) -> float:
        """
        Predict revenue given a spend allocation.

        Args:
            spend_allocation: Dict mapping campaign names to spend amounts
            baseline_features: Optional baseline features (time controls, etc.)

        Returns:
            Predicted revenue
        """
        # Start with baseline (intercept)
        predicted_revenue = self.mmm_model.model.intercept_

        # Add contribution from each campaign
        for campaign_feature, spend in spend_allocation.items():
            # Transform spend
            transformed_spend = self._apply_transformations(spend, campaign_feature)

            # Get coefficient
            coef = self.coefficients[campaign_feature]

            # Add contribution
            predicted_revenue += coef * transformed_spend

        # If we have baseline features (time controls), add those too
        # For simplicity, we assume average effect of time controls
        # In production, you'd specify the time period you're optimizing for

        return predicted_revenue

    def optimize(
            self,
            total_budget: float,
            current_spend: Optional[Dict[str, float]] = None,
            min_spend_pct: float = 0.05,
            max_spend_pct: float = 0.60,
            channel_bounds: Optional[Dict[str, Tuple[float, float]]] = None
    ) -> Dict:
        """
        Find optimal budget allocation.

        Problem formulation:
            Maximize: Predicted Revenue
            Subject to:
                - Sum of all spend = total_budget
                - min_spend <= spend_i <= max_spend for each channel

        Args:
            total_budget: Total marketing budget to allocate
            current_spend: Current spend by channel (for comparison)
            min_spend_pct: Minimum % of budget per channel (default 5%)
            max_spend_pct: Maximum % of budget per channel (default 60%)
            channel_bounds: Optional dict with custom bounds per channel
                          e.g., {'Campaign_A': (5000, 50000)}

        Returns:
            Dict with optimal allocation, expected revenue, comparison to current
        """
        print("\n" + "=" * 70)
        print(" " * 20 + "BUDGET OPTIMIZATION")
        print("=" * 70)

        print(f"\nOptimization Settings:")
        print(f"  Total Budget: ${total_budget:,.2f}")
        print(f"  Channels: {len(self.campaign_features)}")
        print(f"  Min spend per channel: {min_spend_pct * 100:.0f}% of budget")
        print(f"  Max spend per channel: {max_spend_pct * 100:.0f}% of budget")

        # Get campaign names (remove the _adstock_log suffix)
        campaign_names = []
        for feat in self.campaign_features:
            name = feat.replace('_adstock_log', '').replace('_adstock', '')
            campaign_names.append(name)

        n_channels = len(self.campaign_features)

        # Set up bounds for each channel
        if channel_bounds is None:
            # Use percentage-based bounds
            min_spend = total_budget * min_spend_pct
            max_spend = total_budget * max_spend_pct
            bounds = Bounds(
                lb=[min_spend] * n_channels,
                ub=[max_spend] * n_channels
            )
        else:
            # Use custom bounds
            lb = []
            ub = []
            for feat in self.campaign_features:
                campaign = feat.replace('_adstock_log', '').replace('_adstock', '')
                if campaign in channel_bounds:
                    lb.append(channel_bounds[campaign][0])
                    ub.append(channel_bounds[campaign][1])
                else:
                    lb.append(total_budget * min_spend_pct)
                    ub.append(total_budget * max_spend_pct)
            bounds = Bounds(lb=lb, ub=ub)

        # Constraint: sum of all spend = total_budget
        budget_constraint = LinearConstraint(
            A=np.ones(n_channels),
            lb=total_budget,
            ub=total_budget
        )

        # Objective function: maximize revenue (minimize negative revenue)
        def objective(spend_array):
            spend_dict = {}
            for i, feat in enumerate(self.campaign_features):
                spend_dict[feat] = spend_array[i]

            revenue = self._predict_revenue_from_spend(spend_dict)
            return -revenue  # Minimize negative = maximize positive

        # Initial guess: equal allocation
        x0 = np.array([total_budget / n_channels] * n_channels)

        print(f"\nRunning optimization...")

        # Optimize
        result = minimize(
            objective,
            x0=x0,
            method='SLSQP',
            bounds=bounds,
            constraints=[budget_constraint],
            options={'maxiter': 1000, 'ftol': 1e-6}
        )

        if not result.success:
            print(f"  ⚠️  Warning: Optimization did not fully converge")
            print(f"     Message: {result.message}")
        else:
            print(f"  ✓ Optimization converged successfully")

        # Extract optimal allocation
        optimal_spend = {}
        for i, feat in enumerate(self.campaign_features):
            campaign = feat.replace('_adstock_log', '').replace('_adstock', '')
            optimal_spend[campaign] = result.x[i]

        # Calculate predicted revenue
        optimal_revenue = -result.fun  # Negate because we minimized negative revenue

        # Compare to current if provided
        comparison = {}
        if current_spend is not None:
            # Calculate current revenue
            current_spend_dict = {}
            for feat in self.campaign_features:
                campaign = feat.replace('_adstock_log', '').replace('_adstock', '')
                current_spend_dict[feat] = current_spend.get(campaign, 0)

            current_revenue = self._predict_revenue_from_spend(current_spend_dict)
            current_total_spend = sum(current_spend.values())

            # Calculate improvements
            revenue_increase = optimal_revenue - current_revenue
            revenue_increase_pct = (revenue_increase / current_revenue) * 100

            comparison = {
                'current_revenue': current_revenue,
                'current_total_spend': current_total_spend,
                'optimal_revenue': optimal_revenue,
                'revenue_increase': revenue_increase,
                'revenue_increase_pct': revenue_increase_pct
            }

        # Print results
        self._print_results(
            optimal_spend,
            optimal_revenue,
            current_spend,
            comparison
        )

        # Return comprehensive results
        return {
            'optimal_spend': optimal_spend,
            'optimal_revenue': optimal_revenue,
            'total_budget': total_budget,
            'comparison': comparison,
            'optimization_success': result.success,
            'optimization_message': result.message
        }

    def _print_results(
            self,
            optimal_spend: Dict[str, float],
            optimal_revenue: float,
            current_spend: Optional[Dict[str, float]],
            comparison: Dict
    ):
        """Print optimization results in a nice format."""

        print(f"\n{'=' * 70}")
        print("OPTIMAL BUDGET ALLOCATION")
        print(f"{'=' * 70}\n")

        # Create comparison DataFrame
        df_comparison = pd.DataFrame({
            'Campaign': list(optimal_spend.keys()),
            'Optimal_Spend': list(optimal_spend.values())
        })

        if current_spend is not None:
            df_comparison['Current_Spend'] = df_comparison['Campaign'].map(current_spend).fillna(0)
            df_comparison['Change'] = df_comparison['Optimal_Spend'] - df_comparison['Current_Spend']
            df_comparison['Change_Pct'] = (df_comparison['Change'] / df_comparison['Current_Spend'] * 100).fillna(0)

        df_comparison['Optimal_Pct'] = (df_comparison['Optimal_Spend'] / df_comparison['Optimal_Spend'].sum() * 100)

        # Sort by optimal spend
        df_comparison = df_comparison.sort_values('Optimal_Spend', ascending=False)

        # Print table
        for _, row in df_comparison.iterrows():
            print(f"{row['Campaign']:40s}")
            print(f"  Optimal: ${row['Optimal_Spend']:>12,.2f} ({row['Optimal_Pct']:>5.1f}% of budget)")
            if current_spend is not None and row['Campaign'] in current_spend:
                print(f"  Current: ${row['Current_Spend']:>12,.2f}")
                print(f"  Change:  ${row['Change']:>+12,.2f} ({row['Change_Pct']:>+6.1f}%)")
            print()

        print(f"{'=' * 70}")
        print(f"Expected Revenue: ${optimal_revenue:,.2f}")

        if comparison:
            print(f"\nComparison to Current Allocation:")
            print(f"  Current Revenue:  ${comparison['current_revenue']:,.2f}")
            print(f"  Optimal Revenue:  ${comparison['optimal_revenue']:,.2f}")
            print(
                f"  Improvement:      ${comparison['revenue_increase']:+,.2f} ({comparison['revenue_increase_pct']:+.1f}%)")

        print(f"{'=' * 70}\n")

    def scenario_analysis(
            self,
            budget_scenarios: List[float],
            current_spend: Optional[Dict[str, float]] = None
    ) -> pd.DataFrame:
        """
        Run optimization for multiple budget levels.

        This answers: "What if we had 10% more budget? 20% more?"

        Args:
            budget_scenarios: List of budget amounts to test
            current_spend: Current spend for comparison

        Returns:
            DataFrame with results for each scenario
        """
        print("\n" + "=" * 70)
        print(" " * 20 + "SCENARIO ANALYSIS")
        print("=" * 70)

        results = []

        for budget in budget_scenarios:
            print(f"\n--- Scenario: Total Budget = ${budget:,.2f} ---")

            result = self.optimize(
                total_budget=budget,
                current_spend=current_spend
            )

            scenario_data = {
                'Total_Budget': budget,
                'Predicted_Revenue': result['optimal_revenue'],
                'Expected_ROI': result['optimal_revenue'] / budget
            }

            # Add optimal spend by channel
            for campaign, spend in result['optimal_spend'].items():
                scenario_data[f'{campaign}_Spend'] = spend

            results.append(scenario_data)

        df_scenarios = pd.DataFrame(results)

        print(f"\n{'=' * 70}")
        print("SCENARIO SUMMARY")
        print(f"{'=' * 70}\n")
        print(df_scenarios[['Total_Budget', 'Predicted_Revenue', 'Expected_ROI']].to_string(index=False))
        print(f"\n{'=' * 70}\n")

        return df_scenarios


if __name__ == "__main__":
    """
    Test the optimizer with sample data.
    """
    print("=" * 80)
    print(" " * 25 + "TESTING BUDGET OPTIMIZER")
    print("=" * 80)


    # This would normally import your trained MMM
    # For testing, we'll create a mock

    class MockMMM:
        def __init__(self):
            self.model = type('obj', (object,), {
                'intercept_': 10000,
                'coef_': np.array([500, 300, 200, -100])
            })()
            self.feature_names = [
                'Campaign_A_adstock_log',
                'Campaign_B_adstock_log',
                'Campaign_C_adstock_log',
                'Campaign_D_adstock_log'
            ]
            self.campaign_features = self.feature_names


    mmm = MockMMM()

    # Initialize optimizer
    optimizer = BudgetOptimizer(mmm)

    # Current spend
    current_spend = {
        'Campaign_A': 30000,
        'Campaign_B': 25000,
        'Campaign_C': 20000,
        'Campaign_D': 25000
    }

    # Optimize with 10% budget increase
    total_budget = 110000  # 10% increase from current 100K

    result = optimizer.optimize(
        total_budget=total_budget,
        current_spend=current_spend
    )

    # Run scenario analysis
    scenarios = [90000, 100000, 110000, 120000, 150000]
    df_scenarios = optimizer.scenario_analysis(scenarios, current_spend)

    print("\n✅ Budget optimizer test completed successfully!")