"""
Master Data Pipeline Script

This script runs the complete data pipeline:
1. Fetch/Generate data
2. Validate data quality
3. Clean data
4. Transform and create features
5. Store processed data

"""

from src.data.fetch_data import generate_realistic_facebook_ads_data
from src.data.validate_data import validate_data
from src.data.clean_data import clean_data
from src.data.transform_data import transform_data
from src.data.store_data import store_data, store_mmm_results
from src.models.feature_engineering import MMMFeatureEngineer
from src.models.mmm_model import MarketingMixModel
from src.models.optimizer import BudgetOptimizer


def run_complete_pipeline(n_days=90, output_dir='data'):
    """
    Run the complete data pipeline with MMM.

    Args:
        n_days: Number of days of data to generate
        output_dir: Directory to save output files

    Returns:
        Final processed DataFrame and file paths
    """

    print("\n")
    print("=" * 60)
    print(" " * 15 + "FACEBOOK ADS DATA PIPELINE + MMM")
    print("=" * 60)
    print()

    try:
        # ===========================
        # STEP 1: FETCH/GENERATE DATA
        # ===========================
        print("STEP 1: FETCHING DATA")
        print("=" * 60)
        print(f"Generating {n_days} days of sample Facebook Ads data...\n")

        df_raw = generate_realistic_facebook_ads_data(n_days=n_days, seed=42)
        print()

        # ===========================
        # STEP 2: VALIDATE DATA
        # ===========================
        validation_results = validate_data(df_raw)

        if not validation_results['valid']:
            print("\n️  Validation found issues (will be fixed in cleaning)")
            print()

        # ===========================
        # STEP 3: CLEAN DATA
        # ===========================
        df_clean = clean_data(df_raw)

        # Extra cleaning for MMM
        df_clean = df_clean[df_clean['spend'] >= 0].copy()
        df_clean['revenue'] = df_clean['revenue'].fillna(0)

        # ===========================
        # STEP 4: TRANSFORM DATA
        # ===========================
        df_final = transform_data(df_clean)

        # ===========================
        # STEP 5: STORE DATA
        # ===========================
        saved_files = store_data(df_final, output_dir=output_dir)

        # ===========================
        # STEPS 6-10: MMM ANALYSIS
        # ===========================
        print("\n")
        print("=" * 60)
        print("RUNNING MMM ANALYSIS")
        print("=" * 60)
        print()

        # Step 6: MMM Feature Engineering
        print("Step 6: MMM Feature Engineering")
        print("-" * 60)

        adstock_params = {
            'Brand_Awareness_Q1': 0.7,
            'Product_Launch_Spring': 0.5,
            'Retargeting_Warm_Audience': 0.3,
            'Lead_Generation_B2B': 0.4
        }

        engineer = MMMFeatureEngineer(
            adstock_params=adstock_params,
            apply_saturation=True,
            include_time_features=True
        )

        df_mmm_features = engineer.fit_transform(df_clean, target_col='revenue')
        X_train, X_test, y_train, y_test = engineer.train_test_split(df_mmm_features, test_size=0.2)

        # Get campaign feature names (those with adstock_log)
        campaign_features = [col for col in engineer.feature_columns_ if 'adstock_log' in col]

        print(" Features ready\n")

        # Step 7: Train MMM Model
        print("Step 7: Train MMM Model")
        print("-" * 60)

        mmm = MarketingMixModel(model_type='ridge', scale_features=True)
        mmm.fit(X_train, y_train, campaign_features=campaign_features,
                tune_hyperparameters=True, cv_folds=5)

        print(" Model trained\n")

        # Step 8: Evaluate
        print("Step 8: Evaluate Model")
        print("-" * 60)

        metrics = mmm.evaluate(X_test, y_test)
        print(" Evaluation complete\n")

        # Step 9: Channel Attribution
        print("Step 9: Channel Attribution")
        print("-" * 60)

        df_spend = df_clean.pivot_table(index='date', columns='campaign_name',
                                        values='spend', aggfunc='sum', fill_value=0)
        test_start = df_mmm_features.iloc[len(X_train)]['date']
        test_end = df_mmm_features.iloc[-1]['date']
        df_spend_test = df_spend[(df_spend.index >= test_start) & (df_spend.index <= test_end)]

        contributions = mmm.get_channel_contributions(X_test, original_spend_data=df_spend_test)
        print(" Attribution complete\n")

        # Step 10: Budget Optimization
        print("Step 10: Budget Optimization")
        print("-" * 60)

        current_spend = df_spend_test.sum().to_dict()
        current_total = sum(current_spend.values())

        optimizer = BudgetOptimizer(mmm, engineer)
        optimization_result = optimizer.optimize(total_budget=current_total * 1.1,
                                                 current_spend=current_spend)

        scenarios = optimizer.scenario_analysis([current_total * 0.9, current_total * 1.0,
                                                 current_total * 1.1, current_total * 1.2,
                                                 current_total * 1.5], current_spend)

        print(" Optimization complete\n")

        # Store MMM Results
        store_mmm_results(mmm, engineer, metrics, contributions, optimization_result,
                          scenarios, X_test, y_test, df_mmm_features, output_dir)

        # ===========================
        # SUMMARY
        # ===========================
        print("\n")
        print("=" * 60)
        print("*** PIPELINE COMPLETED SUCCESSFULLY ***")
        print("=" * 60)
        print()
        print(f"Processed {len(df_final)} rows with {len(df_final.columns)} columns")
        print(f"Date range: {df_final['date'].min()} to {df_final['date'].max()}")
        print(f"Campaigns: {df_final['campaign_name'].nunique()}")
        print()
        print("Saved files:")
        for key, path in saved_files.items():
            if key in ['csv', 'database', 'summary']:
                print(f"   {key}: {path}")
        print()
        print("Key metrics:")
        print(f"  Total Spend: ${df_final['spend'].sum():,.2f}")
        print(f"  Total Revenue: ${df_final['revenue'].sum():,.2f}")
        print(f"  Overall ROAS: {df_final['revenue'].sum() / df_final['spend'].sum():.2f}x")
        print(f"  Total Conversions: {df_final['conversions'].sum():,}")
        print()
        print("MMM Results:")
        print(f"  Model R²: {metrics['r2']:.3f}")
        print(
            f"  Expected Revenue Lift: +{optimization_result.get('comparison', {}).get('revenue_increase_pct', 0):.1f}%")
        print()
        print("=" * 60)
        print()

        return df_final, saved_files

    except Exception as e:
        print(f"\n Error in pipeline: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None


if __name__ == "__main__":
    # Run the pipeline
    df, files = run_complete_pipeline(n_days=90, output_dir='data')

    if df is not None:
        print(" Pipeline completed successfully!")
        print("\nDatabase tables created:")
        print("  - facebook_ads (original data)")
        print("  - mmm_channel_contributions")
        print("  - mmm_budget_optimization")
        print("  - mmm_budget_scenarios")
        print("  - mmm_model_performance")
        print("  - mmm_model_coefficients")
        print("  - mmm_predictions")
        print("\nReady for Power BI!")