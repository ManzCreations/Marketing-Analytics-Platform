"""
Master Data Pipeline Script

This script runs the complete data pipeline:
1. Fetch/Generate data
2. Validate data quality
3. Clean data
4. Transform and create features
5. Store processed data

Usage:
    python run_pipeline.py
"""

import sys
from pathlib import Path

from src.data.fetch_data import generate_realistic_facebook_ads_data
from src.data.validate_data import validate_data
from src.data.clean_data import clean_data
from src.data.transform_data import transform_data
from src.data.store_data import store_data


def run_complete_pipeline(n_days=90, output_dir='data'):
    """
    Run the complete data pipeline.

    Args:
        n_days: Number of days of data to generate
        output_dir: Directory to save output files

    Returns:
        Final processed DataFrame and file paths
    """

    print("\n")
    print("=" * 60)
    print(" " * 15 + "FACEBOOK ADS DATA PIPELINE")
    print("=" * 60)
    print()

    try:
        # ===========================
        # STEP 1: FETCH/GENERATE DATA
        # ===========================
        print("STEP 1: FETCHING DATA")
        print("=" * 60)
        print(f"Generating {n_days} days of sample Facebook Ads data...\n")

        df_raw = generate_realistic_facebook_ads_data(n_days=n_days)
        print()

        # ===========================
        # STEP 2: VALIDATE DATA
        # ===========================
        validation_results = validate_data(df_raw)

        if not validation_results['valid']:
            print("\n❌ Validation failed! Critical issues found:")
            for issue in validation_results['issues']:
                if issue.startswith('ERROR'):
                    print(f"  {issue}")
            print("\nPipeline stopped.")
            return None, None

        # ===========================
        # STEP 3: CLEAN DATA
        # ===========================
        df_clean = clean_data(df_raw)

        # ===========================
        # STEP 4: TRANSFORM DATA
        # ===========================
        df_final = transform_data(df_clean)

        # ===========================
        # STEP 5: STORE DATA
        # ===========================
        saved_files = store_data(df_final, output_dir=output_dir)

        # ===========================
        # SUMMARY
        # ===========================
        print("\n")
        print("=" * 60)
        print("✓✓✓ PIPELINE COMPLETED SUCCESSFULLY ✓✓✓")
        print("=" * 60)
        print()
        print(f"Processed {len(df_final)} rows with {len(df_final.columns)} columns")
        print(f"Date range: {df_final['date'].min()} to {df_final['date'].max()}")
        print(f"Campaigns: {df_final['campaign_name'].nunique()}")
        print()
        print("Saved files:")
        for key, path in saved_files.items():
            if key in ['csv', 'database', 'summary']:
                print(f"  📄 {key}: {path}")
        print()
        print("Key metrics:")
        print(f"  💰 Total Spend: ${df_final['spend'].sum():,.2f}")
        print(f"  💵 Total Revenue: ${df_final['revenue'].sum():,.2f}")
        print(f"  📈 Overall ROAS: {df_final['revenue'].sum() / df_final['spend'].sum():.2f}x")
        print(f"  🎯 Total Conversions: {df_final['conversions'].sum():,}")
        print()
        print("=" * 60)
        print()

        return df_final, saved_files

    except Exception as e:
        print(f"\n❌ Error in pipeline: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None


if __name__ == "__main__":
    # Run the pipeline
    df, files = run_complete_pipeline(n_days=90, output_dir='data')

    if df is not None:
        print("✅ Pipeline completed successfully!")
        print("\nYou can now use this data for:")
        print("  - Marketing Mix Modeling")
        print("  - Customer Lifetime Value analysis")
        print("  - Budget optimization")
        print("  - Dashboard visualization")
        print("\nNext step: Build the MMM model!")