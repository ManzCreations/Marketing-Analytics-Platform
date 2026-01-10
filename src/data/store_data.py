"""
Data storage module.

This module handles saving processed data to various formats.
"""

import pandas as pd
import sqlite3
from pathlib import Path


def store_data(df, output_dir='data'):
    """
    Store processed data to CSV and SQLite database.

    Args:
        df: Transformed DataFrame to store
        output_dir: Directory to save files (default: 'data')

    Returns:
        dict with file paths of saved data
    """

    print("=" * 50)
    print("STEP 5: STORING DATA")
    print("=" * 50)

    # Create output directory if it doesn't exist
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    saved_files = {}

    # ===========================
    # 1. SAVE TO CSV
    # ===========================
    print("\n1. Saving to CSV...")
    csv_file = output_path / 'facebook_ads_processed.csv'
    df.to_csv(csv_file, index=False)
    saved_files['csv'] = str(csv_file)
    print(f"    Saved to: {csv_file}")
    print(f"    Rows: {len(df)}, Columns: {len(df.columns)}")

    # ===========================
    # 2. SAVE TO SQLITE DATABASE
    # ===========================
    print("\n2. Saving to SQLite database...")
    db_file = output_path / 'marketing_data.db'

    # Connect to database (creates if doesn't exist)
    conn = sqlite3.connect(db_file)

    # Save to table
    table_name = 'facebook_ads'
    df.to_sql(table_name, conn, if_exists='replace', index=False)

    # Verify save
    cursor = conn.cursor()
    cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
    row_count = cursor.fetchone()[0]

    saved_files['database'] = str(db_file)
    saved_files['table_name'] = table_name

    print(f"    Saved to: {db_file}")
    print(f"    Table: {table_name}")
    print(f"    Rows: {row_count}")

    # Create an index on date for faster queries
    cursor.execute(f"CREATE INDEX IF NOT EXISTS idx_date ON {table_name}(date)")
    cursor.execute(f"CREATE INDEX IF NOT EXISTS idx_campaign ON {table_name}(campaign_name)")
    print(f"    Created indexes on date and campaign_name")

    conn.close()

    # ===========================
    # 3. SAVE SUMMARY STATISTICS
    # ===========================
    print("\n3. Saving summary statistics...")
    summary_file = output_path / 'data_summary.txt'

    with open(summary_file, 'w') as f:
        f.write("Facebook Ads Data Summary\n")
        f.write("=" * 50 + "\n\n")

        f.write(f"Total Rows: {len(df)}\n")
        f.write(f"Total Columns: {len(df.columns)}\n")
        f.write(f"Date Range: {df['date'].min()} to {df['date'].max()}\n")
        f.write(f"Campaigns: {df['campaign_name'].nunique()}\n\n")

        f.write("Campaign List:\n")
        for campaign in df['campaign_name'].unique():
            f.write(f"  - {campaign}\n")
        f.write("\n")

        f.write("Key Metrics Summary:\n")
        f.write(f"  Total Spend: ${df['spend'].sum():,.2f}\n")
        f.write(f"  Total Revenue: ${df['revenue'].sum():,.2f}\n")
        f.write(f"  Overall ROAS: {df['revenue'].sum() / df['spend'].sum():.2f}x\n")
        f.write(f"  Total Conversions: {df['conversions'].sum():,}\n")
        f.write(f"  Average Daily Spend: ${df['spend'].mean():,.2f}\n\n")

        f.write("Column List:\n")
        for col in df.columns:
            f.write(f"  - {col}\n")

    saved_files['summary'] = str(summary_file)
    print(f"    Saved to: {summary_file}")

    # ===========================
    # Summary
    # ===========================
    print("\n" + "=" * 50)
    print("  STORAGE COMPLETE")
    print(f"Files saved to: {output_path.absolute()}")
    print("=" * 50 + "\n")

    return saved_files


def load_data_from_csv(csv_path='data/facebook_ads_processed.csv'):
    """
    Load processed data from CSV.

    Args:
        csv_path: Path to CSV file

    Returns:
        DataFrame with processed data
    """
    df = pd.read_csv(csv_path, parse_dates=['date'])
    print(f"Loaded {len(df)} rows from {csv_path}")
    return df


def load_data_from_db(db_path='data/marketing_data.db', table_name='facebook_ads'):
    """
    Load processed data from SQLite database.

    Args:
        db_path: Path to database file
        table_name: Name of table to load

    Returns:
        DataFrame with processed data
    """
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn, parse_dates=['date'])
    conn.close()
    print(f"Loaded {len(df)} rows from {db_path} (table: {table_name})")
    return df


if __name__ == "__main__":
    # Test the full pipeline
    from fetch_data import generate_facebook_ads_sample
    from validate_data import validate_data
    from clean_data import clean_data
    from transform_data import transform_data

    print("Running COMPLETE data pipeline...\n")
    print("=" * 50)
    print("FACEBOOK ADS DATA PIPELINE")
    print("=" * 50)
    print()

    # Step 1: Fetch
    print("STEP 1: FETCHING DATA")
    print("=" * 50)
    df_raw = generate_facebook_ads_sample(n_days=90)
    print()

    # Step 2: Validate
    validation_results = validate_data(df_raw)

    # Step 3: Clean
    df_clean = clean_data(df_raw)

    # Step 4: Transform
    df_final = transform_data(df_clean)

    # Step 5: Store
    saved_files = store_data(df_final)

    print("\n" + "=" * 50)
    print("    PIPELINE COMPLETE    ")
    print("=" * 50)
    print("\nSaved files:")
    for key, path in saved_files.items():
        print(f"  {key}: {path}")

    # Test loading
    print("\n" + "=" * 50)
    print("Testing data loading...")
    print("=" * 50)

    df_from_csv = load_data_from_csv(saved_files['csv'])
    df_from_db = load_data_from_db(saved_files['database'], saved_files['table_name'])

    print(f"\n  Both loading methods work!")
    print(f"CSV shape: {df_from_csv.shape}")
    print(f"DB shape: {df_from_db.shape}")