# Marketing Mix Model (MMM) Pipeline

A complete end-to-end Marketing Mix Modeling system that attributes revenue to marketing channels, calculates ROI, and optimizes budget allocation using Ridge regression and constrained optimization.

---

## Table of Contents

1. [Overview](#overview)
2. [What This Tool Does](#what-this-tool-does)
3. [Project Structure](#project-structure)
4. [Requirements](#requirements)
5. [Installation](#installation)
6. [Quick Start](#quick-start)
7. [How It Works](#how-it-works)
8. [Key Assumptions](#key-assumptions)
9. [Output Files](#output-files)
10. [Customization](#customization)
11. [Troubleshooting](#troubleshooting)

---

## Overview

This project implements a **Marketing Mix Model (MMM)** - a statistical technique used to measure the impact of marketing campaigns on revenue. It answers critical business questions:

- Which marketing channels actually drive revenue?
- What's the ROI of each campaign?
- How should we reallocate our marketing budget to maximize returns?

**Key Features:**
- ✅ Automated data generation (or use your own data)
- ✅ Feature engineering with adstock & saturation transformations
- ✅ Ridge regression model with hyperparameter tuning
- ✅ Channel attribution and ROI calculation
- ✅ Budget optimization with scenario analysis
- ✅ Exports everything to SQLite database for Power BI

---

## What This Tool Does

### Input:
Daily marketing data with:
- Date
- Campaign name
- Spend
- Impressions, clicks, conversions
- Revenue

### Process:
1. **Cleans data** - Handles missing values, outliers, negatives
2. **Engineers features** - Applies adstock (carryover effects) and saturation (diminishing returns)
3. **Trains model** - Ridge regression with time-based cross-validation
4. **Evaluates performance** - Calculates R², MAE, MAPE
5. **Attributes revenue** - Determines each channel's contribution
6. **Optimizes budget** - Finds allocation that maximizes revenue

### Output:
- SQLite database with 7 tables
- CSV export of processed data
- Summary report with key findings

---

## Project Structure

```
Marketing-Analytics-Platform/
│
├── run_pipeline.py                    # Main script - run this!
│
├── src/
|   ├── __init__.py                    # Makes src a Python package
|
|   └── data/
│       ├── fetch_data.py              # Original simple data generation
│       ├── validate_data.py           # Data validation (required columns, types, logic)
│       ├── clean_data.py              # Data cleaning (missing values, outliers, negatives)
│       ├── transform_data.py          # Basic transformations (CTR, CVR, ROAS, features)
│       └── store_data_enhanced.py     # Database storage with MMM results
|
│   └── models/
│       ├── feature_engineering.py     # Adstock, saturation, time features
│       ├── mmm_model.py               # Ridge regression model
│       └── optimizer.py               # Budget optimization
│
├── README.md                          # This file
└── requirements.txt                   # Required dependencies

```

---

## Requirements

### Python Version:
- Python 3.8 or higher

### Dependencies:
```
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
scipy>=1.7.0
```

---

## Installation

### Step 1: Clone the Repository
```bash
git clone https://github.com/yourusername/marketing-mix-model.git
cd marketing-mix-model
```

### Step 2: Install Dependencies
```bash
pip install pandas numpy scikit-learn scipy
```

Or with requirements file:
```bash
pip install -r requirements.txt
```

### Step 3: Verify Installation
```bash
python -c "import pandas, numpy, sklearn, scipy; print('All dependencies installed!')"
```

---

## Quick Start

### Run the Complete Pipeline:
```bash
python run_pipeline.py
```

**That's it!** The script will:
1. Generate 90 days of realistic marketing data
2. Clean and validate it
3. Train the MMM model
4. Optimize budget allocation
5. Save everything to `data/marketing_data.db`

**Time:** ~2-3 minutes to complete

### Check the Output:
```bash
ls data/
# Should see:
# - marketing_data.db
# - facebook_ads_processed.csv
# - data_summary.txt
```

---

## How It Works

### Step 1: Data Generation (or Load Your Data)

**Default:** Generates synthetic Facebook Ads data with realistic patterns:
- 4 campaigns with different performance profiles
- Seasonality (weekends, holidays, month-end)
- Adstock effects (carryover from previous days)
- Diminishing returns (saturation)
- Data quality issues (missing values, outliers)

**Using Your Own Data:**
Replace the data generation in `run_pipeline.py` line 45 with:
```python
df_raw = pd.read_csv('your_data.csv')
```

**Required columns:**
- `date` (datetime)
- `campaign_name` (string)
- `spend` (float)
- `impressions` (int)
- `clicks` (int)
- `conversions` (int)
- `revenue` (float)

---

### Step 2: Feature Engineering

**Three key transformations:**

#### A. Adstock (Carryover Effects)
```
Formula: Adstock_t = Spend_t + (decay_rate × Adstock_t-1)
```
- Models how yesterday's advertising still affects today's revenue
- Campaign-specific decay rates:
  - Brand Awareness: 0.7 (long carryover)
  - Product Launch: 0.5 (medium)
  - Retargeting: 0.3 (short - direct response)
  - Lead Gen: 0.4 (medium)

**Where:** `src/models/feature_engineering.py` → `apply_adstock()`

#### B. Saturation (Diminishing Returns)
```
Formula: Saturated_Spend = log(Spend + 1)
```
- Models how first $100 is more effective than 10,000th $100
- Captures the concave revenue relationship

**Where:** `src/models/feature_engineering.py` → `apply_saturation()`

#### C. Time Features
- Day of week (Monday-Sunday dummies)
- Weekend indicator
- Month dummies (January-December)
- Holiday indicator (Black Friday, Christmas)
- Linear time trend

**Where:** `src/models/feature_engineering.py` → `create_time_features()`

---

### Step 3: Model Training

**Algorithm:** Ridge Regression

**Why Ridge?**
- Handles multicollinearity (campaigns run simultaneously)
- Interpretable coefficients (can extract contributions)
- Stable predictions
- Industry standard for MMM

**Hyperparameter Tuning:**
- Method: GridSearchCV
- Cross-validation: TimeSeriesSplit (5 folds)
- Parameter: alpha (regularization strength)
- Range: [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]

**Train/Test Split:**
- Training: First 80% of days (chronologically)
- Test: Last 20% of days
- **CRITICAL:** Time-based split prevents data leakage

**Where:** `src/models/mmm_model.py` → `MarketingMixModel`

---

### Step 4: Model Evaluation

**Metrics:**
- **R²:** Proportion of variance explained (0.6-0.7 is typical for marketing data)
- **MAE:** Mean Absolute Error in dollars
- **RMSE:** Root Mean Squared Error
- **MAPE:** Mean Absolute Percentage Error

**Where:** `src/models/mmm_model.py` → `evaluate()`

---

### Step 5: Channel Attribution

**Method:** Direct coefficient interpretation
```
Channel Contribution = Model Coefficient × Feature Values
```

**Outputs:**
- Total revenue contribution per channel
- Percentage of total revenue
- ROI/ROAS (Revenue / Spend)

**Where:** `src/models/mmm_model.py` → `get_channel_contributions()`

---

### Step 6: Budget Optimization

**Method:** Constrained optimization (scipy.optimize, SLSQP algorithm)

**Objective:** Maximize predicted revenue

**Constraints:**
- Total spend = Budget
- Minimum spend per channel: 5% of budget
- Maximum spend per channel: 60% of budget

**Scenario Analysis:**
Tests multiple budget levels:
- 90% of current
- 100% of current (baseline)
- 110% of current
- 120% of current
- 150% of current

**Where:** `src/models/optimizer.py` → `BudgetOptimizer`

---

## Key Assumptions

### 1. **Linear Response (After Transformations)**
The model assumes that after applying adstock and saturation transformations, the relationship between spend and revenue is approximately linear. This is standard in MMM.

### 2. **Stable Market Conditions**
The model assumes that competitive dynamics, market size, and consumer behavior remain relatively stable during the analysis period.

### 3. **No External Shocks**
Major events (pandemics, economic crashes, etc.) that fundamentally change consumer behavior are not explicitly modeled.

### 4. **Independence of Channels**
The model treats each marketing channel as independent. Synergy effects (e.g., TV + Social working together) are not captured unless explicitly engineered.

### 5. **Adstock Decay Rates**
Decay rates are set based on industry benchmarks:
- Brand campaigns: 0.6-0.8 (awareness builds over time)
- Performance campaigns: 0.2-0.4 (immediate response)

You can adjust these in `run_pipeline.py` lines 85-90.

### 6. **Test Period Representativeness**
The model's recommendations assume the test period (last 20% of data) is representative of future performance.

### 7. **Causality vs Correlation**
While MMM is designed to estimate causal effects, it cannot completely rule out confounding variables. Use with business judgment.

---

## Output Files

### 1. SQLite Database: `data/marketing_data.db`

**7 tables created:**

#### `facebook_ads` (353 rows)
Daily campaign performance with all metrics and calculated features (CTR, CVR, ROAS, etc.)

**Key Columns:**
- date, campaign_name, spend, revenue, conversions
- ctr, cvr, cpa, roas
- day_of_week, is_weekend, is_holiday
- campaign_type, time_period

#### `mmm_channel_contributions` (4 rows)
Attribution results - how much revenue each channel contributed

**Key Columns:**
- Campaign, Total_Contribution, Percentage
- Total_Spend, ROI, ROAS

#### `mmm_budget_optimization` (4 rows)
Recommended budget allocation

**Key Columns:**
- Campaign, Optimal_Spend, Optimal_Pct
- Expected_Revenue, Revenue_Increase_Pct

#### `mmm_budget_scenarios` (5 rows)
What-if analysis at different budget levels

**Key Columns:**
- Total_Budget, Predicted_Revenue, Expected_ROI
- Individual campaign spend columns

#### `mmm_model_performance` (1 row)
Model evaluation metrics

**Key Columns:**
- r2, mae, rmse, mape
- n_samples, model_type, regularization_alpha

#### `mmm_model_coefficients` (16 rows)
Feature importance - which variables drive revenue

**Key Columns:**
- Feature, Coefficient, Feature_Type

#### `mmm_predictions` (18 rows)
Actual vs predicted revenue on test set

**Key Columns:**
- date, actual_revenue, predicted_revenue, error

---

### 2. CSV Export: `data/facebook_ads_processed.csv`

Same as `facebook_ads` table but in CSV format for easier access.

---

### 3. Summary Report: `data/data_summary.txt`

Text file with:
- Data overview (rows, columns, date range)
- Campaign list
- Key metrics summary
- Column list

---

## Customization

### Change Number of Days:
```python
# In run_pipeline.py, line 195
df, files = run_complete_pipeline(n_days=180, output_dir='data')
```

### Use Your Own Data:
```python
# In run_pipeline.py, replace line 45
# OLD:
df_raw = generate_realistic_facebook_ads_data(n_days=n_days, seed=42)

# NEW:
df_raw = pd.read_csv('your_marketing_data.csv')
df_raw['date'] = pd.to_datetime(df_raw['date'])
```

### Adjust Adstock Decay Rates:
```python
# In run_pipeline.py, lines 85-90
adstock_params = {
    'Your_Campaign_1': 0.6,  # Change these values
    'Your_Campaign_2': 0.4,  # 0.0 = no carryover, 1.0 = perfect carryover
    'Your_Campaign_3': 0.3,
    'Your_Campaign_4': 0.5
}
```

### Change Train/Test Split:
```python
# In run_pipeline.py, line 96
X_train, X_test, y_train, y_test = engineer.train_test_split(
    df_mmm_features, 
    test_size=0.3  # Change from 0.2 to 0.3 for 30% test
)
```

### Adjust Budget Constraints:
```python
# In src/models/optimizer.py, line 142-143
optimization_result = optimizer.optimize(
    total_budget=current_total * 1.1,
    min_spend_pct=0.10,  # Change from 0.05 to 0.10 = 10% minimum
    max_spend_pct=0.50   # Change from 0.60 to 0.50 = 50% maximum
)
```

### Change Model Type:
```python
# In run_pipeline.py, line 103
mmm = MarketingMixModel(
    model_type='lasso',  # Options: 'ridge', 'lasso', 'elasticnet'
    scale_features=True
)
```

---

## Troubleshooting

### Issue: "No module named 'src'"
**Solution:** Make sure you're running from the project root directory:
```bash
cd /path/to/marketing-mix-model
python run_pipeline.py
```

### Issue: "No module named 'pandas'" (or other dependency)
**Solution:** Install dependencies:
```bash
pip install pandas numpy scikit-learn scipy
```

### Issue: "Validation found issues"
**Expected behavior.** The synthetic data includes intentional quality issues to test cleaning. The pipeline will fix them and continue.

### Issue: Database file already exists
**Solution:** The pipeline overwrites existing tables. If you want to keep old results:
```python
# In run_pipeline.py, line 195
df, files = run_complete_pipeline(n_days=90, output_dir='data_backup')
```

### Issue: Model R² is very low (< 0.4)
**Possible causes:**
1. Not enough data (try increasing n_days to 180+)
2. Campaigns have very inconsistent performance
3. External factors dominating (seasonality, events)

**Solution:** Check data quality and consider adding more control variables.

### Issue: Negative ROI for some channels
**This can be legitimate!** Possible reasons:
1. Test period anomalies (holidays, special events)
2. Channel has indirect effects not captured (brand halo)
3. Campaign genuinely underperforming
4. Attribution lag (brand campaigns take time)

**Recommendation:** Investigate with business context before cutting budget.

### Issue: Optimization doesn't converge
**Solution:** The optimizer includes a warning but still returns results. Usually close enough. If needed, adjust bounds in `optimizer.py`.

---

## Technical Details

### Why Time-Based Train/Test Split?

Marketing data is time series. Random splits leak future information into training:
- ❌ Random split: Model sees future data during training
- ✅ Time-based split: Model only sees past, predicts future

This tests true predictive ability.

### Why Ridge Regression?

| Algorithm | Pros | Cons | Decision |
|-----------|------|------|----------|
| Linear Regression | Simple | Multicollinearity issues | ❌ |
| Ridge | Handles correlation, interpretable | No feature selection | ✅ |
| Lasso | Feature selection | May drop important channels | ⚠️ |
| XGBoost | High accuracy | Black box, can't extract contributions | ❌ |

### Why Log Saturation?

Alternative: Hill saturation curves (S-curves)
- **Hill:** More flexible, requires parameter optimization
- **Log:** Simpler, robust, industry standard

Log is sufficient for most use cases.

---

## Performance Expectations

**Typical Results:**
- R²: 0.55-0.75 (marketing data has lots of noise)
- Budget optimization lift: 3-10% with same total spend
- Runtime: 2-3 minutes for 90 days, 4 campaigns

**Good Performance:**
- R² > 0.6
- MAE < 15% of average daily revenue
- Optimization recommendations align with business intuition

---

## Next Steps After Running Pipeline

1. **Open the database** in SQLite Browser or DBeaver to explore tables
2. **Import to Power BI** (or Tableau) for visualization
3. **Review attribution results** - do they match business expectations?
4. **Test recommendations** - run A/B tests with optimized allocation
5. **Retrain monthly** - as new data comes in

---

## Contributing

Contributions welcome! Areas for improvement:
- Bayesian MMM implementation
- More sophisticated saturation curves
- Hierarchical models (grouping campaigns)
- Automated hyperparameter tuning for adstock decay rates
- Multi-touch attribution integration

---

## License

MIT License - feel free to use for commercial or personal projects.

---

## Contact & Support

Questions? Open an issue on GitHub or contact [your contact info].

---

## Acknowledgments

Built for marketing data science interviews and real-world MMM applications.

Inspired by:
- Google's Meridian MMM
- Meta's Robyn package
- PyMC-Marketing

---

**Ready to optimize your marketing budget?** Run `python run_pipeline.py` and let the data guide your decisions! 🚀
