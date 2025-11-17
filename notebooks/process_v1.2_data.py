"""
Austin Investment Analyzer v1.2 - Data Processing Script
Executes the v1.2 analysis pipeline and generates processed data for the Streamlit app.
"""

import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("AUSTIN INVESTMENT ANALYZER v1.2 - DATA PROCESSING")
print("="*80)
print(f"\nStarting analysis at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

# =============================================================================
# USER-ADJUSTABLE ASSUMPTIONS
# =============================================================================

print("📋 Loading investment assumptions...")

# --- FINANCING PARAMETERS ---
DOWN_PAYMENT_PCT = 0.20
INTEREST_RATE = 0.07
LOAN_TERM_YEARS = 30

# --- OPERATING EXPENSE RATES ---
PROPERTY_TAX_RATE = 0.022  # 2.2% - Austin actual
INSURANCE_RATE = 0.005
MAINTENANCE_RATE = 0.01

# --- STR-SPECIFIC EXPENSES ---
STR_UTILITIES_MONTHLY = 200
STR_CLEANING_PER_TURNOVER = 100
STR_SUPPLIES_MONTHLY = 75
STR_PLATFORM_FEE_PCT = 0.03
ESTIMATED_TURNOVERS_PER_MONTH = 3

# --- LTR-SPECIFIC EXPENSES ---
LTR_VACANCY_RATE = 0.08
STR_VACANCY_EQUIVALENT = 0.35

# --- APPRECIATION & EXIT ASSUMPTIONS ---
ANNUAL_RENT_INCREASE = 0.04
ANNUAL_APPRECIATION = 0.035
HOLD_PERIOD_YEARS = 5
SELLING_COSTS_PCT = 0.06
REFINANCE_COSTS = 5000
REFI_CASH_OUT_PCT = 0.80

# --- FILTERING CRITERIA ---
MIN_AIRBNB_LISTINGS = 7

print(f"✅ Assumptions loaded")
print(f"   - Property Tax: {PROPERTY_TAX_RATE*100:.1f}% (CORRECTED from 1.2%)")
print(f"   - Interest Rate: {INTEREST_RATE*100:.1f}%")
print(f"   - Down Payment: {DOWN_PAYMENT_PCT*100:.0f}%")

# =============================================================================
# LOAD DATA
# =============================================================================

print("\n📂 Loading raw data files...")

try:
    df_housing = pd.read_csv('data/raw/zillow_housing_v1.2_20251116.csv')
    print(f"✅ Loaded Zillow housing: {len(df_housing):,} rows")

    df_rent = pd.read_csv('data/raw/zillow_rent_v1.2_20251116.csv')
    print(f"✅ Loaded Zillow rent: {len(df_rent):,} rows")

    df_airbnb = pd.read_csv('data/raw/airbnb_austin_v1.2_20251116.csv.gz', compression='gzip')
    print(f"✅ Loaded Airbnb: {len(df_airbnb):,} rows")
except FileNotFoundError as e:
    print(f"❌ ERROR: Could not find data file: {e}")
    exit(1)

# =============================================================================
# FILTER AUSTIN DATA
# =============================================================================

print("\n🔍 Filtering for Austin neighborhoods...")

# Housing
df_austin_housing = df_housing[
    (df_housing['RegionType'] == 'neighborhood') &
    (df_housing['City'] == 'Austin')
].copy()
print(f"✅ Austin housing neighborhoods: {len(df_austin_housing)}")

# Rent
df_austin_rent = df_rent[
    df_rent['RegionType'] == 'neighborhood'
].copy()
print(f"✅ Austin rent neighborhoods: {len(df_austin_rent)}")

# Airbnb - clean data
df_airbnb_clean = df_airbnb[
    (df_airbnb['price'].notna()) &
    (df_airbnb['bedrooms'].notna()) &
    (df_airbnb['host_neighbourhood'].notna())
].copy()

# Extract price
df_airbnb_clean['price_clean'] = df_airbnb_clean['price'].str.replace('$', '').str.replace(',', '').astype(float)

# Filter hotels
df_airbnb_clean = df_airbnb_clean[df_airbnb_clean['price_clean'] <= 1000]
print(f"✅ Cleaned Airbnb listings: {len(df_airbnb_clean):,} (removed hotels > $1000/night)")

# =============================================================================
# CALCULATE APPRECIATION METRICS
# =============================================================================

print("\n📈 Calculating appreciation metrics...")

date_columns = [col for col in df_austin_housing.columns if col.startswith('20')]
latest_date = date_columns[-1]
df_austin_housing['current_price'] = df_austin_housing[latest_date]

print(f"   Current price date: {latest_date}")

# Baseline CAGR (Weighted 2000-2025)
# Uses all historical data with reduced weight for anomaly period (June 2020 - April 2023)
print("   Using weighted regression (2000-2025) with 0.3 weight for June 2020 - April 2023")

def calculate_weighted_cagr(row, date_cols):
    """
    Calculate CAGR using weighted linear regression in log space.
    Applies 0.3 weight to June 2020 - April 2023 anomaly period.
    """
    # Extract valid prices and dates
    prices = []
    years_since_2000 = []
    weights = []

    for col in date_cols:
        price = row[col]
        if pd.notna(price) and price > 0:
            date = datetime.strptime(col, '%Y-%m-%d')
            prices.append(price)
            years_since_2000.append((date - datetime(2000, 1, 1)).days / 365.25)

            # Apply reduced weight to anomaly period
            if datetime(2020, 6, 30) <= date <= datetime(2023, 4, 30):
                weights.append(0.3)
            else:
                weights.append(1.0)

    if len(prices) < 2:
        return np.nan

    # Convert to numpy arrays
    prices = np.array(prices)
    years = np.array(years_since_2000)
    weights = np.array(weights)
    log_prices = np.log(prices)

    # Weighted linear regression: log(price) = a + b*years
    W = np.diag(weights)
    X = np.column_stack([np.ones(len(years)), years])

    try:
        # Solve: (X^T W X)^-1 X^T W y
        coeffs = np.linalg.solve(X.T @ W @ X, X.T @ W @ log_prices)
        # CAGR = exp(b) - 1
        cagr = (np.exp(coeffs[1]) - 1) * 100
        return cagr
    except:
        return np.nan

df_austin_housing['baseline_cagr'] = df_austin_housing.apply(
    lambda row: calculate_weighted_cagr(row, date_columns), axis=1
)

# Recovery CAGR (2023-present)
recovery_start = '2023-01-31'
recovery_end = latest_date

recovery_start_date = datetime.strptime(recovery_start, '%Y-%m-%d')
recovery_end_date = datetime.strptime(recovery_end, '%Y-%m-%d')
recovery_years = (recovery_end_date - recovery_start_date).days / 365.25

df_austin_housing['recovery_cagr'] = (
    (df_austin_housing[recovery_end] / df_austin_housing[recovery_start]) ** (1/recovery_years) - 1
) * 100

# Peak analysis
peak_columns = [col for col in date_columns if '2022' in col]
df_austin_housing['peak_price_2022'] = df_austin_housing[peak_columns].max(axis=1)

df_austin_housing['distance_from_peak'] = (
    (df_austin_housing['current_price'] / df_austin_housing['peak_price_2022']) - 1
) * 100

print(f"✅ Appreciation metrics calculated")
print(f"   Median baseline CAGR (Weighted 2000-2025): {df_austin_housing['baseline_cagr'].median():.2f}%")
print(f"   Median distance from peak: {df_austin_housing['distance_from_peak'].median():.2f}%")

# =============================================================================
# PROCESS AIRBNB DATA
# =============================================================================

print("\n🏠 Processing Airbnb STR metrics...")

# Calculate occupancy
df_airbnb_clean['occupancy_rate'] = (
    df_airbnb_clean['estimated_occupancy_l365d'] / df_airbnb_clean['availability_365'] * 100
)
df_airbnb_clean['occupancy_rate'] = df_airbnb_clean['occupancy_rate'].fillna(0)
df_airbnb_clean.loc[df_airbnb_clean['occupancy_rate'] > 100, 'occupancy_rate'] = 100

# Calculate monthly income
df_airbnb_clean['monthly_str_income'] = (
    df_airbnb_clean['price_clean'] * 30.4 * (df_airbnb_clean['occupancy_rate'] / 100)
)

# Aggregate by neighborhood
airbnb_by_neighborhood = df_airbnb_clean.groupby('host_neighbourhood').agg({
    'price_clean': 'median',
    'monthly_str_income': 'median',
    'bedrooms': 'median',
    'occupancy_rate': 'median',
    'id': 'count'
}).reset_index()

airbnb_by_neighborhood.columns = [
    'neighborhood', 'median_nightly_rate', 'median_monthly_str_income',
    'median_bedrooms', 'occupancy_rate', 'listing_count'
]

# Ensure neighborhood is string type for merging
airbnb_by_neighborhood['neighborhood'] = airbnb_by_neighborhood['neighborhood'].astype(str)

# Filter by minimum listings
airbnb_by_neighborhood = airbnb_by_neighborhood[
    airbnb_by_neighborhood['listing_count'] >= MIN_AIRBNB_LISTINGS
]

print(f"✅ {len(airbnb_by_neighborhood)} neighborhoods with {MIN_AIRBNB_LISTINGS}+ listings")

# =============================================================================
# MERGE DATA
# =============================================================================

print("\n🔗 Merging datasets...")

df_housing_merge = df_austin_housing[[
    'RegionName', 'current_price', 'baseline_cagr', 'recovery_cagr',
    'peak_price_2022', 'distance_from_peak'
]].copy()
df_housing_merge.columns = [
    'neighborhood', 'current_price', 'baseline_cagr', 'recovery_cagr',
    'peak_price_2022', 'distance_from_peak'
]

# Ensure neighborhood is string type for merging
df_housing_merge['neighborhood'] = df_housing_merge['neighborhood'].astype(str)

df_merged = df_housing_merge.merge(
    airbnb_by_neighborhood,
    on='neighborhood',
    how='left'  # Keep all housing neighborhoods, add STR data where available
)

# Count neighborhoods with and without STR data
has_str = df_merged['listing_count'].notna().sum()
no_str = df_merged['listing_count'].isna().sum()

print(f"✅ Merged dataset: {len(df_merged)} total neighborhoods")
print(f"   - {has_str} with STR data (7+ Airbnb listings)")
print(f"   - {no_str} without STR data (LTR/Appreciation only)")

# Fill missing STR data for neighborhoods without Airbnb listings
df_merged['median_nightly_rate'] = df_merged['median_nightly_rate'].fillna(0)
df_merged['median_monthly_str_income'] = df_merged['median_monthly_str_income'].fillna(0)
df_merged['median_bedrooms'] = df_merged['median_bedrooms'].fillna(0)
df_merged['occupancy_rate'] = df_merged['occupancy_rate'].fillna(0)
df_merged['listing_count'] = df_merged['listing_count'].fillna(0).astype(int)

# =============================================================================
# CALCULATE OPERATING EXPENSES (CORRECTED)
# =============================================================================

print("\n💰 Calculating operating expenses (v1.2 CORRECTED)...")

# Mortgage
loan_amount = df_merged['current_price'] * (1 - DOWN_PAYMENT_PCT)
monthly_interest_rate = INTEREST_RATE / 12
num_payments = LOAN_TERM_YEARS * 12

df_merged['monthly_mortgage'] = (
    loan_amount *
    (monthly_interest_rate * (1 + monthly_interest_rate)**num_payments) /
    ((1 + monthly_interest_rate)**num_payments - 1)
)

# Operating expenses
df_merged['monthly_property_tax'] = df_merged['current_price'] * PROPERTY_TAX_RATE / 12
df_merged['monthly_insurance'] = df_merged['current_price'] * INSURANCE_RATE / 12
df_merged['monthly_maintenance'] = df_merged['current_price'] * MAINTENANCE_RATE / 12

df_merged['monthly_base_costs'] = (
    df_merged['monthly_mortgage'] +
    df_merged['monthly_property_tax'] +
    df_merged['monthly_insurance'] +
    df_merged['monthly_maintenance']
)

# STR-specific costs (NEW)
df_merged['str_utilities'] = STR_UTILITIES_MONTHLY
df_merged['str_cleaning'] = STR_CLEANING_PER_TURNOVER * ESTIMATED_TURNOVERS_PER_MONTH
df_merged['str_supplies'] = STR_SUPPLIES_MONTHLY
df_merged['str_platform_fees'] = df_merged['median_monthly_str_income'] * STR_PLATFORM_FEE_PCT

df_merged['str_additional_costs'] = (
    df_merged['str_utilities'] +
    df_merged['str_cleaning'] +
    df_merged['str_supplies'] +
    df_merged['str_platform_fees']
)

df_merged['str_total_costs'] = df_merged['monthly_base_costs'] + df_merged['str_additional_costs']

print(f"✅ Operating expenses calculated")
print(f"   Median base costs: ${df_merged['monthly_base_costs'].median():.0f}/month")
print(f"   Median STR additional: ${df_merged['str_additional_costs'].median():.0f}/month")
print(f"   Median STR total: ${df_merged['str_total_costs'].median():.0f}/month")

# =============================================================================
# STRATEGY #1: CASH FLOW
# =============================================================================

print("\n📊 Calculating STRATEGY #1: Cash Flow Analysis...")

# STR cash flow
df_merged['str_monthly_cash_flow'] = (
    df_merged['median_monthly_str_income'] - df_merged['str_total_costs']
)
df_merged['str_annual_cash_flow'] = df_merged['str_monthly_cash_flow'] * 12
df_merged['str_cash_on_cash_return'] = (
    df_merged['str_annual_cash_flow'] / (df_merged['current_price'] * DOWN_PAYMENT_PCT) * 100
)

# LTR estimates - use price-tier approach for realistic SFH rents
def calculate_ltr_rent_by_price_tier(price):
    """Price-tier based rent: higher-priced homes have lower rent-to-price ratios"""
    if price < 300000:
        return price * 0.0080  # 0.80% - Entry-level SFH (2-3BR) - validated Nov 2025
    elif price < 500000:
        return price * 0.0065  # 0.65% - Mid-market SFH (3-4BR)
    elif price < 700000:
        return price * 0.0060  # 0.60% - Upper-mid SFH (4BR)
    else:
        return price * 0.0065  # 0.65% - Luxury SFH (4-5BR)

df_merged['estimated_ltr_rent'] = df_merged['current_price'].apply(calculate_ltr_rent_by_price_tier)
df_merged['ltr_effective_rent'] = df_merged['estimated_ltr_rent'] * (1 - LTR_VACANCY_RATE)

print(f"\n💡 LTR Rent Calculation (Price-Tier Method):")
print(f"   <$300k: 0.80% | $300-500k: 0.65% | $500-700k: 0.60% | >$700k: 0.65%")
print(f"   Median ${df_merged['current_price'].median():.0f} home → ${df_merged['estimated_ltr_rent'].median():.0f}/month rent")
df_merged['ltr_monthly_cash_flow'] = df_merged['ltr_effective_rent'] - df_merged['monthly_base_costs']
df_merged['ltr_annual_cash_flow'] = df_merged['ltr_monthly_cash_flow'] * 12
df_merged['ltr_cash_on_cash_return'] = (
    df_merged['ltr_annual_cash_flow'] / (df_merged['current_price'] * DOWN_PAYMENT_PCT) * 100
)

positive_str = (df_merged['str_monthly_cash_flow'] > 0).sum()
positive_ltr = (df_merged['ltr_monthly_cash_flow'] > 0).sum()

print(f"✅ Cash flow calculated")
print(f"   STR positive cash flow: {positive_str}/{len(df_merged)} ({positive_str/len(df_merged)*100:.1f}%)")
print(f"   LTR positive cash flow: {positive_ltr}/{len(df_merged)} ({positive_ltr/len(df_merged)*100:.1f}%)")

# =============================================================================
# STRATEGY #2: TOTAL ROI
# =============================================================================

print(f"\n📊 Calculating STRATEGY #2: Total ROI ({HOLD_PERIOD_YEARS} years)...")

def calculate_total_roi(row, hold_years, rental_type='str'):
    down_payment = row['current_price'] * DOWN_PAYMENT_PCT
    closing_costs = row['current_price'] * 0.03
    total_invested = down_payment + closing_costs

    if rental_type == 'str':
        year_1_cf = row['str_annual_cash_flow']
    else:
        year_1_cf = row['ltr_annual_cash_flow']

    cumulative_cf = 0
    for year in range(1, hold_years + 1):
        year_cf = year_1_cf * ((1 + ANNUAL_RENT_INCREASE) ** (year - 1))
        cumulative_cf += year_cf

    future_value = row['current_price'] * ((1 + ANNUAL_APPRECIATION) ** hold_years)
    appreciation_gain = future_value - row['current_price']

    loan_amount = row['current_price'] * (1 - DOWN_PAYMENT_PCT)
    principal_paydown = loan_amount * 0.15 * (hold_years / 5)

    total_return = cumulative_cf + appreciation_gain + principal_paydown
    roi_pct = (total_return / total_invested) * 100

    return pd.Series({
        'total_invested': total_invested,
        'cumulative_cash_flow': cumulative_cf,
        'appreciation_gain': appreciation_gain,
        'principal_paydown': principal_paydown,
        'total_return': total_return,
        'roi_pct': roi_pct,
        'annualized_roi': ((1 + roi_pct/100) ** (1/hold_years) - 1) * 100
    })

# Calculate for STR and LTR
str_roi = df_merged.apply(lambda row: calculate_total_roi(row, HOLD_PERIOD_YEARS, 'str'), axis=1)
str_roi.columns = ['str_' + col for col in str_roi.columns]

ltr_roi = df_merged.apply(lambda row: calculate_total_roi(row, HOLD_PERIOD_YEARS, 'ltr'), axis=1)
ltr_roi.columns = ['ltr_' + col for col in ltr_roi.columns]

df_merged = pd.concat([df_merged, str_roi, ltr_roi], axis=1)

print(f"✅ Total ROI calculated")
print(f"   Median STR ROI: {df_merged['str_roi_pct'].median():.1f}%")
print(f"   Median LTR ROI: {df_merged['ltr_roi_pct'].median():.1f}%")

# =============================================================================
# SAVE PROCESSED DATA
# =============================================================================

print("\n💾 Saving processed data...")

output_file = 'data/processed/neighborhoods_v1.2_complete.csv'
df_merged.to_csv(output_file, index=False)

print(f"✅ Saved to {output_file}")
print(f"   Columns: {len(df_merged.columns)}")
print(f"   Rows: {len(df_merged)}")

# =============================================================================
# SUMMARY REPORT
# =============================================================================

print("\n" + "="*80)
print("SUMMARY REPORT - v1.2 DATA PROCESSING COMPLETE")
print("="*80)

median_price = df_merged['current_price'].median()

print(f"\n📊 DATASET SUMMARY:")
print(f"   Total neighborhoods: {len(df_merged)}")
print(f"   Median property price: ${median_price:,.0f}")
print(f"   Data through: {latest_date}")

print(f"\n🔧 CRITICAL CORRECTIONS vs v1.0:")
old_tax = median_price * 0.012 / 12
new_tax = median_price * PROPERTY_TAX_RATE / 12
print(f"   Property Tax: ${old_tax:.0f}/mo → ${new_tax:.0f}/mo (+${new_tax - old_tax:.0f})")
print(f"   STR Expenses: $0/mo → ${df_merged['str_additional_costs'].median():.0f}/mo")
print(f"   TOTAL IMPACT: ~${(new_tax - old_tax) + df_merged['str_additional_costs'].median():.0f}/mo more in costs")

print(f"\n📈 INVESTMENT OUTCOMES (STR, {HOLD_PERIOD_YEARS} years):")
print(f"   Median monthly cash flow: ${df_merged['str_monthly_cash_flow'].median():.0f}")
print(f"   Positive cash flow: {positive_str}/{len(df_merged)} neighborhoods ({positive_str/len(df_merged)*100:.1f}%)")
print(f"   Median Cash-on-Cash return: {df_merged['str_cash_on_cash_return'].median():.2f}%")
print(f"   Median Total ROI: {df_merged['str_roi_pct'].median():.1f}%")

print("\n✅ DATA PROCESSING COMPLETE - Ready for Streamlit app!")
print("="*80)
print(f"\nCompleted at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
