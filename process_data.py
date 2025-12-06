"""
Real Estate Investment Analyzer v2.0 - Data Processing Script
Processes Zillow and Airbnb data for multiple metro areas.
"""

import argparse
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import sys
import warnings
warnings.filterwarnings('ignore')

# Add project root to path for config imports
sys.path.insert(0, str(Path(__file__).parent))

from config.metro_config import get_config_loader, MetroConfig

# =============================================================================
# CONSTANTS (Shared across all metros)
# =============================================================================

# --- FINANCING PARAMETERS ---
DOWN_PAYMENT_PCT = 0.20
INTEREST_RATE = 0.07
LOAN_TERM_YEARS = 30

# --- STR-SPECIFIC EXPENSES ---
STR_UTILITIES_MONTHLY = 200
STR_CLEANING_PER_TURNOVER = 100
STR_SUPPLIES_MONTHLY = 75
STR_PLATFORM_FEE_PCT = 0.03
ESTIMATED_TURNOVERS_PER_MONTH = 3

# --- LTR-SPECIFIC EXPENSES ---
LTR_VACANCY_RATE = 0.08
MAINTENANCE_RATE = 0.01

# --- APPRECIATION & EXIT ASSUMPTIONS ---
ANNUAL_RENT_INCREASE = 0.04
ANNUAL_APPRECIATION = 0.035
HOLD_PERIOD_YEARS = 5

# --- FILTERING CRITERIA ---
MIN_AIRBNB_LISTINGS = 7


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def calculate_weighted_cagr(row, date_cols):
    """
    Calculate CAGR using weighted linear regression in log space.
    Applies 0.3 weight to June 2020 - April 2023 anomaly period.
    """
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

    prices = np.array(prices)
    years = np.array(years_since_2000)
    weights = np.array(weights)
    log_prices = np.log(prices)

    W = np.diag(weights)
    X = np.column_stack([np.ones(len(years)), years])

    try:
        coeffs = np.linalg.solve(X.T @ W @ X, X.T @ W @ log_prices)
        cagr = (np.exp(coeffs[1]) - 1) * 100
        return cagr
    except:
        return np.nan


def clean_airbnb_data(df_airbnb):
    """
    Clean and filter Airbnb data with enhanced outlier handling.

    Steps:
    1. Filter by room type (exclude hotels)
    2. Statistical outlier removal per neighborhood
    3. Hard cap as final safety net
    """
    # Basic cleaning
    df_clean = df_airbnb[
        (df_airbnb['price'].notna()) &
        (df_airbnb['bedrooms'].notna()) &
        (df_airbnb['host_neighbourhood'].notna())
    ].copy()

    # Extract price
    df_clean['price_clean'] = df_clean['price'].str.replace(r'[\$,]', '', regex=True).astype(float)

    # Step 1: Filter by room type (exclude hotels)
    if 'room_type' in df_clean.columns:
        before_hotel = len(df_clean)
        df_clean = df_clean[df_clean['room_type'] != 'Hotel room']
        print(f"      Removed {before_hotel - len(df_clean)} hotel rooms")

    # Step 2: Statistical outlier removal per neighborhood (3 std from median)
    def remove_price_outliers(df, n_std=3):
        """Remove listings > n standard deviations from neighborhood median."""
        median = df.groupby('host_neighbourhood')['price_clean'].transform('median')
        std = df.groupby('host_neighbourhood')['price_clean'].transform('std')
        std = std.fillna(0)  # Handle neighborhoods with single listing
        upper_bound = median + (n_std * std)
        # Don't filter if std is 0 (single listing neighborhoods)
        mask = (std == 0) | (df['price_clean'] <= upper_bound)
        return df[mask]

    before_outlier = len(df_clean)
    df_clean = remove_price_outliers(df_clean, n_std=3)
    print(f"      Removed {before_outlier - len(df_clean)} statistical outliers (>3 std)")

    # Step 3: Hard cap as final safety net
    before_cap = len(df_clean)
    df_clean = df_clean[df_clean['price_clean'] <= 1500]
    print(f"      Removed {before_cap - len(df_clean)} listings > $1500/night")

    return df_clean


def process_airbnb_for_metro(df_airbnb_clean):
    """Process cleaned Airbnb data into neighborhood aggregates."""
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

    airbnb_by_neighborhood['neighborhood'] = airbnb_by_neighborhood['neighborhood'].astype(str)

    # Filter by minimum listings
    airbnb_by_neighborhood = airbnb_by_neighborhood[
        airbnb_by_neighborhood['listing_count'] >= MIN_AIRBNB_LISTINGS
    ]

    return airbnb_by_neighborhood


def calculate_total_roi(row, hold_years, rental_type, interest_rate, down_payment_pct, loan_term_years):
    """Calculate total ROI for a neighborhood."""
    down_payment = row['current_price'] * down_payment_pct
    closing_costs = row['current_price'] * 0.03
    total_invested = down_payment + closing_costs

    if rental_type == 'str':
        year_1_cf = row['str_annual_cash_flow']
    else:
        year_1_cf = row['ltr_annual_cash_flow']

    cumulative_cf = 0
    for year in range(1, hold_years + 1):
        if rental_type == 'ltr':
            year_cf = year_1_cf * ((1 + ANNUAL_RENT_INCREASE) ** (year - 1))
        else:
            year_cf = year_1_cf
        cumulative_cf += year_cf

    future_value = row['current_price'] * ((1 + ANNUAL_APPRECIATION) ** hold_years)
    appreciation_gain = future_value - row['current_price']

    loan_amount = row['current_price'] * (1 - down_payment_pct)

    # Calculate actual principal paydown using amortization formula
    monthly_rate = interest_rate / 12
    total_payments = loan_term_years * 12
    payments_made = hold_years * 12

    if monthly_rate > 0:
        remaining_balance = loan_amount * (
            ((1 + monthly_rate)**total_payments - (1 + monthly_rate)**payments_made) /
            ((1 + monthly_rate)**total_payments - 1)
        )
    else:
        remaining_balance = loan_amount * (1 - payments_made / total_payments)

    principal_paydown = loan_amount - remaining_balance

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


# =============================================================================
# METRO PROCESSING
# =============================================================================

def process_metro(metro_key: str, config: MetroConfig, df_housing: pd.DataFrame,
                  airbnb_data: dict) -> pd.DataFrame:
    """
    Process data for a single metro area.

    Args:
        metro_key: The metro identifier (e.g., 'austin')
        config: MetroConfig object with city-specific settings
        df_housing: National Zillow housing data
        airbnb_data: Dict mapping metro_key to cleaned Airbnb DataFrame

    Returns:
        DataFrame with processed neighborhood data for this metro
    """
    print(f"\n{'='*60}")
    print(f"Processing: {config.display_name}")
    print(f"{'='*60}")

    # Filter housing data for this metro (supports multiple cities)
    cities_to_include = config.get_zillow_cities()
    df_metro_housing = df_housing[
        (df_housing['RegionType'] == 'neighborhood') &
        (df_housing['City'].isin(cities_to_include))
    ].copy()

    if len(df_metro_housing) == 0:
        print(f"  WARNING: No neighborhoods found for {', '.join(cities_to_include)}")
        return pd.DataFrame()

    print(f"  Found {len(df_metro_housing)} neighborhoods across {len(cities_to_include)} cities")

    # Calculate appreciation metrics
    date_columns = [col for col in df_metro_housing.columns if col.startswith('20')]
    latest_date = date_columns[-1]
    df_metro_housing['current_price'] = df_metro_housing[latest_date]

    print(f"  Calculating appreciation metrics (data through {latest_date})...")

    df_metro_housing['baseline_cagr'] = df_metro_housing.apply(
        lambda row: calculate_weighted_cagr(row, date_columns), axis=1
    )

    # Recovery CAGR (2023-present)
    recovery_start = '2023-01-31'
    if recovery_start in df_metro_housing.columns:
        recovery_start_date = datetime.strptime(recovery_start, '%Y-%m-%d')
        recovery_end_date = datetime.strptime(latest_date, '%Y-%m-%d')
        recovery_years = (recovery_end_date - recovery_start_date).days / 365.25

        if recovery_years > 0:
            df_metro_housing['recovery_cagr'] = (
                (df_metro_housing[latest_date] / df_metro_housing[recovery_start]) ** (1/recovery_years) - 1
            ) * 100
        else:
            df_metro_housing['recovery_cagr'] = 0
    else:
        df_metro_housing['recovery_cagr'] = 0

    # Peak analysis
    peak_columns = [col for col in date_columns if '2022' in col]
    if peak_columns:
        df_metro_housing['peak_price_2022'] = df_metro_housing[peak_columns].max(axis=1)
        df_metro_housing['distance_from_peak'] = (
            (df_metro_housing['current_price'] / df_metro_housing['peak_price_2022']) - 1
        ) * 100
    else:
        df_metro_housing['peak_price_2022'] = df_metro_housing['current_price']
        df_metro_housing['distance_from_peak'] = 0

    print(f"  Median CAGR: {df_metro_housing['baseline_cagr'].median():.2f}%")

    # Process Airbnb data if available
    if config.has_str_data and metro_key in airbnb_data:
        print(f"  Processing STR data...")
        airbnb_by_neighborhood = airbnb_data[metro_key]
        print(f"  {len(airbnb_by_neighborhood)} neighborhoods with {MIN_AIRBNB_LISTINGS}+ listings")
    else:
        print(f"  STR data: Not available (LTR only)")
        airbnb_by_neighborhood = pd.DataFrame()

    # Prepare housing data for merge
    df_housing_merge = df_metro_housing[[
        'RegionName', 'current_price', 'baseline_cagr', 'recovery_cagr',
        'peak_price_2022', 'distance_from_peak'
    ]].copy()
    df_housing_merge.columns = [
        'neighborhood', 'current_price', 'baseline_cagr', 'recovery_cagr',
        'peak_price_2022', 'distance_from_peak'
    ]
    df_housing_merge['neighborhood'] = df_housing_merge['neighborhood'].astype(str)

    # Merge with Airbnb data
    if not airbnb_by_neighborhood.empty:
        df_merged = df_housing_merge.merge(
            airbnb_by_neighborhood,
            on='neighborhood',
            how='left'
        )
    else:
        df_merged = df_housing_merge.copy()
        df_merged['median_nightly_rate'] = 0
        df_merged['median_monthly_str_income'] = 0
        df_merged['median_bedrooms'] = 0
        df_merged['occupancy_rate'] = 0
        df_merged['listing_count'] = 0

    # Fill missing STR data
    df_merged['median_nightly_rate'] = df_merged['median_nightly_rate'].fillna(0)
    df_merged['median_monthly_str_income'] = df_merged['median_monthly_str_income'].fillna(0)
    df_merged['median_bedrooms'] = df_merged['median_bedrooms'].fillna(0)
    df_merged['occupancy_rate'] = df_merged['occupancy_rate'].fillna(0)
    df_merged['listing_count'] = df_merged['listing_count'].fillna(0).astype(int)

    # Calculate operating expenses using metro-specific rates
    print(f"  Calculating operating expenses (tax: {config.property_tax_rate*100:.1f}%, ins: {config.insurance_rate*100:.1f}%)...")

    loan_amount = df_merged['current_price'] * (1 - DOWN_PAYMENT_PCT)
    monthly_interest_rate = INTEREST_RATE / 12
    num_payments = LOAN_TERM_YEARS * 12

    df_merged['monthly_mortgage'] = (
        loan_amount *
        (monthly_interest_rate * (1 + monthly_interest_rate)**num_payments) /
        ((1 + monthly_interest_rate)**num_payments - 1)
    )

    # Use metro-specific rates
    df_merged['monthly_property_tax'] = df_merged['current_price'] * config.property_tax_rate / 12
    df_merged['monthly_insurance'] = df_merged['current_price'] * config.insurance_rate / 12
    df_merged['monthly_maintenance'] = df_merged['current_price'] * MAINTENANCE_RATE / 12

    df_merged['monthly_base_costs'] = (
        df_merged['monthly_mortgage'] +
        df_merged['monthly_property_tax'] +
        df_merged['monthly_insurance'] +
        df_merged['monthly_maintenance']
    )

    # STR-specific costs
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

    # STR cash flow
    df_merged['str_monthly_cash_flow'] = (
        df_merged['median_monthly_str_income'] - df_merged['str_total_costs']
    )
    df_merged['str_annual_cash_flow'] = df_merged['str_monthly_cash_flow'] * 12
    df_merged['str_cash_on_cash_return'] = (
        df_merged['str_annual_cash_flow'] / (df_merged['current_price'] * DOWN_PAYMENT_PCT) * 100
    )

    # LTR calculations using metro-specific tiers
    print(f"  Calculating LTR rents using metro-specific tiers...")
    df_merged['estimated_ltr_rent'] = df_merged['current_price'].apply(config.calculate_ltr_rent)
    df_merged['ltr_effective_rent'] = df_merged['estimated_ltr_rent'] * (1 - LTR_VACANCY_RATE)

    df_merged['ltr_monthly_cash_flow'] = df_merged['ltr_effective_rent'] - df_merged['monthly_base_costs']
    df_merged['ltr_annual_cash_flow'] = df_merged['ltr_monthly_cash_flow'] * 12
    df_merged['ltr_cash_on_cash_return'] = (
        df_merged['ltr_annual_cash_flow'] / (df_merged['current_price'] * DOWN_PAYMENT_PCT) * 100
    )

    # Calculate ROI
    print(f"  Calculating ROI ({HOLD_PERIOD_YEARS}-year hold)...")

    str_roi = df_merged.apply(
        lambda row: calculate_total_roi(row, HOLD_PERIOD_YEARS, 'str', INTEREST_RATE, DOWN_PAYMENT_PCT, LOAN_TERM_YEARS),
        axis=1
    )
    str_roi.columns = ['str_' + col for col in str_roi.columns]

    ltr_roi = df_merged.apply(
        lambda row: calculate_total_roi(row, HOLD_PERIOD_YEARS, 'ltr', INTEREST_RATE, DOWN_PAYMENT_PCT, LOAN_TERM_YEARS),
        axis=1
    )
    ltr_roi.columns = ['ltr_' + col for col in ltr_roi.columns]

    df_merged = pd.concat([df_merged, str_roi, ltr_roi], axis=1)

    # Add metro identifiers
    df_merged['metro'] = metro_key
    df_merged['metro_display'] = config.display_name
    df_merged['state'] = config.state
    df_merged['has_str_data'] = config.has_str_data

    # Summary
    positive_str = (df_merged['str_monthly_cash_flow'] > 0).sum() if config.has_str_data else 0
    positive_ltr = (df_merged['ltr_monthly_cash_flow'] > 0).sum()

    print(f"  Results: {len(df_merged)} neighborhoods")
    print(f"    Median price: ${df_merged['current_price'].median():,.0f}")
    print(f"    LTR positive cash flow: {positive_ltr}/{len(df_merged)}")
    if config.has_str_data:
        print(f"    STR positive cash flow: {positive_str}/{len(df_merged)}")

    return df_merged


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Process real estate data for Investment Analyzer')
    parser.add_argument('--metro', type=str, default=None,
                        help='Process specific metro only (default: all)')
    parser.add_argument('--housing-file', type=str, default='data/raw/zillow_zhvi_neighborhoods.csv',
                        help='Path to Zillow housing CSV')
    parser.add_argument('--output', type=str, default='data/processed/neighborhoods_multi_metro.csv',
                        help='Output CSV path')
    parser.add_argument('--legacy-output', type=str, default='data/processed/neighborhoods_v1.2_complete.csv',
                        help='Legacy Austin-only output for backward compatibility')
    args = parser.parse_args()

    print("="*80)
    print("REAL ESTATE INVESTMENT ANALYZER v2.0 - DATA PROCESSING")
    print("="*80)
    print(f"\nStarting analysis at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # Load configuration
    loader = get_config_loader()
    print(f"Loaded configuration for {len(loader.list_metros())} metros")
    print(f"  STR data available: {', '.join(loader.get_metros_with_str())}")
    print(f"  LTR only: {', '.join(loader.get_metros_without_str())}")

    # Load Zillow housing data
    print(f"\nLoading Zillow housing data from {args.housing_file}...")
    try:
        df_housing = pd.read_csv(args.housing_file)
        print(f"  Loaded {len(df_housing):,} rows")
    except FileNotFoundError:
        print(f"ERROR: Housing data not found at {args.housing_file}")
        sys.exit(1)

    # Load Airbnb data for metros that have it
    airbnb_data = {}
    print("\nLoading Airbnb data...")

    for metro_key in loader.get_metros_with_str():
        config = loader.get_metro(metro_key)
        if config.airbnb_file:
            airbnb_path = f'data/raw/{config.airbnb_file}'
            if Path(airbnb_path).exists():
                print(f"  Loading {config.display_name} from {config.airbnb_file}...")
                try:
                    df_airbnb_raw = pd.read_csv(airbnb_path, compression='gzip')
                    print(f"    Raw listings: {len(df_airbnb_raw):,}")
                    df_airbnb_clean = clean_airbnb_data(df_airbnb_raw)
                    print(f"    After cleaning: {len(df_airbnb_clean):,}")
                    airbnb_data[metro_key] = process_airbnb_for_metro(df_airbnb_clean)
                except Exception as e:
                    print(f"    WARNING: Could not load {config.airbnb_file}: {e}")
            else:
                print(f"  WARNING: {config.airbnb_file} not found for {config.display_name}")

    # Determine which metros to process
    if args.metro:
        metros_to_process = [args.metro]
        if args.metro not in loader.list_metros():
            print(f"ERROR: Unknown metro '{args.metro}'. Available: {loader.list_metros()}")
            sys.exit(1)
    else:
        metros_to_process = loader.list_metros()

    print(f"\nProcessing {len(metros_to_process)} metro(s)...")

    # Process each metro
    all_results = []
    for metro_key in metros_to_process:
        try:
            config = loader.get_metro(metro_key)
            df_metro = process_metro(metro_key, config, df_housing, airbnb_data)
            if not df_metro.empty:
                all_results.append(df_metro)
        except Exception as e:
            print(f"ERROR processing {metro_key}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Combine all metros
    if all_results:
        df_final = pd.concat(all_results, ignore_index=True)

        # Save consolidated output
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        df_final.to_csv(args.output, index=False)
        print(f"\n{'='*60}")
        print(f"SAVED: {args.output}")
        print(f"  Total neighborhoods: {len(df_final)}")
        print(f"  Metros: {df_final['metro'].nunique()}")
        print(f"  Columns: {len(df_final.columns)}")

        # Also save Austin-only for backward compatibility
        df_austin = df_final[df_final['metro'] == 'austin']
        if len(df_austin) > 0:
            df_austin.to_csv(args.legacy_output, index=False)
            print(f"\nLegacy Austin output: {args.legacy_output} ({len(df_austin)} neighborhoods)")
    else:
        print("\nERROR: No data processed successfully")
        sys.exit(1)

    print(f"\nCompleted at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)


if __name__ == "__main__":
    main()
