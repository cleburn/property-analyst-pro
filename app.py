import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import sys

# Add project root to path for config imports
sys.path.insert(0, str(Path(__file__).parent))

from config.metro_config import get_config_loader, get_metro_config

# Page configuration
st.set_page_config(
    page_title="Real Estate Investment Analyzer v2.0",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# CONFIGURATION
# =============================================================================

# Load metro configuration
config_loader = get_config_loader()

# =============================================================================
# DATA LOADING
# =============================================================================

@st.cache_data
def load_data():
    """Load multi-metro processed data"""
    # Try new multi-metro file first, fall back to legacy
    try:
        df = pd.read_csv('data/processed/neighborhoods_multi_metro.csv')
        return df
    except FileNotFoundError:
        try:
            # Fall back to legacy Austin-only file
            df = pd.read_csv('data/processed/neighborhoods_v1.2_complete.csv')
            # Add metro columns for backward compatibility
            if 'metro' not in df.columns:
                df['metro'] = 'austin'
                df['metro_display'] = 'Austin, TX'
                df['state'] = 'TX'
                df['has_str_data'] = True
            return df
        except FileNotFoundError:
            st.error("Data not found. Please run process_data.py first to generate processed data.")
            st.stop()
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        st.stop()

# Load all data once
df_all = load_data()

# =============================================================================
# LOCATION SELECTION (Top of sidebar) - State filter → Multi-select metros
# =============================================================================

st.sidebar.header("Location Selection")

# Get available metros from data
available_metros = df_all['metro'].unique().tolist()
metro_options = {m: config_loader.get_metro(m).display_name for m in available_metros if m in config_loader.list_metros()}

# Build state groupings from available metros
available_states = {}
for metro_key in metro_options.keys():
    state = config_loader.get_metro(metro_key).state
    if state not in available_states:
        available_states[state] = []
    available_states[state].append(metro_key)

# State display names
state_names = {
    'TX': 'Texas',
    'FL': 'Florida'
}

# State selection (All States / Florida / Texas)
state_options = ['All States'] + sorted([state_names.get(s, s) for s in available_states.keys()])
selected_state = st.sidebar.selectbox(
    "State",
    options=state_options,
    index=0,
    help="Filter by state or view all states"
)

# Determine which metros to show based on state selection
if selected_state == 'All States':
    filtered_metro_keys = list(metro_options.keys())
    filtered_state_codes = list(available_states.keys())
else:
    # Convert display name back to state code
    state_code = [k for k, v in state_names.items() if v == selected_state][0]
    filtered_metro_keys = available_states.get(state_code, [])
    filtered_state_codes = [state_code]

# Sort metros alphabetically by display name
sorted_metro_keys = sorted(filtered_metro_keys, key=lambda x: metro_options[x])

# Metro multi-select (empty = all metros in selected state)
selected_metros = st.sidebar.multiselect(
    "Metro Areas (leave empty for all)",
    options=sorted_metro_keys,
    format_func=lambda x: metro_options[x],
    default=[],
    help="Select specific metros or leave empty to include all"
)

# If no metros selected, use all metros from the filtered state
if not selected_metros:
    selected_metros = filtered_metro_keys

# Filter data for selected metros
df = df_all[df_all['metro'].isin(selected_metros)].copy()

if len(df) == 0:
    st.error("No data available for the selected location(s). Please check that data has been processed.")
    st.stop()

# Determine STR availability across selected metros
selected_configs = [get_metro_config(m) for m in selected_metros]
any_has_str = any(c.has_str_data for c in selected_configs)
all_have_str = all(c.has_str_data for c in selected_configs)

# For single metro, use its config; for multiple, we'll use aggregated settings
if len(selected_metros) == 1:
    metro_config = selected_configs[0]
    is_multi_metro = False
else:
    # Create a "virtual" config for display purposes - use first metro as base
    metro_config = selected_configs[0]
    is_multi_metro = True

# Show selection summary
if is_multi_metro:
    st.sidebar.caption(f"Analyzing {len(selected_metros)} metros, {len(df)} neighborhoods")
else:
    cities_in_metro = metro_config.get_zillow_cities()
    if len(cities_in_metro) > 1:
        st.sidebar.caption(f"Includes: {', '.join(cities_in_metro[:5])}{'...' if len(cities_in_metro) > 5 else ''}")

# Show STR availability status
if not any_has_str:
    st.sidebar.warning("STR (Airbnb) data is not available for the selected location(s). Only LTR analysis is available.")
elif not all_have_str and is_multi_metro:
    str_metros = [c.display_name for c in selected_configs if c.has_str_data]
    st.sidebar.info(f"STR data available for: {', '.join(str_metros)}")

st.sidebar.markdown("---")

# =============================================================================
# HEADER
# =============================================================================

# Dynamic title based on selection
if is_multi_metro:
    if selected_state == 'All States':
        title_location = "Multi-State"
    else:
        title_location = selected_state
    st.title(f"{title_location} Real Estate Investment Analyzer")
    st.caption(f"v2.0 - Analyzing {len(selected_metros)} metros, {len(df)} neighborhoods")
else:
    st.title(f"{metro_config.display_name} Real Estate Investment Analyzer")
    st.caption("v2.0 - Multi-Metro Edition")
st.markdown("---")

# =============================================================================
# SIDEBAR - USER INPUTS
# =============================================================================

st.sidebar.header("Investment Parameters")

# Budget range
st.sidebar.subheader("Budget Range")
budget_min, budget_max = st.sidebar.slider(
    "Property Price Range",
    min_value=50000,
    max_value=1000000,
    value=(100000, 500000),
    step=10000,
    format="$%d"
)

# Neighborhood filter
st.sidebar.subheader("Neighborhood Filter")
all_neighborhoods = sorted(df['neighborhood'].unique().tolist())
neighborhood_filter = st.sidebar.multiselect(
    "Select Specific Neighborhoods (optional)",
    options=all_neighborhoods,
    default=[],
    help="Leave empty to search all neighborhoods, or select specific ones to analyze"
)

# Financing
st.sidebar.subheader("Financing Terms")
down_payment_pct = st.sidebar.slider(
    "Down Payment %",
    min_value=3.5,
    max_value=100.0,
    value=20.0,
    step=2.5,
    help="100% = Cash purchase (no financing)"
) / 100

# Show cash purchase indicator
if down_payment_pct == 1.0:
    st.sidebar.success("Cash Purchase (No Financing)")
    interest_rate = 0.0  # No interest for cash purchase
else:
    interest_rate = st.sidebar.slider(
        "Interest Rate %",
        min_value=4.0,
        max_value=10.0,
        value=7.0,
        step=0.25
    ) / 100

# Investment Strategy
st.sidebar.subheader("Strategy Selection")
strategy = st.sidebar.selectbox(
    "Primary Investment Strategy",
    ["Cash Flow (Monthly Profit)",
     "Total ROI",
     "Appreciation Potential"]
)

# Rental Type - conditionally show based on STR data availability
if any_has_str:
    rental_type = st.sidebar.radio(
        "Rental Type",
        ["Short-Term Rental (STR/Airbnb)", "Long-Term Rental (LTR)"]
    )
    rental_code = "str" if "Short-Term" in rental_type else "ltr"
    # Note about partial STR coverage in multi-metro
    if is_multi_metro and not all_have_str and rental_code == "str":
        st.sidebar.caption("Note: STR data only available for some selected metros. Others will be excluded from STR results.")
else:
    st.sidebar.info("Only LTR analysis available for selected location(s) (no Airbnb data)")
    rental_type = "Long-Term Rental (LTR)"
    rental_code = "ltr"

# LTR rent calculation info now shown in results section

# Advanced assumptions
with st.sidebar.expander("⚙️ Advanced Assumptions"):
    if is_multi_metro:
        # Multi-metro mode: use pre-calculated values, just show info
        st.caption("Using pre-calculated metro-specific rates for tax, insurance, and rent estimates.")
        st.markdown("""
        **Rates by metro are applied automatically:**
        - Property tax: 1.6% - 2.4% depending on metro
        - Insurance: 0.5% - 1.2% depending on metro
        """)
        # Set default values (won't be used for main calculations - pre-calculated data is used)
        # But needed for any fallback calculations that reference these variables
        property_tax = 0.022  # Average fallback
        insurance = 0.008  # Average fallback
    else:
        # Single metro: allow override
        default_tax = metro_config.property_tax_rate * 100
        default_insurance = metro_config.insurance_rate * 100

        property_tax = st.number_input(
            "Property Tax Rate %",
            value=default_tax,
            min_value=0.5,
            max_value=4.0,
            step=0.1,
            help=metro_config.property_tax_note if metro_config.property_tax_note else None
        ) / 100

        # Show Florida property tax note if applicable
        if metro_config.property_tax_note:
            st.caption(f"Note: {metro_config.property_tax_note}")

        insurance = st.number_input(
            "Insurance Rate %",
            value=default_insurance,
            min_value=0.3,
            max_value=3.0,  # Higher max for Florida
            step=0.1,
            help="Varies by location. Florida/coastal areas typically higher due to hurricane risk."
        ) / 100

    maintenance = st.number_input("Maintenance Rate %", value=1.0, min_value=0.5, max_value=3.0, step=0.1) / 100

    if rental_code == "str":
        st.markdown("**STR-Specific:**")
        str_utilities = st.number_input("Monthly Utilities $", value=200, min_value=0, max_value=500, step=25)
        str_cleaning = st.number_input("Cleaning per Turnover $", value=100, min_value=0, max_value=300, step=25)
        turnovers_per_month = st.number_input("Turnovers/Month", value=3, min_value=1, max_value=10, step=1)
        str_supplies = st.number_input("Monthly Supplies $", value=75, min_value=0, max_value=200, step=25)
        annual_rent_increase = 0.0  # Not applicable for STR
    else:
        st.markdown("**LTR-Specific:**")
        annual_rent_increase = st.number_input("Annual Rent Increase %", value=4.0, min_value=0.0, max_value=10.0, step=0.5) / 100

    hold_period = st.number_input("Hold Period (Years)", value=5, min_value=1, max_value=30, step=1)

    st.markdown("**Tax Considerations:**")
    tax_bracket = st.selectbox(
        "Federal Tax Bracket %",
        [10, 12, 22, 24, 32, 35, 37],
        index=3,  # Default to 24%
        help="Used to calculate tax benefits from mortgage interest deduction and depreciation"
    ) / 100

    st.info("Appreciation rates are auto-calculated from neighborhood historical data")

# Analyze button
analyze_button = st.sidebar.button("Find Best Neighborhoods", type="primary")

st.sidebar.markdown("---")

# =============================================================================
# TAX BENEFIT CALCULATION FUNCTIONS
# =============================================================================

def calculate_mortgage_interest(loan_amount, interest_rate, year):
    """
    Calculate mortgage interest paid in a specific year.
    Uses amortization schedule to determine interest vs principal split.

    Args:
        loan_amount: Original loan amount
        interest_rate: Annual interest rate (e.g., 0.07 for 7%)
        year: Which year (1-30)

    Returns:
        Total interest paid in that year
    """
    if loan_amount == 0 or interest_rate == 0:
        return 0

    monthly_rate = interest_rate / 12
    total_payments = 30 * 12
    monthly_payment = loan_amount * (monthly_rate * (1 + monthly_rate)**total_payments) / ((1 + monthly_rate)**total_payments - 1)

    remaining_balance = loan_amount
    start_month = (year - 1) * 12
    end_month = year * 12

    # First, iterate through prior months to get correct starting balance
    for month in range(0, start_month):
        if month >= total_payments:
            break
        monthly_interest = remaining_balance * monthly_rate
        monthly_principal = monthly_payment - monthly_interest
        remaining_balance -= monthly_principal

    # Now calculate interest for the target year
    annual_interest = 0
    for month in range(start_month, end_month):
        if month >= total_payments:
            break

        monthly_interest = remaining_balance * monthly_rate
        monthly_principal = monthly_payment - monthly_interest

        annual_interest += monthly_interest
        remaining_balance -= monthly_principal

    return annual_interest

def calculate_depreciation_deduction(property_value):
    """
    Calculate annual depreciation deduction for residential rental property.

    Residential rental properties depreciate over 27.5 years.
    Land cannot be depreciated - assume 20% land value, 80% building.

    Args:
        property_value: Total property value

    Returns:
        Annual depreciation deduction
    """
    depreciable_basis = property_value * 0.80  # 80% building, 20% land
    annual_depreciation = depreciable_basis / 27.5
    return annual_depreciation

def calculate_tax_benefits(property_value, loan_amount, interest_rate, operating_expenses, year=1):
    """
    Calculate total tax deductions and tax savings for rental property.

    Tax deductible items:
    - Mortgage interest
    - Depreciation
    - Operating expenses (property tax, insurance, maintenance, utilities, etc.)

    Args:
        property_value: Property value
        loan_amount: Mortgage loan amount
        interest_rate: Annual interest rate
        operating_expenses: Annual operating expenses
        year: Which year of ownership (for interest calculation)

    Returns:
        dict with interest, depreciation, total_deductions, tax_savings
    """
    # Calculate mortgage interest for this year
    mortgage_interest = calculate_mortgage_interest(loan_amount, interest_rate, year)

    # Calculate depreciation
    depreciation = calculate_depreciation_deduction(property_value)

    # Total deductions
    total_deductions = mortgage_interest + depreciation + operating_expenses

    return {
        'mortgage_interest': mortgage_interest,
        'depreciation': depreciation,
        'operating_expenses': operating_expenses,
        'total_deductions': total_deductions
    }

# =============================================================================
# MAIN CONTENT
# =============================================================================

if analyze_button:

    # Recalculate with user parameters
    with st.spinner("Calculating investment metrics with your parameters..."):

        # Filter by budget
        df_filtered = df[
            (df['current_price'] >= budget_min) &
            (df['current_price'] <= budget_max)
        ].copy()

        # Apply neighborhood filter if specified
        if neighborhood_filter:
            df_filtered = df_filtered[df_filtered['neighborhood'].isin(neighborhood_filter)]
            if len(df_filtered) == 0:
                st.warning(f"None of the selected neighborhoods are in the ${budget_min:,}-${budget_max:,} price range.")
                st.info(f"Try adjusting your budget or selecting different neighborhoods.")
                st.stop()

        if len(df_filtered) == 0:
            st.warning("No neighborhoods match your budget range. Try adjusting the slider.")
            st.stop()

        # Filter by rental type - STR requires Airbnb data
        if rental_code == "str":
            df_filtered = df_filtered[df_filtered['listing_count'] > 0]
            if len(df_filtered) == 0:
                st.warning("No neighborhoods in your budget have STR (Airbnb) data available.")
                st.info("Try selecting LTR (Long-term rental) instead, or adjust your budget range.")
                st.stop()

        # Recalculate with user inputs
        # Mortgage calculation
        if down_payment_pct == 1.0:
            # Cash purchase - no mortgage
            df_filtered['calc_monthly_mortgage'] = 0
        else:
            # Financed purchase
            loan_amount = df_filtered['current_price'] * (1 - down_payment_pct)
            monthly_interest_rate = interest_rate / 12
            num_payments = 30 * 12

            df_filtered['calc_monthly_mortgage'] = (
                loan_amount *
                (monthly_interest_rate * (1 + monthly_interest_rate)**num_payments) /
                ((1 + monthly_interest_rate)**num_payments - 1)
            )

        # Operating expenses
        if is_multi_metro:
            # Use pre-calculated values from data (metro-specific rates already applied)
            df_filtered['calc_property_tax'] = df_filtered['monthly_property_tax']
            df_filtered['calc_insurance'] = df_filtered['monthly_insurance']
        else:
            # Single metro - use user-specified or default rates
            df_filtered['calc_property_tax'] = df_filtered['current_price'] * property_tax / 12
            df_filtered['calc_insurance'] = df_filtered['current_price'] * insurance / 12
        df_filtered['calc_maintenance'] = df_filtered['current_price'] * maintenance / 12

        df_filtered['calc_base_costs'] = (
            df_filtered['calc_monthly_mortgage'] +
            df_filtered['calc_property_tax'] +
            df_filtered['calc_insurance'] +
            df_filtered['calc_maintenance']
        )

        # STR-specific costs
        if rental_code == "str":
            df_filtered['calc_str_utilities'] = str_utilities
            df_filtered['calc_str_cleaning'] = str_cleaning * turnovers_per_month
            df_filtered['calc_str_supplies'] = str_supplies
            df_filtered['calc_str_platform_fees'] = df_filtered['median_monthly_str_income'] * 0.03

            df_filtered['calc_str_additional'] = (
                df_filtered['calc_str_utilities'] +
                df_filtered['calc_str_cleaning'] +
                df_filtered['calc_str_supplies'] +
                df_filtered['calc_str_platform_fees']
            )

            df_filtered['calc_total_costs'] = df_filtered['calc_base_costs'] + df_filtered['calc_str_additional']
            df_filtered['calc_monthly_cf'] = df_filtered['median_monthly_str_income'] - df_filtered['calc_total_costs']
            income_col = 'median_monthly_str_income'
        else:
            # LTR - use metro-specific price-tier method for realistic SFH rents
            if is_multi_metro:
                # Use pre-calculated LTR rent from data (metro-specific tiers already applied)
                df_filtered['calc_ltr_rent'] = df_filtered['estimated_ltr_rent']
                # Calculate rent-to-price ratio for display
                df_filtered['calc_ltr_rent_pct'] = (df_filtered['calc_ltr_rent'] / df_filtered['current_price'] * 100).apply(lambda x: f"{x:.2f}%")
            else:
                # Single metro - use metro config to calculate
                df_filtered['calc_ltr_rent'] = df_filtered['current_price'].apply(metro_config.calculate_ltr_rent)
                df_filtered['calc_ltr_rent_pct'] = df_filtered['current_price'].apply(metro_config.get_ltr_rate_display)
            df_filtered['calc_ltr_effective_rent'] = df_filtered['calc_ltr_rent'] * 0.92  # 8% vacancy
            df_filtered['calc_total_costs'] = df_filtered['calc_base_costs']
            df_filtered['calc_monthly_cf'] = df_filtered['calc_ltr_effective_rent'] - df_filtered['calc_total_costs']
            income_col = 'calc_ltr_rent'

        # Cash-on-cash return (pre-tax)
        df_filtered['calc_coc_return'] = (
            df_filtered['calc_monthly_cf'] * 12 / (df_filtered['current_price'] * down_payment_pct) * 100
        )

        # =============================================================================
        # TAX BENEFIT CALCULATIONS
        # =============================================================================

        # Calculate loan amount for tax benefit calculations
        df_filtered['calc_loan_amount'] = df_filtered['current_price'] * (1 - down_payment_pct)

        # Calculate annual operating expenses (tax deductible)
        # Includes: property tax, insurance, maintenance, and STR-specific costs
        df_filtered['calc_annual_operating_expenses'] = (
            df_filtered['calc_property_tax'] * 12 +
            df_filtered['calc_insurance'] * 12 +
            df_filtered['calc_maintenance'] * 12
        )

        if rental_code == "str":
            df_filtered['calc_annual_operating_expenses'] += df_filtered['calc_str_additional'] * 12

        # Calculate tax benefits for Year 1
        def apply_tax_benefits(row):
            benefits = calculate_tax_benefits(
                property_value=row['current_price'],
                loan_amount=row['calc_loan_amount'],
                interest_rate=interest_rate,
                operating_expenses=row['calc_annual_operating_expenses'],
                year=1
            )
            return pd.Series(benefits)

        tax_benefits_df = df_filtered.apply(apply_tax_benefits, axis=1)
        df_filtered['calc_mortgage_interest'] = tax_benefits_df['mortgage_interest']
        df_filtered['calc_depreciation'] = tax_benefits_df['depreciation']
        df_filtered['calc_total_deductions'] = tax_benefits_df['total_deductions']

        # Calculate tax savings
        df_filtered['calc_tax_savings'] = df_filtered['calc_total_deductions'] * tax_bracket

        # After-tax cash flow = Pre-tax cash flow + Tax savings
        df_filtered['calc_monthly_cf_after_tax'] = df_filtered['calc_monthly_cf'] + (df_filtered['calc_tax_savings'] / 12)

        # After-tax cash-on-cash return
        df_filtered['calc_coc_return_after_tax'] = (
            df_filtered['calc_monthly_cf_after_tax'] * 12 / (df_filtered['current_price'] * down_payment_pct) * 100
        )

        # Total ROI (simplified)
        down_payment = df_filtered['current_price'] * down_payment_pct
        closing_costs = df_filtered['current_price'] * 0.03
        total_invested = down_payment + closing_costs

        # Cumulative cash flow with rent increases (LTR only)
        cumulative_cf = 0
        for year in range(1, int(hold_period) + 1):
            if rental_code == "ltr":
                # LTR: apply annual rent increase
                year_cf = df_filtered['calc_monthly_cf'] * 12 * ((1 + annual_rent_increase) ** (year - 1))
            else:
                # STR: flat revenue (already accounts for seasonal fluctuations)
                year_cf = df_filtered['calc_monthly_cf'] * 12
            cumulative_cf += year_cf

        # Appreciation - use neighborhood's actual historical CAGR
        # Convert CAGR percentage to decimal (baseline_cagr is already in %)
        annual_appreciation = df_filtered['baseline_cagr'] / 100
        future_value = df_filtered['current_price'] * ((1 + annual_appreciation) ** hold_period)
        appreciation_gain = future_value - df_filtered['current_price']

        # Principal paydown (actual amortization calculation)
        if down_payment_pct == 1.0:
            # Cash purchase - no principal paydown
            principal_paydown = 0
        else:
            # Financed - calculate actual principal paid using amortization formula
            loan_amount = df_filtered['current_price'] * (1 - down_payment_pct)
            monthly_rate = interest_rate / 12
            total_payments = 30 * 12  # 360 months
            payments_made = int(hold_period) * 12

            # Remaining balance after hold_period years
            remaining_balance = loan_amount * (
                ((1 + monthly_rate)**total_payments - (1 + monthly_rate)**payments_made) /
                ((1 + monthly_rate)**total_payments - 1)
            )
            principal_paydown = loan_amount - remaining_balance

        total_return = cumulative_cf + appreciation_gain + principal_paydown
        df_filtered['calc_total_roi'] = (total_return / total_invested) * 100
        df_filtered['calc_annualized_roi'] = ((1 + df_filtered['calc_total_roi']/100) ** (1/hold_period) - 1) * 100

    # Sort by strategy
    if "Cash Flow" in strategy:
        sort_col = 'calc_monthly_cf'
        metric_name = "Monthly Cash Flow"
    elif "Total ROI" in strategy:
        sort_col = 'calc_total_roi'
        metric_name = f"{int(hold_period)}-Year Total ROI"
    else:  # Appreciation
        sort_col = 'baseline_cagr'
        metric_name = "Historical Appreciation (CAGR)"

    df_sorted = df_filtered.sort_values(sort_col, ascending=False)

    # Summary metrics
    st.header("Investment Analysis Summary")

    # Show filter status
    if neighborhood_filter:
        st.info(f"Filtering to {len(neighborhood_filter)} selected neighborhood(s): {', '.join(neighborhood_filter)}")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if neighborhood_filter:
            st.metric("Neighborhoods Selected", f"{len(neighborhood_filter)}")
        else:
            st.metric("Neighborhoods Available", f"{len(df):,}")

    with col2:
        st.metric("Matching Criteria", f"{len(df_filtered):,}")

    with col3:
        positive_cf_after_tax = (df_filtered['calc_monthly_cf_after_tax'] > 0).sum()
        st.metric("Positive Cash Flow (After-Tax)", f"{positive_cf_after_tax} ({positive_cf_after_tax/len(df_filtered)*100:.1f}%)")

    with col4:
        median_price = df_filtered['current_price'].median()
        st.metric("Median Price", f"${median_price:,.0f}")

    st.markdown("---")

    # Top 5 neighborhoods
    st.header(f"Top 5 Neighborhoods: {strategy}")
    st.caption("Click any result to expand full investment details")

    top_5 = df_sorted.head(5)

    for idx, row in top_5.iterrows():
        rank = top_5.index.get_loc(idx) + 1

        # Build key metric string based on strategy
        if "Cash Flow" in strategy:
            metric_str = f"${row['calc_monthly_cf']:,.0f}/mo"
        elif "Total ROI" in strategy:
            metric_str = f"{row['calc_total_roi']:.1f}% ROI"
        else:  # Appreciation
            metric_str = f"{row['baseline_cagr']:.1f}% CAGR"

        # Include metro name in title for multi-metro mode
        if is_multi_metro:
            expander_title = f"#{rank} - {row['neighborhood']} ({row['metro_display']}) — {metric_str}"
        else:
            expander_title = f"#{rank} - {row['neighborhood']} — {metric_str}"

        with st.expander(expander_title, expanded=False):

            # Zillow link - use the specific metro's URL template
            row_metro_config = get_metro_config(row['metro'])
            zillow_url = row_metro_config.zillow_search_url.format(
                neighborhood=row['neighborhood'].replace(' ', '-').lower()
            )
            st.markdown(f"**[Search Zillow for {row['neighborhood']}]({zillow_url})** ")

            # Key metrics
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric(
                    "Property Price",
                    f"${row['current_price']:,.0f}",
                    f"{row['distance_from_peak']:.1f}% from '22 peak"
                )

            with col2:
                st.metric(
                    "Monthly Cash Flow (Pre-Tax)",
                    f"${row['calc_monthly_cf']:,.0f}",
                    f"${row['calc_monthly_cf_after_tax']:,.0f} after-tax"
                )

            with col3:
                st.metric(
                    "Cash-on-Cash Return (After-Tax)",
                    f"{row['calc_coc_return_after_tax']:.1f}%",
                    f"{row['calc_coc_return']:.1f}% pre-tax"
                )

            with col4:
                st.metric(
                    f"{int(hold_period)}-Year Total ROI",
                    f"{row['calc_total_roi']:.1f}%",
                    f"{row['calc_annualized_roi']:.1f}% annualized"
                )

            st.markdown("---")

            # Detailed breakdown
            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown("**Monthly Income:**")
                st.write(f"Rental Income ({rental_code.upper()}): ${row[income_col]:,.0f}")
                if rental_code == "str":
                    st.write(f"Occupancy Rate: {row['occupancy_rate']:.1f}%")
                    st.write(f"Nightly Rate: ${row['median_nightly_rate']:.0f}")
                    st.write(f"Listings in Area: {row['listing_count']:.0f}")
                else:
                    st.caption(f"Rents estimated at {row['calc_ltr_rent_pct']} of home value")

            with col2:
                st.markdown("**Monthly Costs:**")
                if down_payment_pct == 1.0:
                    st.write(f"Mortgage: $0 (Cash Purchase)")
                else:
                    st.write(f"Mortgage (P&I): ${row['calc_monthly_mortgage']:,.0f}")
                # Show rates - calculate from values in multi-metro mode
                if is_multi_metro:
                    tax_rate_display = row['calc_property_tax'] * 12 / row['current_price'] * 100
                    ins_rate_display = row['calc_insurance'] * 12 / row['current_price'] * 100
                else:
                    tax_rate_display = property_tax * 100
                    ins_rate_display = insurance * 100
                st.write(f"Property Tax ({tax_rate_display:.1f}%): ${row['calc_property_tax']:,.0f}")
                st.write(f"Insurance ({ins_rate_display:.1f}%): ${row['calc_insurance']:,.0f}")
                st.write(f"Maintenance ({maintenance*100:.1f}%): ${row['calc_maintenance']:,.0f}")
                if rental_code == "str":
                    st.write(f"STR Operating Costs: ${row['calc_str_additional']:,.0f}")
                st.write(f"**TOTAL: ${row['calc_total_costs']:,.0f}**")

            with col3:
                st.markdown("**Tax Benefits (Year 1):**")
                st.write(f"Mortgage Interest: ${row['calc_mortgage_interest']:,.0f}")
                st.write(f"Depreciation: ${row['calc_depreciation']:,.0f}")
                st.write(f"Operating Expenses: ${row['calc_annual_operating_expenses']:,.0f}")
                st.write(f"**Total Deductions: ${row['calc_total_deductions']:,.0f}**")
                st.write(f"**Tax Savings ({tax_bracket*100:.0f}%): ${row['calc_tax_savings']:,.0f}/yr**")
                st.caption(f"${row['calc_tax_savings']/12:,.0f}/mo reduces effective costs")

            st.markdown("---")

            # Investment Summary Row
            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown("**Pre-Tax vs After-Tax:**")
                st.write(f"Pre-Tax Cash Flow: ${row['calc_monthly_cf']:,.0f}/mo")
                st.write(f"After-Tax Cash Flow: ${row['calc_monthly_cf_after_tax']:,.0f}/mo")
                st.write(f"**Tax Benefit: ${row['calc_monthly_cf_after_tax'] - row['calc_monthly_cf']:,.0f}/mo**")

            with col2:
                st.markdown("**Investment Summary:**")
                down = row['current_price'] * down_payment_pct
                if down_payment_pct == 1.0:
                    st.write(f"Cash Purchase: ${down:,.0f}")
                else:
                    st.write(f"Down Payment ({down_payment_pct*100:.0f}%): ${down:,.0f}")
                st.write(f"Est. Closing Costs: ${row['current_price']*0.03:,.0f}")
                st.write(f"**Total Initial Investment: ${down + row['current_price']*0.03:,.0f}**")

            with col3:
                st.markdown("**Property Details:**")
                st.write(f"Historical Appreciation: {row['baseline_cagr']:.2f}%/yr")
                st.write(f"Bedrooms (median): {row['median_bedrooms']:.0f}")
                st.write(f"Distance from '22 Peak: {row['distance_from_peak']:.1f}%")

            st.markdown("---")

            # Exit Strategy Analysis (Sell vs Refi)
            st.markdown(f"**Exit Strategy (After {int(hold_period)} Years, based on your selected entry strategy):**")
            col1, col2 = st.columns(2)

            # Calculate future property value
            appreciation_rate = row['baseline_cagr'] / 100
            future_value = row['current_price'] * ((1 + appreciation_rate) ** hold_period)
            appreciation_gain = future_value - row['current_price']

            # Calculate remaining mortgage balance if financed
            if down_payment_pct < 1.0:
                loan_amount = row['current_price'] * (1 - down_payment_pct)
                # Proper amortization formula
                monthly_rate = interest_rate / 12
                total_payments = 30 * 12  # 30-year loan
                payments_made = hold_period * 12

                # Remaining balance formula
                remaining_balance = loan_amount * (
                    ((1 + monthly_rate)**total_payments - (1 + monthly_rate)**payments_made) /
                    ((1 + monthly_rate)**total_payments - 1)
                )
            else:
                remaining_balance = 0

            # Calculate equity at exit
            equity = future_value - remaining_balance

            # Calculate future rent (LTR only - STR stays flat)
            if rental_code == "ltr":
                future_rent = row['calc_ltr_rent'] * ((1 + annual_rent_increase) ** hold_period)
                future_effective_rent = future_rent * 0.92  # 8% vacancy
            else:  # STR
                future_rent = row['median_monthly_str_income']  # STR stays at current level
                future_effective_rent = row['median_monthly_str_income']  # Already has occupancy applied

            with col1:
                st.markdown("*Sell (w/ Cap Gains Tax):*")
                # Long-term cap gains tax (15% for most investors)
                # Note: Investment property, no primary residence exclusion
                cap_gains_tax = appreciation_gain * 0.15

                net_proceeds = future_value - remaining_balance - cap_gains_tax - (future_value * 0.06)  # 6% selling costs

                st.caption("⚠️ Assumes 15% long-term cap gains rate. Consult tax professional for depreciation recapture.")

                st.write(f"Future Value: ${future_value:,.0f}")
                st.write(f"Equity: ${equity:,.0f}")
                st.write(f"Cap Gains Tax (15%): ${cap_gains_tax:,.0f}")
                st.write(f"Selling Costs (6%): ${future_value * 0.06:,.0f}")
                if remaining_balance > 0:
                    st.write(f"Pay Off Loan: ${remaining_balance:,.0f}")
                st.write(f"**Net Proceeds: ${net_proceeds:,.0f}**")

            with col2:
                st.markdown("*Cash-Out Refi (80% LTV):*")

                # Industry standard: 80% loan-to-value on future property value
                refi_closing_costs = 5000
                max_new_loan = future_value * 0.80

                # Cash out = new loan - remaining balance - closing costs
                refi_cash_out = max_new_loan - remaining_balance - refi_closing_costs

                # Verify sufficient equity
                if refi_cash_out < 0:
                    refi_cash_out = 0
                    max_new_loan = remaining_balance  # Can't refi if insufficient equity

                new_loan_amount = max_new_loan

                # New mortgage payment (6% refi rate, 30-year term)
                refi_rate = 0.06
                monthly_refi_rate = refi_rate / 12
                n_payments = 30 * 12
                new_mortgage = new_loan_amount * (monthly_refi_rate * (1 + monthly_refi_rate)**n_payments) / ((1 + monthly_refi_rate)**n_payments - 1)

                # New cash flow with updated rent and new mortgage
                # For multi-metro, calculate rates from current values; for single metro use user rates
                if is_multi_metro:
                    future_tax = row['calc_property_tax'] * (future_value / row['current_price'])
                    future_ins = row['calc_insurance'] * (future_value / row['current_price'])
                else:
                    future_tax = future_value * property_tax / 12
                    future_ins = future_value * insurance / 12
                new_monthly_costs = (new_mortgage +
                                   future_tax +
                                   future_ins +
                                   (future_value * maintenance / 12) +
                                   (row['calc_str_additional'] if rental_code == "str" else 0))
                new_cash_flow = future_effective_rent - new_monthly_costs

                st.write(f"Equity at Exit: ${equity:,.0f}")
                st.write(f"Cash Out (Tax-Free): ${refi_cash_out:,.0f}")
                st.write(f"New Loan (6% rate): ${new_loan_amount:,.0f}")
                st.write(f"New Mortgage (P&I): ${new_mortgage:,.0f}/mo")
                st.write(f"Future Rent: ${future_rent:,.0f}/mo")
                st.write(f"**New Cash Flow: ${new_cash_flow:,.0f}/mo**")

else:
    # Initial state
    st.info("Configure your investment parameters in the sidebar and click 'Find Best Neighborhoods'")

# Footer
st.markdown("---")

with st.expander("What's New in v2.0?"):
    st.markdown("""
    **v2.0 - Multi-Metro Edition:**
    - **Multiple Metro Areas:** Analyze real estate investments across Texas and Florida
    - **Metro-Specific Defaults:** Property tax rates, insurance rates, and LTR rent tiers customized per market
    - **Enhanced STR Filtering:** Improved outlier detection removes hotel rooms and statistical anomalies
    - **Corrected Principal Paydown:** Uses actual amortization formula (was overstated in v1.x)

    **Previous Fixes (v1.2):**
    - Property Tax Rate: Corrected from 1.2% to 2.2% (Austin actual)
    - STR Operating Expenses: Added utilities, cleaning, supplies, platform fees
    - 3 Primary Investment Strategies: Cash Flow, Total ROI, Appreciation
    - Exit Strategy Analysis: Sell vs Refinance comparisons
    """)

# Show current selection info
if is_multi_metro:
    metro_names = [get_metro_config(m).display_name for m in selected_metros]
    str_status = f"{sum(1 for c in selected_configs if c.has_str_data)}/{len(selected_metros)} metros have STR data"
    st.caption(f"Currently viewing: {len(selected_metros)} metros ({', '.join(metro_names[:3])}{'...' if len(metro_names) > 3 else ''}) | {str_status}")
else:
    st.caption(f"Currently viewing: {metro_config.display_name} | STR Data: {'Available' if metro_config.has_str_data else 'Not Available'}")

st.markdown("""
**Created by:** [Cleburn Walker](https://linkedin.com/in/cleburnwalker) | [GitHub](https://github.com/cleburn)

**Data Sources:** Zillow ZHVI (Housing Prices), Zillow ZORI (Rent Prices), Inside Airbnb
""")
