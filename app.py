import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go

# Page configuration
st.set_page_config(
    page_title="Austin Investment Analyzer v1.2",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# DATA LOADING
# =============================================================================

@st.cache_data
def load_data():
    """Load v1.2 processed data"""
    try:
        df = pd.read_csv('data/processed/neighborhoods_v1.2_complete.csv')
        return df
    except FileNotFoundError:
        st.error("v1.2 data not found. Please run the austin_analyzer_v1.2.ipynb notebook first to generate processed data.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        st.stop()

# Load data
df = load_data()

# =============================================================================
# HEADER
# =============================================================================

st.title("Austin Real Estate Investment Analyzer v1.2")
st.markdown("---")

# =============================================================================
# SIDEBAR - USER INPUTS
# =============================================================================

st.sidebar.header("Investment Parameters")

# Budget range
st.sidebar.subheader("Budget Range")
budget_min, budget_max = st.sidebar.slider(
    "Property Price Range",
    min_value=150000,
    max_value=800000,
    value=(250000, 400000),
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

# Rental Type
rental_type = st.sidebar.radio(
    "Rental Type",
    ["Short-Term Rental (STR/Airbnb)", "Long-Term Rental (LTR)"]
)
rental_code = "str" if "Short-Term" in rental_type else "ltr"

# LTR rent calculation info now shown in results section

# Advanced assumptions
with st.sidebar.expander("⚙️ Advanced Assumptions"):
    property_tax = st.number_input("Property Tax Rate %", value=2.2, min_value=0.5, max_value=4.0, step=0.1) / 100
    insurance = st.number_input("Insurance Rate %", value=0.5, min_value=0.3, max_value=2.0, step=0.1) / 100
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

    # Calculate interest for each month of the year
    annual_interest = 0
    remaining_balance = loan_amount

    start_month = (year - 1) * 12
    end_month = year * 12

    for month in range(start_month, end_month):
        if month >= total_payments:
            break

        # Interest for this month
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
            # LTR - use price-tier method for realistic SFH rents
            def calc_ltr_rent_tier(price):
                """Price-tier: <$300k=0.80%, $300-500k=0.65%, $500-700k=0.60%, >$700k=0.65%"""
                if price < 300000:
                    return price * 0.0080
                elif price < 500000:
                    return price * 0.0065
                elif price < 700000:
                    return price * 0.0060
                else:
                    return price * 0.0065

            def get_ltr_rent_percentage(price):
                """Get the percentage text for LTR rent calculation"""
                if price < 300000:
                    return "0.80%"
                elif price < 500000:
                    return "0.65%"
                elif price < 700000:
                    return "0.60%"
                else:
                    return "0.65%"

            df_filtered['calc_ltr_rent'] = df_filtered['current_price'].apply(calc_ltr_rent_tier)
            df_filtered['calc_ltr_rent_pct'] = df_filtered['current_price'].apply(get_ltr_rent_percentage)
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

        # Principal paydown (approximation)
        if down_payment_pct == 1.0:
            # Cash purchase - no principal paydown
            principal_paydown = 0
        else:
            # Financed - estimate principal paid down
            loan_amount = df_filtered['current_price'] * (1 - down_payment_pct)
            principal_paydown = loan_amount * 0.15 * (hold_period / 5)

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

    top_5 = df_sorted.head(5)

    for idx, row in top_5.iterrows():
        rank = top_5.index.get_loc(idx) + 1

        with st.expander(f"#{rank} - {row['neighborhood']}", expanded=(rank <= 3)):

            # Zillow link
            st.markdown(f"**[Search Zillow for {row['neighborhood']}](https://www.zillow.com/austin-tx/{row['neighborhood'].replace(' ', '-').lower()}/)** ")

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
                st.write(f"Property Tax ({property_tax*100:.1f}%): ${row['calc_property_tax']:,.0f}")
                st.write(f"Insurance ({insurance*100:.1f}%): ${row['calc_insurance']:,.0f}")
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
                new_monthly_costs = (new_mortgage +
                                   (future_value * property_tax / 12) +
                                   (future_value * insurance / 12) +
                                   (future_value * maintenance / 12) +
                                   (row['calc_str_additional'] if rental_code == "str" else 0))
                new_cash_flow = future_effective_rent - new_monthly_costs

                st.write(f"Equity at Exit: ${equity:,.0f}")
                st.write(f"Cash Out (Tax-Free): ${refi_cash_out:,.0f}")
                st.write(f"New Loan (6% rate): ${new_loan_amount:,.0f}")
                st.write(f"New Mortgage (P&I): ${new_mortgage:,.0f}/mo")
                st.write(f"Future Rent: ${future_rent:,.0f}/mo")
                st.write(f"**New Cash Flow: ${new_cash_flow:,.0f}/mo**")

    st.markdown("---")

    # Visualization
    st.header("Visual Analysis")

    # Create comparison chart
    top_10 = df_sorted.head(10)

    if "Cash Flow" in strategy:
        fig = px.bar(
            top_10,
            x='neighborhood',
            y='calc_monthly_cf',
            title=f'Top 10 Neighborhoods - Monthly Cash Flow ({rental_code.upper()})',
            labels={'calc_monthly_cf': 'Monthly Cash Flow ($)', 'neighborhood': 'Neighborhood'},
            color='calc_monthly_cf',
            color_continuous_scale='RdYlGn'
        )
        fig.update_layout(xaxis_tickangle=-45, height=500)
        st.plotly_chart(fig, use_container_width=True)

    elif "Total ROI" in strategy:
        fig = px.bar(
            top_10,
            x='neighborhood',
            y='calc_total_roi',
            title=f'Top 10 Neighborhoods - {int(hold_period)}-Year Total ROI ({rental_code.upper()})',
            labels={'calc_total_roi': f'{int(hold_period)}-Year Total ROI (%)', 'neighborhood': 'Neighborhood'},
            color='calc_total_roi',
            color_continuous_scale='Blues'
        )
        fig.update_layout(xaxis_tickangle=-45, height=500)
        st.plotly_chart(fig, use_container_width=True)

    else:  # Appreciation
        fig = px.bar(
            top_10,
            x='neighborhood',
            y='baseline_cagr',
            title='Top 10 Neighborhoods - Historical Appreciation (2015-2019 CAGR)',
            labels={'baseline_cagr': 'Annual Growth Rate (%)', 'neighborhood': 'Neighborhood'},
            color='baseline_cagr',
            color_continuous_scale='Greens'
        )
        fig.update_layout(xaxis_tickangle=-45, height=500)
        st.plotly_chart(fig, use_container_width=True)

    # Cash Flow vs Total ROI scatter
    fig2 = px.scatter(
        df_sorted.head(20),
        x='calc_monthly_cf',
        y='calc_total_roi',
        size='current_price',
        color='baseline_cagr',
        hover_name='neighborhood',
        title=f'Top 20: Cash Flow vs Total ROI (sized by price, colored by appreciation)',
        labels={
            'calc_monthly_cf': 'Monthly Cash Flow ($)',
            'calc_total_roi': f'{int(hold_period)}-Year Total ROI (%)',
            'baseline_cagr': 'Historical CAGR (%)'
        }
    )
    st.plotly_chart(fig2, use_container_width=True)

else:
    # Initial state
    st.info("Configure your investment parameters in the sidebar and click 'Find Best Neighborhoods'")

# Footer
st.markdown("---")

with st.expander("🔍 What's New in v1.2?"):
    st.markdown("""
    **Critical Fixes:**
    - ✅ **Property Tax Rate:** Corrected from 1.2% to 2.2% (Austin actual)
    - ✅ **STR Operating Expenses:** Added utilities ($200/mo), cleaning (~$300/mo), supplies ($75/mo), platform fees (3%)
    - ✅ **Total Impact:** ~$700/month correction in cash flow calculations

    **New Features:**
    - 🎯 **3 Primary Investment Strategies:** Cash Flow, Total ROI, Appreciation
    - 🔑 **Entry/Exit Analysis:** Detailed comparisons in results (Cash vs Financed, Sell vs Refi)
    - ⚙️ **User-Adjustable Assumptions:** Customize all parameters to your situation
    - 💰 **100% Cash Purchase Option:** Compare cash vs financed scenarios
    - 🏘️ **Neighborhood Filter:** Select specific neighborhoods to analyze
    - 📊 **Price-Tier LTR Rents:** Realistic rent estimates based on home value
    - 📈 **Rental Growth Modeling:** LTR rents increase 3-5% annually
    - 🎨 **Interactive Visualizations:** Dynamic charts with Plotly
    """)

st.markdown("""
**Created by:** [Cleburn Walker](https://linkedin.com/in/cleburnwalker) | [GitHub](https://github.com/cleburn)

**Data Sources:** Zillow ZHVI (Housing Prices), Zillow ZORI (Rent Prices), Inside Airbnb (through June 2025)
""")
