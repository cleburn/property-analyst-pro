import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Austin Investment Analyzer",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🏠 Austin Real Estate Investment Analyzer")

st.markdown("---")

@st.cache_data
def load_data():
    """Load and process all datasets"""
    try:
        df_final = pd.read_csv('data/processed/final_neighborhoods.csv')
        return df_final
    except Exception as e:
        st.error(f"❌ Error loading data: {str(e)}")
        st.error("If using Safari, please try Chrome browser for best experience.")
        st.stop()

# Load data
df = load_data()



st.success(f"✅ Data loaded: {len(df)} neighborhoods analyzed")

st.sidebar.header("Investment Criteria")

# Budget range slider
budget_min, budget_max = st.sidebar.slider(
    "Budget Range",
    min_value=150000,
    max_value=800000,
    value=(250000, 400000),  # Default values
    step=10000,
    format="$%d"
)

# Investment strategy selector
strategy = st.sidebar.selectbox(
    "Investment Strategy",
    ["Cash Flow", "Appreciation"]
)

# Rental type selector
rental_type = st.sidebar.radio(
    "Rental Type",
    ["STR (Short-term / Airbnb)", "LTR (Long-term)"]
)

# Convert rental type to code
rental_code = "STR" if "STR" in rental_type else "LTR"

# Analyze button
analyze_button = st.sidebar.button("🔍 Find Best Neighborhoods", type="primary")

st.sidebar.markdown("---")
st.sidebar.info(f"**Your Criteria:**\n\n- Budget: ${budget_min:,} - ${budget_max:,}\n- Strategy: {strategy}\n- Rental: {rental_code}")

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    st.header("Top Recommended Neighborhoods")

with col2:
    st.header("Analysis Summary")

st.markdown("---")


if analyze_button:

    # Filter by budget
    budget_filtered = df[
        (df['current_price_2025'] >= budget_min) &
        (df['current_price_2025'] <= budget_max)
    ]

    if len(budget_filtered) == 0:
        st.warning("No neighborhoods match your budget range. Try adjusting the slider.")
        st.stop()

    # Rank by strategy
    if strategy == "Cash Flow":
        if rental_code == "STR":
            sort_column = 'str_monthly_cash_flow'
        else:
            sort_column = 'ltr_monthly_cash_flow'
    else:  # Appreciation
        sort_column = 'baseline_cagr'

    # Get top 3
    top_3 = budget_filtered.sort_values(sort_column, ascending=False).head(3).reset_index(drop=True)
    top_3['rank'] = range(1, len(top_3) + 1)

    # Summary stats
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Neighborhoods Analyzed", len(df))

    with col2:
        st.metric("In Your Budget", len(budget_filtered))

    with col3:
        positive_cashflow = len(budget_filtered[budget_filtered[sort_column] > 0])
        st.metric("Positive Cash Flow", positive_cashflow)

    with col4:
        avg_price = budget_filtered['current_price_2025'].median()
        st.metric("Median Price", f"${avg_price:,.0f}")

    st.markdown("---")



    # Display results
    st.subheader(f"🎯 Top 3 Neighborhoods for {strategy} ({rental_code})")

    for idx, row in top_3.head(3).iterrows():
        neighborhood = row['neighborhood']
    
        with st.expander(f"#{row['rank']} - {neighborhood}", expanded=True):
        
            # Zillow search link
            st.markdown(f"🔍 **[Search Zillow for properties](https://www.zillow.com/austin-tx/)** → Search: **{neighborhood}**")
            st.markdown("---")
        
            col1, col2, col3 = st.columns(3)
            # ... rest of your code

            with col1:
                st.metric(
                    "Median Price",
                    f"${row['current_price_2025']:,.0f}",
                    f"{row['distance_from_peak']:.1f}% from peak"
                )

            with col2:
                if rental_code == "STR":
                    cash_flow = row['str_monthly_cash_flow']
                    st.metric(
                        "Monthly Cash Flow (STR)",
                        f"${cash_flow:,.0f}",
                        f"${cash_flow * 12:,.0f}/year"
                    )
                else:
                    cash_flow = row['ltr_monthly_cash_flow']
                    st.metric(
                        "Monthly Cash Flow (LTR)",
                        f"${cash_flow:,.0f}",
                        f"${cash_flow * 12:,.0f}/year"
                    )

            with col3:
                st.metric(
                    "Historical Growth",
                    f"{row['baseline_cagr']:.2f}%",
                    "annually (2015-2019)"
                )

            st.markdown("---")

            # Detailed metrics
            st.markdown("**💰 Investment Details:**")

            col1, col2 = st.columns(2)

            with col1:
                st.write(f"**Monthly Costs:** ${row['monthly_costs']:,.0f}")
                # Get correct income column based on rental type
                if rental_code == "STR":
                    income = row['median_monthly_str_income']
                else:  # LTR
                    income = row['estimated_ltr_monthly_rent']

                st.write(f"**Monthly Income ({rental_code}):** ${income:,.0f}")
                st.write(f"**Airbnb Listings:** {row['listing_count']:.0f}")

            with col2:
                down_payment = row['current_price_2025'] * 0.20
                st.write(f"**Down Payment (20%):** ${down_payment:,.0f}")
    
                # Calculate occupancy rate from listing data
                # Display occupancy rate (neighborhood-specific data)
                if pd.notna(row.get('occupancy_rate')) and row['occupancy_rate'] > 0:
                    st.write(f"**Occupancy Rate:** {row['occupancy_rate']:.1f}%")
                else:
                    st.write(f"**Est. Occupancy:** 62% (Austin avg)")

else:
    st.info("👈 Set your investment criteria in the sidebar and click 'Find Best Neighborhoods'")


# Footer
st.markdown("---")
st.markdown("""
**🚧 Beta Version - Proof of Concept**

This tool is an MVP demonstrating the core analysis pipeline. Current Version:
- Analyzes 185 Austin neighborhoods using Zillow & Airbnb data
- Data current through September 2025
- Assumes: 20% down payment, 7% interest rate, 30-year mortgage
- Conservative expense modeling (taxes, insurance, maintenance, vacancies)

---            

**Upcoming Features Include:**
- Additional metro areas (Houston, Dallas, San Antonio)
- Real-time MLS integration with live property listings
- Off-market deal sources (wholesalers, foreclosures, pre-foreclosures)
- Customizable financing scenarios (down payment %, interest rates, loan terms)
- Advanced filtering options (property condition, renovation potential, school districts)
- Portfolio optimization and diversification recommendations

---

**Created by:** [Cleburn Walker](https://linkedin.com/in/cleburnwalker) | [GitHub](https://github.com/cleburn)
""")