import pandas as pd
import sys

# Test if the data file exists and loads
try:
    df = pd.read_csv('data/processed/neighborhoods_v1.2_complete.csv')
    print(f'✅ Data loads successfully')
    print(f'   Rows: {len(df)}')
    print(f'   Columns: {len(df.columns)}')
    print(f'   Sample neighborhoods: {list(df["neighborhood"].head(3))}')

    # Check for required columns
    required_cols = ['neighborhood', 'current_price', 'baseline_cagr', 'str_monthly_cash_flow',
                     'median_monthly_str_income', 'median_nightly_rate', 'occupancy_rate',
                     'listing_count', 'estimated_ltr_rent']

    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        print(f'⚠️  Missing columns: {missing}')
    else:
        print(f'✅ All required columns present')

    # Check data quality
    print(f'   Price range: ${df["current_price"].min():,.0f} - ${df["current_price"].max():,.0f}')
    print(f'   Positive STR cash flow: {(df["str_monthly_cash_flow"] > 0).sum()}/{len(df)}')

except FileNotFoundError:
    print('❌ Data file not found at data/processed/neighborhoods_v1.2_complete.csv')
    sys.exit(1)
except Exception as e:
    print(f'❌ Error loading data: {e}')
    sys.exit(1)

print('\n✅ Data validation complete - ready for Streamlit!')
