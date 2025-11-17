"""
Calculate realistic LTR rents using proportional scaling from Zillow data.

Approach:
1. Austin MSA median rent: $1,601/month (from ZORI - all rental types)
2. For single-family homes, apply 1.5x multiplier = $2,402 baseline
3. Scale each neighborhood proportionally to its price vs median
4. If neighborhood is 150% of median price → rent is 150% of baseline rent
"""

import pandas as pd

# Load data
df_merged = pd.read_csv('data/processed/neighborhoods_v1.2_complete.csv')

# Austin MSA baseline (from ZORI latest data)
AUSTIN_MSA_RENT_ALL_TYPES = 1601.08  # Includes apartments, condos
SFH_MULTIPLIER = 1.5  # Single-family homes rent for more than apartments
AUSTIN_SFH_BASELINE_RENT = AUSTIN_MSA_RENT_ALL_TYPES * SFH_MULTIPLIER

print(f"Austin MSA Baseline Rent (all types): ${AUSTIN_MSA_RENT_ALL_TYPES:,.0f}/month")
print(f"Single-Family Home Multiplier: {SFH_MULTIPLIER}x")
print(f"SFH Baseline Rent: ${AUSTIN_SFH_BASELINE_RENT:,.0f}/month")

# Calculate median home price across our neighborhoods
median_home_price = df_merged['current_price'].median()
print(f"\nMedian Home Price (our neighborhoods): ${median_home_price:,.0f}")

# Calculate proportional rents
df_merged['price_ratio'] = df_merged['current_price'] / median_home_price
df_merged['realistic_ltr_rent'] = df_merged['price_ratio'] * AUSTIN_SFH_BASELINE_RENT

print(f"\n{'Neighborhood':<20} | {'Price':>10} | {'Old (0.8%)':>12} | {'New (Scaled)':>14} | {'Ratio':>6}")
print("-" * 85)

for _, row in df_merged.head(15).iterrows():
    old_rent = row['estimated_ltr_rent']
    new_rent = row['realistic_ltr_rent']
    print(f"{row['neighborhood'][:19]:<20} | ${row['current_price']:>9,.0f} | ${old_rent:>11,.0f} | ${new_rent:>13,.0f} | {row['price_ratio']:>5.2f}x")

print(f"\n{'='*85}")
print(f"{'MEDIAN':<20} | ${median_home_price:>9,.0f} | ${df_merged['estimated_ltr_rent'].median():>11,.0f} | ${df_merged['realistic_ltr_rent'].median():>13,.0f} |")
print(f"{'='*85}")

# Calculate what this means for actual rent-to-price ratio
avg_ratio = (df_merged['realistic_ltr_rent'] / df_merged['current_price']).median()
print(f"\nResulting median rent-to-price ratio: {avg_ratio*100:.3f}%")
print(f"(Previous fixed 0.8% was {'higher' if 0.008 > avg_ratio else 'lower'} than data-driven)")

# Save comparison
comparison = df_merged[['neighborhood', 'current_price', 'estimated_ltr_rent', 'realistic_ltr_rent', 'price_ratio']].copy()
comparison.to_csv('ltr_rent_comparison.csv', index=False)
print(f"\n✅ Saved comparison to ltr_rent_comparison.csv")
