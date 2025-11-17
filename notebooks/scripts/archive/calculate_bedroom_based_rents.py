"""
Calculate LTR rents using bedroom count as a size proxy.

Approach:
1. Use median bedrooms per neighborhood from Airbnb data
2. Calculate Austin-wide rent per bedroom baseline
3. Apply to each neighborhood based on its bedroom count
4. Adjust for price tier (luxury homes command premium per bedroom)
"""

import pandas as pd
import numpy as np

# Load processed data
df = pd.read_csv('data/processed/neighborhoods_v1.2_complete.csv')

print("="*80)
print("BEDROOM-BASED LTR RENT CALCULATION")
print("="*80)

# Analyze bedroom distribution
print(f"\n📊 Bedroom Data by Neighborhood:")
print(f"   Median bedrooms across all neighborhoods: {df['median_bedrooms'].median():.1f}")
print(f"   Range: {df['median_bedrooms'].min():.0f} - {df['median_bedrooms'].max():.0f}")

# Austin market baseline rent per bedroom
# Research shows Austin LTR: ~$800-1200 per bedroom for SFH
# We'll use a tiered approach based on home price
def calculate_rent_per_bedroom(price, bedrooms):
    """
    Calculate rent per bedroom based on price tier.
    Higher-priced homes command premium per bedroom.
    """
    if price < 300000:
        # Affordable tier: $700/bedroom
        base_per_bedroom = 700
    elif price < 500000:
        # Mid-tier: $900/bedroom
        base_per_bedroom = 900
    elif price < 700000:
        # Upper-mid: $1100/bedroom
        base_per_bedroom = 1100
    else:
        # Luxury: $1300/bedroom
        base_per_bedroom = 1300

    # Additional premium for larger homes (economies of scale don't apply linearly)
    if bedrooms >= 4:
        base_per_bedroom *= 1.1  # 10% premium for 4+ bedroom homes

    return base_per_bedroom * bedrooms

# Calculate bedroom-based rents
df['bedroom_based_ltr_rent'] = df.apply(
    lambda row: calculate_rent_per_bedroom(row['current_price'], row['median_bedrooms']),
    axis=1
)

# Compare with current 0.8% method
print(f"\n📈 Comparison: 0.8% Method vs Bedroom-Based Method\n")
print(f"{'Neighborhood':<20} | {'Price':>10} | {'Beds':>5} | {'Old (0.8%)':>12} | {'New (Beds)':>12} | {'Diff':>8}")
print("-" * 95)

for _, row in df.head(15).iterrows():
    old_rent = row['estimated_ltr_rent']
    new_rent = row['bedroom_based_ltr_rent']
    diff = ((new_rent - old_rent) / old_rent * 100) if old_rent > 0 else 0

    print(f"{row['neighborhood'][:19]:<20} | ${row['current_price']:>9,.0f} | {row['median_bedrooms']:>4.0f}  | "
          f"${old_rent:>11,.0f} | ${new_rent:>11,.0f} | {diff:>6.0f}%")

# Summary statistics
print(f"\n{'='*95}")
print(f"{'MEDIAN':<20} | ${df['current_price'].median():>9,.0f} | {df['median_bedrooms'].median():>4.1f}  | "
      f"${df['estimated_ltr_rent'].median():>11,.0f} | ${df['bedroom_based_ltr_rent'].median():>11,.0f} |")
print(f"{'='*95}")

# Calculate effective rent-to-price ratios
df['old_ratio'] = df['estimated_ltr_rent'] / df['current_price'] * 100
df['new_ratio'] = df['bedroom_based_ltr_rent'] / df['current_price'] * 100

print(f"\n💡 Rent-to-Price Ratios:")
print(f"   Old method (0.8% flat): {df['old_ratio'].median():.3f}%")
print(f"   New method (bedroom-based): {df['new_ratio'].median():.3f}%")
print(f"   Range (new): {df['new_ratio'].min():.3f}% - {df['new_ratio'].max():.3f}%")

# Show by price tier
print(f"\n📊 Rent-to-Price Ratio by Home Price Tier (Bedroom Method):")
df['price_tier'] = pd.cut(df['current_price'],
                          bins=[0, 300000, 500000, 700000, float('inf')],
                          labels=['<$300k', '$300-500k', '$500-700k', '>$700k'])

for tier in ['<$300k', '$300-500k', '$500-700k', '>$700k']:
    tier_data = df[df['price_tier'] == tier]
    if len(tier_data) > 0:
        avg_ratio = tier_data['new_ratio'].mean()
        avg_rent = tier_data['bedroom_based_ltr_rent'].mean()
        avg_beds = tier_data['median_bedrooms'].mean()
        print(f"   {tier:<12}: {avg_ratio:.3f}% | Avg rent: ${avg_rent:,.0f} | Avg beds: {avg_beds:.1f}")

# Check which method seems more realistic
print(f"\n✅ Reality Check:")
print(f"   2BR in <$300k range: ${df[df['median_bedrooms']==2]['bedroom_based_ltr_rent'].min():,.0f}/mo")
print(f"   3BR in $300-500k range: ~${df[(df['median_bedrooms']==3) & (df['current_price'].between(300000, 500000))]['bedroom_based_ltr_rent'].mean():,.0f}/mo")
print(f"   4BR in >$700k range: ~${df[(df['median_bedrooms']>=4) & (df['current_price']>700000)]['bedroom_based_ltr_rent'].mean():,.0f}/mo")

# Save results
output = df[['neighborhood', 'current_price', 'median_bedrooms', 'estimated_ltr_rent',
             'bedroom_based_ltr_rent', 'old_ratio', 'new_ratio']].copy()
output.to_csv('bedroom_based_ltr_comparison.csv', index=False)
print(f"\n💾 Saved comparison to bedroom_based_ltr_comparison.csv")
