"""
Implement price-tier based LTR rent estimation.

Uses realistic Austin SFH bedroom counts and rent ranges by price tier:
- <$300k    → 2-3BR → $2,000-2,500/mo
- $300-500k → 3-4BR → $2,800-3,500/mo
- $500-700k → 4BR   → $3,800-4,500/mo
- >$700k    → 4-5BR → $5,000-7,000/mo
"""

import pandas as pd
import numpy as np

def calculate_ltr_rent_by_price_tier(price):
    """
    Calculate realistic LTR rent based on home price tier.
    Uses Austin market research for typical SFH rents.
    """
    if price < 300000:
        # Entry-level SFH: 2-3BR, older homes, outer neighborhoods
        # Typical rent: $2,000-2,500/month
        return price * 0.0075  # 0.75% of home value

    elif price < 500000:
        # Mid-market SFH: 3-4BR, established neighborhoods
        # Typical rent: $2,800-3,500/month
        return price * 0.0065  # 0.65% of home value

    elif price < 700000:
        # Upper-mid SFH: 4BR, desirable areas
        # Typical rent: $3,800-4,500/month
        return price * 0.0060  # 0.60% of home value

    else:
        # Luxury SFH: 4-5BR, premium neighborhoods
        # Typical rent: $5,000-7,000/month
        return price * 0.0065  # 0.65% of home value (premium homes)

# Load current data
df = pd.read_csv('data/processed/neighborhoods_v1.2_complete.csv')

# Calculate price-tier based rents
df['price_tier_ltr_rent'] = df['current_price'].apply(calculate_ltr_rent_by_price_tier)

print("="*90)
print("PRICE-TIER BASED LTR RENT CALCULATION")
print("="*90)

print("\n📊 Methodology:")
print("   <$300k:    0.75% of value (Entry-level SFH, 2-3BR)")
print("   $300-500k: 0.65% of value (Mid-market SFH, 3-4BR)")
print("   $500-700k: 0.60% of value (Upper-mid SFH, 4BR)")
print("   >$700k:    0.65% of value (Luxury SFH, 4-5BR)")

# Comparison
print(f"\n📈 Comparison: Old (0.8% flat) vs New (Price-Tier)\n")
print(f"{'Neighborhood':<20} | {'Price':>10} | {'Tier':>10} | {'Old (0.8%)':>12} | {'New (Tier)':>12} | {'Diff':>7}")
print("-" * 95)

for _, row in df.iterrows():
    old_rent = row['estimated_ltr_rent']
    new_rent = row['price_tier_ltr_rent']
    diff = ((new_rent - old_rent) / old_rent * 100) if old_rent > 0 else 0

    # Determine tier label
    if row['current_price'] < 300000:
        tier = "<$300k"
    elif row['current_price'] < 500000:
        tier = "$300-500k"
    elif row['current_price'] < 700000:
        tier = "$500-700k"
    else:
        tier = ">$700k"

    print(f"{row['neighborhood'][:19]:<20} | ${row['current_price']:>9,.0f} | {tier:>10} | "
          f"${old_rent:>11,.0f} | ${new_rent:>11,.0f} | {diff:>6.0f}%")

# Summary by tier
print(f"\n{'='*95}")
print(f"{'SUMMARY BY TIER':<20} | {'Avg Price':>10} | {'Count':>10} | {'Avg Old Rent':>12} | {'Avg New Rent':>12} |")
print(f"{'='*95}")

tiers = [
    ("<$300k", 0, 300000),
    ("$300-500k", 300000, 500000),
    ("$500-700k", 500000, 700000),
    (">$700k", 700000, float('inf'))
]

for tier_name, min_price, max_price in tiers:
    tier_data = df[(df['current_price'] >= min_price) & (df['current_price'] < max_price)]
    if len(tier_data) > 0:
        avg_price = tier_data['current_price'].mean()
        count = len(tier_data)
        avg_old = tier_data['estimated_ltr_rent'].mean()
        avg_new = tier_data['price_tier_ltr_rent'].mean()
        print(f"{tier_name:<20} | ${avg_price:>9,.0f} | {count:>10} | ${avg_old:>11,.0f} | ${avg_new:>11,.0f} |")

print(f"{'='*95}")

# Overall comparison
print(f"\n💡 Overall Statistics:")
print(f"   Median home price: ${df['current_price'].median():,.0f}")
print(f"   Old method (0.8% flat):")
print(f"      - Median rent: ${df['estimated_ltr_rent'].median():,.0f}/month")
print(f"      - Ratio: {(df['estimated_ltr_rent'].median() / df['current_price'].median() * 100):.3f}%")
print(f"   New method (price-tier):")
print(f"      - Median rent: ${df['price_tier_ltr_rent'].median():,.0f}/month")
print(f"      - Ratio: {(df['price_tier_ltr_rent'].median() / df['current_price'].median() * 100):.3f}%")

# Reality check examples
print(f"\n✅ Reality Check (New Method):")
print(f"   $250k home → ${calculate_ltr_rent_by_price_tier(250000):,.0f}/month (0.75%)")
print(f"   $400k home → ${calculate_ltr_rent_by_price_tier(400000):,.0f}/month (0.65%)")
print(f"   $600k home → ${calculate_ltr_rent_by_price_tier(600000):,.0f}/month (0.60%)")
print(f"   $800k home → ${calculate_ltr_rent_by_price_tier(800000):,.0f}/month (0.65%)")

print(f"\n💾 Recommendation: {'ACCEPT' if abs(df['price_tier_ltr_rent'].median() - 3500) < 1000 else 'REVIEW'}")
print(f"   (Target median rent for $541k homes: ~$3,200-3,800/month)")

# Save comparison
comparison = df[['neighborhood', 'current_price', 'estimated_ltr_rent', 'price_tier_ltr_rent']].copy()
comparison['percent_change'] = ((comparison['price_tier_ltr_rent'] - comparison['estimated_ltr_rent']) /
                                comparison['estimated_ltr_rent'] * 100)
comparison.to_csv('price_tier_ltr_comparison.csv', index=False)
print(f"\n💾 Saved comparison to price_tier_ltr_comparison.csv")
