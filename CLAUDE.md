# Claude Code Workflow Instructions

## Development Methodology

**Always discuss logic, confirm data understanding, and validate methodology before generating code. Wait for explicit approval before implementation.**

### Workflow Rules:

1. **Plan Before Coding**
   - Discuss the approach and logic first
   - Confirm understanding of the data structure and requirements
   - Validate the methodology with the user
   - Wait for explicit "yes" or "proceed" before writing code

2. **No Assumptions**
   - Ask clarifying questions when requirements are ambiguous
   - Verify data sources and column names before using them
   - Confirm calculations and formulas before implementing

3. **Incremental Progress**
   - Break complex tasks into smaller, reviewable steps
   - Show intermediate results for validation
   - Test assumptions with sample data before full implementation

4. **Communication**
   - Explain trade-offs and alternatives
   - Present options when multiple valid approaches exist
   - Be transparent about limitations and uncertainties

## Git Commit Guidelines

- Focus on what changed and why, not how
- Do NOT include mentions of Claude, AI assistance, or tool credits
- Do NOT include "Co-Authored-By" attributions
- Use clear, concise language

---

## Project Status (Updated Jan 2026)

### Overview
Real estate investment analysis app with ML-powered appreciation predictions. Helps users identify neighborhoods with strong ROI potential across Texas and Florida metros.

### Current Model: Model C (24-Month Lookback)

**Architecture:**
- **Type**: RandomForest price prediction model
- **Training Data**: 2002-2024 (36,681 examples)
- **Price Range**: $170,000 - $1,000,000 (training population bounds)
- **Min History**: 24 months of price data required
- **Features**: 14 features including cagr_2yr (not 3yr/5yr due to lookback constraint)

**Forward Validation (November 2025):**
- **MAPE**: 2.63% (excellent)
- **R²**: 0.99
- **Coverage**: 3,684 neighborhoods with predictions

**Appreciation Derivation:**
```
appreciation_rate = (predicted_price - current_price) / current_price * 100
```

### Key Files

```
app.py                                    # Main Streamlit app
process_data.py                           # Data processing script
config/metros.yaml                        # Metro configuration
config/metro_config.py                    # Config loader

ml/models/predictor.py                    # Appreciation predictor (loads CSV)
ml/artifacts/model_c/price_model.joblib   # Trained Model C
ml/artifacts/model_c/model_info.json      # Model metadata

notebooks/08_clean_pipeline_v2.ipynb      # Canonical ML pipeline

data/processed/appreciation_predictions_current.csv  # Pre-computed predictions
data/processed/neighborhoods_multi_metro.csv         # App neighborhood data
data/processed/ml_data_v3.pkl                        # Training data
```

### Critical Implementation Details

**1. Sanity Caps on Appreciation:**
- All predictions capped to -10% to +15% annual
- Applied in notebook during prediction generation
- Also applied in app.py for fallback values (baseline_cagr)

**2. Conservative Multi-Year Projection:**
- Year 1: Use ML prediction (capped -10% to +15%)
- Years 2+: min(ML rate, 7%) for positive, 0% for negative
- Prevents unrealistic compounding of high appreciation rates

**3. Negative Appreciation Floor:**
- Declining markets: Year 1 decline only, then flat
- Prevents unrealistic depreciation to $0 over long holds

**4. Metro Key Matching:**
- Predictions use `display_metro` (dallas, fort_worth, miami, fort_lauderdale)
- NOT `training_metro` (dfw, south_florida)
- This enables 71.4% key matching between predictions and app data

**5. Price Filter:**
- Only neighborhoods within $170k-$1M get ML predictions
- Outside this range falls back to baseline_cagr (also capped)

### Data Flow

```
Zillow ZHVI data (raw)
    ↓
08_clean_pipeline_v2.ipynb (feature extraction, model training)
    ↓
appreciation_predictions_current.csv (pre-computed)
    ↓
predictor.py (loads CSV, provides lookup)
    ↓
app.py (uses predictions for ROI calculations)
```

### Metros Supported

**Training Groups (10):**
- DFW (combined: dallas, fort_worth)
- South Florida (combined: miami, fort_lauderdale)
- austin, houston, san_antonio, waco, abilene
- tampa, orlando, jacksonville

**Display Metros (12):**
- dallas, fort_worth, austin, houston, san_antonio, waco, abilene
- miami, fort_lauderdale, tampa, orlando, jacksonville

### Known Limitations

1. **29% fallback rate**: 856 neighborhoods use baseline_cagr instead of ML predictions due to naming differences between Zillow data and predictions
2. **Price bounds**: Homes outside $170k-$1M don't get ML predictions
3. **1-year horizon**: Model predicts 1 year ahead; multi-year is extrapolated with caps

### Future Enhancements (Deferred)

- FRED economic data integration (interest rates, employment)
- Confidence intervals for predictions
- Multi-horizon models (3yr, 5yr direct prediction)
- Improve neighborhood name matching for higher coverage

### Design Decisions

- **ML as default**: Predictions are always used, not a toggle
- **Conservative projections**: Cap long-term appreciation at 7%/year
- **Pre-computed predictions**: CSV lookup, not real-time inference
- **Single canonical notebook**: 08_clean_pipeline_v2.ipynb does everything

---

## Monthly Update Workflow

When new Zillow ZHVI data is released:

```bash
# 1. Download new ZHVI file from Zillow and save to:
data/raw/zillow_zhvi_neighborhoods_updated.csv

# 2. Open and run the notebook (all cells):
notebooks/08_clean_pipeline_v2.ipynb

# 3. Commit and push updated predictions:
git add data/processed/appreciation_predictions_current.csv
git commit -m "Update predictions with [Month Year] Zillow data"
git push
```

Streamlit Cloud will auto-redeploy with new predictions.

**Note:** The model file (`price_model.joblib`) stays local - only the predictions CSV is pushed.
