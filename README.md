# 🏠 Austin Real Estate Investment Analyzer v1.2

**AI-powered tool that helps real estate investors identify the best neighborhoods in Austin based on cash flow potential and appreciation metrics.**

**Version 1.2** brings critical calculation fixes, robust appreciation modeling using 25 years of data, and comprehensive exit strategy analysis.

---

## 📋 Table of Contents

1. [Try It Now](#-try-it-now)
2. [The Problem](#-the-problem)
3. [What It Does](#-what-it-does)
4. [How It Works](#-how-it-works)
5. [Key Insights](#-key-insights-from-current-data-v12)
6. [What's Next](#-whats-next)
7. [Technical Stack](#️-technical-stack)
8. [Contact & Collaboration](#-contact--collaboration)
9. [License](#-license)

---

## 🚀 Try It Now

### 🌐 Use the Live App (Recommended)

**[👉 Launch the Austin Investment Analyzer](https://austin-investment-analyzer.streamlit.app/)**

No installation required - just click and start analyzing neighborhoods!

### 📱 Mobile Compatibility
- ✅ Chrome (Android & iOS): Full support
- ⚠️ Safari (iOS): If app doesn't load, use Chrome or desktop view
- ✅ Desktop: All browsers supported

---

### 💻 Or Run Locally

**If you want to explore the code or run it on your own machine:**

1. **Clone the repository:**
```bash
   git clone https://github.com/cleburn/austin-investment-analyzer.git
   cd austin-investment-analyzer
```

2. **Install dependencies:**
```bash
   pip install -r requirements.txt
```

3. **Data is included:**
   - Pre-processed data (`data/processed/neighborhoods_v1.2_complete.csv`) is included in the repository
   - You can run the app immediately without regenerating data
   - (Optional) To regenerate data with updated sources: `python process_data.py`

4. **Run the web app:**
```bash
   streamlit run app.py
```

5. **Use the tool:**
   - The app will open automatically in your browser at `http://localhost:8501`
   - Adjust your budget range, investment strategy, and rental type in the sidebar
   - Click "Find Best Neighborhoods" to see your top 5 recommendations

---

## 🎯 The Problem

Real estate investors spend hours manually comparing neighborhoods, calculating ROI, and searching for properties that match their investment strategy. The process is:
- **Time-consuming:** Dozens of neighborhoods to evaluate across multiple data sources
- **Complex:** Requires understanding of cash flow formulas, appreciation trends, and market cycles
- **Overwhelming:** Balancing budget constraints with investment goals (cash flow vs. long-term growth)

**This tool automates the entire research process and delivers ranked recommendations in seconds.**

---

## 💡 What It Does

The analyzer takes your inputs:
1. **Budget Range** (e.g., $250K - $400K)
2. **Investment Strategy** (Cash Flow, Total ROI, or Appreciation)
3. **Rental Type** (Short-term rental (STR/Airbnb) or Long-term rental (LTR))
4. **Financing Terms** (Down payment 3.5%-100%, interest rate)
5. **Hold Period** (Investment timeframe for ROI calculations)

Then it:
- Analyzes 185 Austin neighborhoods to find the top 5 based on investor criteria and target
- Ranks them based on your chosen strategy
- Returns the **top 5 neighborhoods** with comprehensive metrics:
  - Monthly cash flow (STR and LTR with realistic rent estimates)
  - Total ROI over your hold period (includes cash flow + equity + appreciation)
  - Historical appreciation rates (weighted 2000-2025 CAGR)
  - Exit strategy analysis (Sell vs Refinance scenarios)
  - Current market position (distance from 2022 peak)
  - Confidence indicators (number of comparable listings)

**Note:** The web app focuses on delivering fast, personalized recommendations. Market trend visualizations are available in the `/visuals` folder. Historical analysis notebooks can be found in `/archive/notebooks`.
---

## 🔍 How It Works

### Data Sources
The tool combines three public datasets to create a complete picture:

1. **Zillow Home Value Index (ZHVI)** — Neighborhood-level median home prices from 2000-2025
2. **Zillow Observed Rent Index (ZORI)** — Long-term rental rates by neighborhood
3. **Inside Airbnb** — 15,000+ short-term rental listings with occupancy and income data

### Analysis Engine (v1.2 - Critical Fixes Applied)
For each neighborhood, the tool calculates:

**Cash Flow Strategy Metrics:**
- **STR income:** `nightly rate × 30 days × occupancy rate`
- **LTR income:** Price-tier based estimates (0.6-0.75% of home value monthly, varies by price range)
- **Monthly costs:** Mortgage (user-adjustable down payment & rate) + property tax (2.2% ✅ corrected from 1.2%) + insurance (0.5%) + maintenance (1%)
- **STR additional costs:** Utilities ($200), cleaning (~$300), supplies ($75), platform fees (3%) ✅
- **Net cash flow:** `Income - Total Costs`

**Total ROI Metrics:**
- **Cumulative cash flow** over hold period
- **Equity buildup** (principal paydown if financed)
- **Appreciation** (property value increase)
- **Total ROI:** `(Cash Flow + Equity + Appreciation) / Initial Investment`

**Appreciation Strategy Metrics:**
- **Weighted baseline CAGR (2000-2025):** Uses 25 years of data with reduced weight (0.3x) for anomaly period (June 2020 - April 2023) ✅
- Current recovery trajectory (2023-2025 CAGR)
- Distance from 2022 market peak (shows potential upside)

**Exit Strategy Analysis:**
- **Sell scenario:** Property value at exit, capital gains tax (15%), selling costs (6%), net proceeds
- **Refinance scenario:** 80% LTV cash-out, $5k closing costs, new loan amount, new mortgage payment, updated cash flow

**Tax Benefits:**
- **Mortgage interest deduction:** Calculated per year based on amortization schedule
- **Depreciation:** 27.5-year residential rental (80% of property value)
- **Operating expense deductions:** All rental expenses reduce taxable income
- **After-tax cash flow:** Shows realistic returns including tax savings

### Ranking System
Neighborhoods are ranked by the metric that matters most to your strategy:
- **Cash Flow investors:** Sorted by highest monthly profit (STR or LTR)
- **Total ROI investors:** Sorted by best risk-adjusted returns over hold period
- **Appreciation investors:** Sorted by strongest weighted historical growth (2000-2025 CAGR)

Only neighborhoods with **7+ Airbnb listings** and **complete data** across all sources are included (ensures reliable analysis).

---

## 📊 Key Insights from Current Data (v1.2)

**Austin Market Reality (as of November 2025):**
- **Pre-tax cash flow is challenging** — With corrected calculations (2.2% property tax + STR expenses), most neighborhoods show negative monthly pre-tax cash flow at 20% down / 7% interest
- **Tax benefits change the picture** — After accounting for mortgage interest deduction, depreciation, and operating expense deductions, many neighborhoods become cash-flow positive
- **Total ROI remains attractive** — Many neighborhoods show positive 5-year total ROI when appreciation, equity buildup, and tax benefits are included
- **LTR more viable than STR** — After fixes, LTR shows better pre-tax cash flow potential than STR in most neighborhoods
- **Appreciation opportunity exists** — Neighborhoods are 20-30% below their 2022 peaks with median 5.52% weighted CAGR (2000-2025)
- **Cash purchase changes the game** — 100% down payment eliminates mortgage interest but loses tax deduction benefits

**Market Stats:**
- **Median home price:** $541,399
- **Median weighted appreciation:** 5.52% CAGR (2000-2025)
- **Neighborhoods analyzed:** 185 Austin neighborhoods
- **Data through:** October 2025

**Key Takeaway:** v1.2's corrected calculations reveal that Austin real estate offers strong returns when tax benefits are included. While pre-tax cash flow can be challenging at current prices, **after-tax cash flow combined with appreciation** makes Austin an attractive market for buy-and-hold investors who can benefit from rental property tax advantages.

---

## 🚀 What's Next

### Current Version (v1.2 - Complete ✅)
**Major improvements over v1.0:**
- ✅ **Critical calculation fixes:** Property tax (2.2%), STR operating expenses (~$600/mo), proper mortgage amortization
- ✅ **Tax benefit calculations:** Mortgage interest deduction, depreciation (27.5 years), operating expense deductions
- ✅ **Pre-tax and after-tax metrics:** Shows both perspectives for accurate ROI analysis
- ✅ **Robust appreciation modeling:** Weighted regression using 25 years of data (2000-2025)
- ✅ **Price-tier LTR rents:** Realistic estimates based on property value
- ✅ **3 investment strategies:** Cash Flow, Total ROI, Appreciation
- ✅ **Exit strategy analysis:** Sell vs Refinance (80% LTV) with accurate formulas
- ✅ **100% cash purchase option:** Compare cash vs financed approaches
- ✅ **User-adjustable assumptions:** Financing, hold period, expenses, tax bracket
- ✅ **Interactive visualizations:** Plotly charts for data exploration

### Roadmap (Future Phases)

**Phase 3:**
- Expand to Houston, Dallas, San Antonio
- Add more detailed financing scenarios and comparisons
- Include off-market deal sources (wholesalers, foreclosures)
- Property condition assessment (turnkey vs. rehab required)

**Phase 4:**
- Real-time MLS integration
- Automated listing alerts when new opportunities match criteria
- Portfolio optimization (suggest best mix of neighborhoods for diversification)
- Advanced tax strategy modeling (1031 exchanges, cost segregation studies)

---

## 🛠️ Technical Stack

**Data Processing:**
- Python 3.13
- Pandas (data manipulation)
- NumPy (numerical calculations, weighted regression)

**Analysis:**
- Statistical modeling for cash flow projections
- Weighted linear regression for appreciation trends
- Time-series analysis (25 years of data)

**Visualization:**
- Plotly (interactive charts)
- Streamlit (interactive web interface)

**Deployment:**
- Streamlit Cloud (deployed ✅)

---

## 📬 Contact & Collaboration

**Cleburn Walker**  
📧 [cleburn.walker@gmail.com](mailto:cleburn.walker@gmail.com)  
💼 [LinkedIn](https://linkedin.com/in/cleburnwalker)  
💻 [GitHub](https://github.com/cleburn)

### For Investors & Partners:
Interested in scaling this tool to additional markets or exploring commercial partnerships? Let's connect. This MVP demonstrates the potential for a full-featured platform serving real estate investors nationwide.

### For Developers:
Feedback, contributions, and collaboration welcome. This is an active learning project and I'm open to suggestions for improvement or new features.

---

## 📄 License

This project is open source and available for educational and personal use. Data sources retain their original licensing terms.

---

**Last Updated:** November 17, 2025
**Version:** 1.2
**Status:** Production — Deployed to Streamlit Cloud with tax benefit calculations and enhanced analytics