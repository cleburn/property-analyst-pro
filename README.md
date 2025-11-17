# 🏠 Austin Real Estate Investment Analyzer v1.2

**AI-powered tool that helps real estate investors identify the best neighborhoods in Austin based on cash flow potential and appreciation metrics.**

**Version 1.2** brings critical calculation fixes, robust appreciation modeling using 25 years of data, and comprehensive exit strategy analysis.

---

## 📋 Table of Contents

1. [Try It Now](#-try-it-now)
2. [The Problem](#-the-problem)
3. [What It Does](#-what-it-does)
4. [How It Works](#-how-it-works)
5. [Key Insights](#-key-insights-from-current-data)
6. [Example Visualizations](#-example-visualizations)
7. [App Screenshots](#-app-screenshots)
8. [What's Next](#-whats-next)
9. [Technical Stack](#️-technical-stack)
10. [Project Structure](#-project-structure)
11. [Background](#-background)
12. [Contact & Collaboration](#-contact--collaboration)
13. [License](#-license)

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

3. **Run the web app:**
```bash
   streamlit run app.py
```

4. **Use the tool:**
   - The app will open automatically in your browser at `http://localhost:8501`
   - Adjust your budget range, investment strategy, and rental type in the sidebar
   - Click "Find Best Neighborhoods" to see your top 3 recommendations

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
- Filters 45 Austin neighborhoods with complete data to match your budget
- Ranks them based on your chosen strategy
- Returns the **top 5 neighborhoods** with comprehensive metrics:
  - Monthly cash flow (STR and LTR with realistic rent estimates)
  - Total ROI over your hold period (includes cash flow + equity + appreciation)
  - Historical appreciation rates (weighted 2000-2025 CAGR)
  - Exit strategy analysis (Sell vs Refinance scenarios)
  - Current market position (distance from 2022 peak)
  - Confidence indicators (number of comparable listings)

**Note:** The web app focuses on delivering fast, personalized recommendations. The full exploratory analysis, methodology documentation, and market trend visualizations are available in the `/notebooks` and `/visuals` folders of this repository.
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
- **Refinance scenario:** 75% equity cash-out, new loan amount, new mortgage payment, updated cash flow

### Ranking System
Neighborhoods are ranked by the metric that matters most to your strategy:
- **Cash Flow investors:** Sorted by highest monthly profit (STR or LTR)
- **Total ROI investors:** Sorted by best risk-adjusted returns over hold period
- **Appreciation investors:** Sorted by strongest weighted historical growth (2000-2025 CAGR)

Only neighborhoods with **10+ Airbnb listings** and **complete data** across all sources are included (ensures reliable analysis).

---

## 📊 Key Insights from Current Data (v1.2)

**Austin Market Reality (as of November 2025):**
- **Cash Flow is challenging** — With corrected calculations (2.2% property tax + STR expenses), 0/45 neighborhoods show positive monthly cash flow at 20% down / 7% interest
- **Total ROI remains attractive** — Despite negative monthly cash flow, many neighborhoods show positive 5-year total ROI when appreciation and equity buildup are included
- **LTR more viable than STR** — After fixes, LTR shows better cash flow potential than STR in most neighborhoods
- **Appreciation opportunity exists** — Neighborhoods are 20-30% below their 2022 peaks with median 5.52% weighted CAGR (2000-2025)
- **Cash purchase changes the game** — 100% down payment eliminates negative cash flow in many neighborhoods

**Market Stats:**
- **Median home price:** $541,399
- **Median weighted appreciation:** 5.52% CAGR (2000-2025)
- **Neighborhoods analyzed:** 45 with complete data
- **Data through:** October 2025

**Key Takeaway:** v1.2's corrected calculations reveal that Austin real estate is primarily an **appreciation play** with **Total ROI** as the better strategy vs pure cash flow. Short-term cash flow is difficult at current prices and interest rates, but long-term wealth building through appreciation remains strong.

---

## 📈 Example Visualizations

The tool generates comprehensive visualizations to help investors understand market trends:

### Austin Metro Price Trends (2000-2025)
Shows the complete price trajectory of the Austin metro area, highlighting:
- Pre-pandemic steady growth (2015-2019)
- COVID boom and 2022 peak
- Current market correction and stabilization

### Top 10 Neighborhoods - Cash Flow Comparison
Compares monthly cash flow potential across the highest-performing neighborhoods for both:
- Short-term rentals (STR/Airbnb)
- Long-term rentals (LTR)

### Top 10 Neighborhoods - Appreciation Potential
Ranks neighborhoods by historical growth rates (2015-2019 baseline CAGR), showing which areas have the strongest long-term appreciation track records.

**All charts are automatically saved to the `/visuals` folder when you run the analysis.**

---

## 📸 App Screenshots

### Home Page & Investment Criteria
![Home Page](screenshots/home_page.png)
*The analyzer's main interface - set your budget, strategy, and rental type to get instant recommendations*

### Cash Flow Strategy - Top 3 Neighborhoods (STR)
![Cash Flow STR Results](screenshots/cashflow_str_top3.png)
*Top 3 neighborhoods ranked by short-term rental (Airbnb) cash flow potential*

### Cash Flow Strategy - Long-Term Rentals
![Cash Flow LTR Results](screenshots/cashflow_ltr.png)
*Compare results when switching to long-term rental strategy - shows how recommendations adapt*

### Appreciation Strategy Results
![Appreciation Strategy](screenshots/appreciation.png)
*Switch to Appreciation strategy to see neighborhoods ranked by historical growth rates*

### Detailed Neighborhood Metrics
![Neighborhood Detail Card](screenshots/detail_card.png)
*Each recommended neighborhood includes comprehensive investment metrics: monthly cash flow, appreciation potential, occupancy rates, and confidence indicators*

### Market Analysis Visualizations
![Top Neighborhoods Comparison](screenshots/top_neighborhoods_cashflow.png)
*Built-in visualizations compare cash flow potential across top-performing neighborhoods*

---

## 🚀 What's Next

### Current Version (v1.2 - Complete ✅)
**Major improvements over v1.0:**
- ✅ **Critical calculation fixes:** Property tax (2.2%), STR operating expenses (~$600/mo)
- ✅ **Robust appreciation modeling:** Weighted regression using 25 years of data (2000-2025)
- ✅ **Price-tier LTR rents:** Realistic estimates based on property value
- ✅ **3 investment strategies:** Cash Flow, Total ROI, Appreciation
- ✅ **Exit strategy analysis:** Sell vs Refinance scenarios with tax implications
- ✅ **100% cash purchase option:** Compare cash vs financed approaches
- ✅ **User-adjustable assumptions:** Financing, hold period, expenses
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
- Advanced tax strategy modeling (1031 exchanges, depreciation)

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

## 📁 Project Structure
```
austin-investment-analyzer/
├── app.py                                # Streamlit web interface (v1.2)
├── requirements.txt                      # Python dependencies
├── notebooks/
│   ├── austin_analyzer_v1.2.ipynb       # v1.2 analysis notebook
│   ├── process_v1.2_data.py             # Data processing script
│   ├── scripts/archive/                 # Archived test scripts
│   └── texas-real-estate-analyzer.ipynb # Legacy v1.0 notebook
├── data/
│   ├── raw/                             # Original datasets (gitignored)
│   └── processed/                       # Cleaned data (gitignored)
├── visuals/                             # Generated charts and screenshots
│   ├── screenshots/                     # App screenshots for README
│   ├── austin_metro_price_trends.png
│   ├── top_neighborhoods_cashflow.png
│   └── top_neighborhoods_appreciation.png
├── .gitignore
└── README.md
```

---

## 🎓 Background

This project is part of a 9-month AI Engineer roadmap focused on building practical, income-generating tools at the intersection of real estate investment and data science.

**Why Austin?**
- Strong market fundamentals (top 10 US metro for growth)
- Rich public data availability
- High short-term rental demand (tourism + business travel)
- Personal expertise: 5+ years managing investment properties in Texas

**Methodology (v1.2):**
The analysis approach prioritizes **accurate calculations** and **conservative assumptions**:
- **Corrected operating expenses:** 2.2% property tax (Austin actual), STR expenses ($200 utilities + $300 cleaning + $75 supplies + 3% platform fees)
- **Realistic rent estimates:** Price-tier based LTR rents (0.6-0.75% of value), actual Airbnb data for STR
- **Robust appreciation modeling:** Weighted regression using 25 years of data (2000-2025) with reduced weight for anomaly period (June 2020 - April 2023)
- **User-adjustable assumptions:** Down payment (3.5%-100%), interest rates, hold period, all expenses
- **Real occupancy data:** Not theoretical maximums
- **Conservative projections:** Current market rates (7%), realistic expense ratios

This ensures recommendations are grounded in reality and remain viable across market conditions.

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

**Last Updated:** November 16, 2025
**Version:** 1.2
**Status:** Production — Deployed to Streamlit Cloud with critical fixes and enhanced analytics