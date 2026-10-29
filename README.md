# Price Elasticity of Demand Forecasting and Dynamic Pricing Model for Online Retail Stores

**Master's Thesis in E-Commerce**  
University of Economics and Law (UEL), Ho Chi Minh City  
Author: Phan Ho Ngoc Han 

Date: October 2026

---

## 📊 Project Overview

This research develops and validates an integrated **dynamic pricing system** for e-commerce that combines:
- **Price elasticity analysis** (econometric approach)
- **Machine learning-based demand forecasting** (Logistic Regression, Random Forest, Gradient Boosting)
- **Mathematical optimization** for profit maximization

Using 2,044 real e-commerce transactions across 4 product categories, we measure price elasticity of **purchase conversion rate** (not traditional quantity elasticity), achieving exceptional ML prediction accuracy (AUC = 0.9999) and identifying significant profit improvement opportunities ($9.9M annually).

---

## 🎯 Key Findings

### 1. Inelastic Demand Pattern
- **Overall elasticity**: -0.1776 (highly inelastic)
- All product categories exhibit |E| < 1
- 1% price increase → only 0.18% decrease in conversion rate
- **Implication**: Price increases can improve profitability

### 2. Exceptional ML Performance
- **Best model**: Logistic Regression
- **AUC**: 0.9999 (near-perfect)
- **Accuracy**: 99.51% (2/409 errors)
- Outperforms complex ensemble methods due to thoughtful feature engineering

### 3. Quality Dominates Price
- **Review_Rating**: 39.6% feature importance
- **All price features combined**: ~32%
- **Insight**: Invest in quality > aggressive pricing

### 4. Profit Improvement Potential
- **Electronics**: +10.8% price → +4.4% profit
- **Clothing**: +8.1% price → +2.7% profit
- **Home & Kitchen**: +11.1% price → +4.6% profit
- **Total annual gain**: $9.9M

### 5. Statistical Analysis
- **ANOVA test**: F(3, 2039) = 1.74, p = 0.157
- No significant elasticity differences across categories
- Supports unified pricing strategy across platform

---

## 📁 Project Structure

```
dynamic_pricing/
├── README.md                                    # This file
├── requirements.txt                              # Python dependencies
│
├── data/
│   └── ecommerce_pricing_data.csv               # Dataset (2,044 transactions)
│
├── code/
│   ├── 01b_price_elasticity_purchase_conversion.py    # Elasticity analysis
│   ├── 02_demand_forecasting_model.py                 # ML models
│   ├── 03_dynamic_pricing_optimization.py             # Profit optimization
│   └── 04_system_architecture_diagram.py              # Architecture diagram
│
├── outputs/
│   ├── figures/                                 # All PNG/PDF visualizations
│   ├── results/                                 # All CSV result files
│   └── models/                                  # Saved ML models (.pkl)
│
├── thesis/
│   ├── Master_Thesis_Complete_APA.docx          # Complete thesis (57 pages)
│   ├── 01_Introduction.docx                     # Section 1
│   ├── 02_Literature_Review.docx                # Section 2
│   ├── 03_Methodology.docx                      # Section 3
│   ├── 04_Results.docx                          # Section 4
│   ├── 05_Discussion.docx                       # Section 5
│   ├── 06_Conclusion.docx                       # Section 6
│   ├── 07_References.docx                       # Section 7 (APA 7th)
│   └── 08_Appendix.docx                         # Section 8
│
└── documentation/
    ├── ANOVA_EXPLANATION.md                     # ANOVA test detailed explanation
    ├── DATASET_INFO.md                          # Dataset information
    ├── THESIS_UPDATE_SUMMARY.md                 # Change log
    └── papers_found.md                          # 19 research papers catalog
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- Virtual environment (recommended)

### Installation

```bash
# Clone repository
git clone [repository-url]
cd dynamic_pricing

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import pandas, numpy, sklearn, matplotlib, seaborn, scipy; print('✅ All libraries installed')"
```

### Run Analysis

```bash
# Step 1: Calculate price elasticity (30 seconds)
python code/01b_price_elasticity_purchase_conversion.py

# Step 2: Train demand forecasting models (45 seconds)
python code/02_demand_forecasting_model.py

# Step 3: Optimize pricing (60 seconds)
python code/03_dynamic_pricing_optimization.py

# Step 4: Generate system architecture diagram (10 seconds)
python code/04_system_architecture_diagram.py

# Total runtime: ~2.5 minutes
```

### Output Files

After running all scripts, you'll have:

**Figures (PNG):**
- `01b_eda_purchase_conversion.png` - Exploratory data analysis (6 charts)
- `01b_elasticity_by_category.png` - Category comparison
- `02_demand_forecasting_results.png` - ML model performance (6 charts)
- `03_dynamic_pricing_optimization.png` - Optimization results (12 charts)
- `04_system_architecture.png` - System architecture diagram

**Results (CSV):**
- `01b_elasticity_results.csv` - Overall elasticity
- `01b_elasticity_by_category.csv` - Category-specific elasticity
- `01b_anova_test.csv` - ANOVA F-test results
- `02_model_comparison.csv` - ML model performance metrics
- `02_feature_importance.csv` - Feature importance rankings
- `03_optimal_prices.csv` - Optimal pricing recommendations

**Models (PKL):**
- `02_best_model.pkl` - Trained Logistic Regression model
- `02_scaler.pkl` - Feature scaler

---

## 📊 Dataset

**Source:** [E-Commerce Pricing Optimization Dataset](https://www.kaggle.com/datasets/zoya77/e-commerce-pricing-optimization-dataset)  
**Platform:** Kaggle  
**Size:** 2,044 transactions × 13 features

### Features

| Column | Type | Description |
|--------|------|-------------|
| Transaction_ID | String | Unique transaction ID |
| Product_ID | String | Unique product ID |
| Product_Category | Categorical | Books, Electronics, Clothing, Home & Kitchen |
| Price | Float | Product price ($10 - $1,000) |
| Discount | Float | Discount amount |
| Customer_Age | Integer | Customer age (18-70) |
| Customer_Gender | Categorical | Male, Female, Other |
| Payment_Method | Categorical | Credit/Debit Card, PayPal, Net Banking |
| Shipping_Type | Categorical | Standard (50.3%), Express (49.7%) |
| Review_Rating | Integer | 1-5 stars (mean: 3.06) |
| **Purchase Probability** | **Binary** | **0 (No) / 1 (Yes) - Target variable** |

---

## 🔬 Methodology

### Phase 1: Price Elasticity Analysis

**Method:** Log-Linear Regression (OLS)

```
P(Purchase) = β₀ + β₁ × ln(Price) + ε
```

**Result:** β₁ = -0.1776 (p < 0.001) → **INELASTIC**

### Phase 2: Demand Forecasting

**Feature Engineering (17 features):**
- Price transformations: `Price_Log`, `Price_Squared`
- Interaction terms: `Price_x_Rating` (critical!)
- Discount features: `Discount_Pct`, `Final_Price`
- Encoded categories: `Category`, `Gender`, `Payment`, `Shipping`

**Models Compared:**
1. Logistic Regression (Winner) - AUC: 0.9999
2. Random Forest - AUC: 0.9962
3. Gradient Boosting - AUC: 0.9968

### Phase 3: Profit Optimization

**Objective Function:**
```python
Profit = (Price - Cost) × Conversion_Rate × Traffic
```

**Method:** Scipy `minimize_scalar` with bounded constraints

**Cost Assumptions:**
- Books: 40%
- Electronics: 60%
- Clothing: 45%
- Home & Kitchen: 50%

---

## 📈 Results Summary

### Elasticity by Category

| Category | Elasticity | Type | Sample Size |
|----------|------------|------|-------------|
| Books | -0.1898 | Inelastic | 513 |
| Home & Kitchen | -0.1813 | Inelastic | 498 |
| Electronics | -0.1754 | Inelastic | 512 |
| Clothing | -0.1653 | Inelastic | 521 |
| **Overall** | **-0.1776** | **Inelastic** | **2,044** |

**ANOVA:** F(3, 2039) = 1.74, p = 0.157 (not significant) → Unified strategy appropriate

### Optimal Pricing Recommendations

| Category | Current Price | Optimal Price | Change | Profit Impact |
|----------|---------------|---------------|--------|---------------|
| Electronics | $503.61 | $558.07 | +10.8% | **+4.4%** |
| Clothing | $511.05 | $552.54 | +8.1% | **+2.7%** |
| Home & Kitchen | $507.46 | $563.59 | +11.1% | **+4.6%** |
| Books | $489.02 | Maintain | 0% | - |

**Total Impact:** $27,120/day = **$9.9M annually** (3 categories)

### Feature Importance (Top 5)

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | Review_Rating | 39.6% |
| 2 | Price_x_Rating | 13.5% |
| 3 | Final_Price | 7.6% |
| 4 | Price_Squared | 7.0% |
| 5 | Price | 6.7% |

---

## 🛠️ Technology Stack

**Language:** Python 3.11

**Core Libraries:**
- `pandas` 2.1+ - Data manipulation
- `numpy` 1.24+ - Numerical computing
- `scikit-learn` 1.3+ - Machine learning
- `scipy` 1.11+ - Optimization
- `matplotlib` 3.7+ - Visualization
- `seaborn` 0.12+ - Statistical plots
- `joblib` 1.3+ - Model persistence

**Development Environment:**
- macOS / Linux / Windows
- Jupyter Notebook (optional for exploration)
- VS Code / PyCharm (recommended IDE)

---

## 📚 Research Contributions

### Theoretical

1. **Conversion Rate Elasticity Framework** - Adapts traditional elasticity for binary outcomes
2. **Quality-Price Interaction Effects** - Quantifies how quality signals moderate price sensitivity

### Methodological

1. **Feature Engineering** - Demonstrates importance of `Price_x_Rating` interaction
2. **Integrated Pipeline** - Combines econometrics + ML + optimization end-to-end

### Practical

1. **Deployable Architecture** - 5-layer system with feedback loops
2. **Quantified Impact** - $9.9M profit improvement with implementation roadmap
3. **Reproducibility** - Complete code, data, and documentation

---
