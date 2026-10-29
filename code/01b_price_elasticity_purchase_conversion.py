"""
================================================================================
TIỂU LUẬN: DỰ BÁO ĐỘ CO GIÃN CẦU VÀ DYNAMIC PRICING
================================================================================
Dataset: E-Commerce Pricing Optimization Dataset (Kaggle - Real Data)

PHIÊN BẢN MỚI: Tính Price Elasticity of Purchase Conversion Rate
- Thay vì Quantity → Dùng Purchase Probability (0/1)
- Đo lường: Giá ảnh hưởng đến tỷ lệ chuyển đổi (conversion rate) như thế nào

Tác giả: Business Forecasting Models - UEL
Ngày: 2026-10-13
================================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import f as f_dist
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import classification_report, roc_auc_score, r2_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Thiết lập
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
pd.set_option('display.max_columns', None)

print("="*80)
print("PRICE ELASTICITY OF PURCHASE CONVERSION RATE")
print("="*80)
print("Dataset: E-Commerce Pricing Optimization (Kaggle)")
print("Metrics: Conversion Rate Elasticity")
print("="*80)

# ============================================================================
# BƯỚC 1: LOAD DỮ LIỆU
# ============================================================================

df = pd.read_csv('ecommerce_pricing_data.csv')

print(f"\n📊 Dataset: {len(df):,} transactions")
print(f"   Conversion Rate: {df['Purchase Probability'].mean()*100:.1f}%")

# ============================================================================
# BƯỚC 2: PHÂN TÍCH MỐI QUAN HỆ PRICE - PURCHASE PROBABILITY
# ============================================================================

print("\n" + "="*80)
print("BƯỚC 1: PHÂN TÍCH MỐI QUAN HỆ PRICE - CONVERSION")
print("="*80)

# Tạo price bins để analyze
df['Price_Bin'] = pd.cut(df['Price'], bins=10)

# Tính conversion rate theo price range
conversion_by_price = df.groupby('Price_Bin').agg({
    'Purchase Probability': ['mean', 'count']
}).reset_index()

conversion_by_price.columns = ['Price_Range', 'Conversion_Rate', 'Count']

print("\n📊 CONVERSION RATE THEO KHOẢNG GIÁ:")
print("-"*80)
for idx, row in conversion_by_price.iterrows():
    print(f"   {row['Price_Range']}: {row['Conversion_Rate']*100:5.1f}% ({row['Count']:3.0f} trans)")

# ============================================================================
# BƯỚC 3: VISUALIZATION
# ============================================================================

fig = plt.figure(figsize=(16, 10))

# Plot 1: Price vs Purchase Probability (Scatter)
ax1 = plt.subplot(2, 3, 1)
purchased = df[df['Purchase Probability'] == 1]
not_purchased = df[df['Purchase Probability'] == 0]

plt.scatter(not_purchased['Price'], not_purchased['Purchase Probability'],
           alpha=0.3, s=20, color='red', label='Not Purchased')
plt.scatter(purchased['Price'], purchased['Purchase Probability'],
           alpha=0.3, s=20, color='green', label='Purchased')
plt.xlabel('Price', fontsize=11)
plt.ylabel('Purchase (0/1)', fontsize=11)
plt.title('Price vs Purchase Decision', fontweight='bold', fontsize=12)
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 2: Conversion Rate by Price Bin
ax2 = plt.subplot(2, 3, 2)
price_midpoints = [interval.mid for interval in conversion_by_price['Price_Range']]
plt.plot(price_midpoints, conversion_by_price['Conversion_Rate']*100,
        marker='o', linewidth=2, markersize=8, color='steelblue')
plt.xlabel('Price (midpoint)', fontsize=11)
plt.ylabel('Conversion Rate (%)', fontsize=11)
plt.title('Conversion Rate vs Price', fontweight='bold', fontsize=12)
plt.grid(True, alpha=0.3)

# Plot 3: Price Distribution
ax3 = plt.subplot(2, 3, 3)
plt.hist(df['Price'], bins=30, color='steelblue', edgecolor='black', alpha=0.7)
plt.xlabel('Price', fontsize=11)
plt.ylabel('Frequency', fontsize=11)
plt.title('Price Distribution', fontweight='bold', fontsize=12)
plt.axvline(df['Price'].mean(), color='red', linestyle='--', linewidth=2,
           label=f'Mean: ${df["Price"].mean():.2f}')
plt.legend()
plt.grid(True, alpha=0.3, axis='y')

# Plot 4: Discount vs Conversion
ax4 = plt.subplot(2, 3, 4)
df['Discount_Pct'] = (df['Discount'] / df['Price']) * 100
discount_bins = pd.cut(df['Discount_Pct'], bins=5)
conv_by_discount = df.groupby(discount_bins)['Purchase Probability'].mean()
discount_midpoints = [interval.mid for interval in conv_by_discount.index]

plt.bar(range(len(conv_by_discount)), conv_by_discount.values*100,
       color='green', alpha=0.7, edgecolor='black')
plt.xlabel('Discount % Range', fontsize=11)
plt.ylabel('Conversion Rate (%)', fontsize=11)
plt.title('Conversion Rate vs Discount %', fontweight='bold', fontsize=12)
plt.xticks(range(len(conv_by_discount)),
          [f'{x:.0f}%' for x in discount_midpoints], rotation=45)
plt.grid(True, alpha=0.3, axis='y')

# Plot 5: Conversion by Category
ax5 = plt.subplot(2, 3, 5)
conv_by_category = df.groupby('Product_Category')['Purchase Probability'].mean().sort_values()
plt.barh(range(len(conv_by_category)), conv_by_category.values*100, color='coral', alpha=0.7)
plt.yticks(range(len(conv_by_category)), conv_by_category.index)
plt.xlabel('Conversion Rate (%)', fontsize=11)
plt.title('Conversion Rate by Category', fontweight='bold', fontsize=12)
plt.grid(True, alpha=0.3, axis='x')

# Plot 6: Price by Category (Boxplot)
ax6 = plt.subplot(2, 3, 6)
df.boxplot(column='Price', by='Product_Category', ax=ax6)
plt.xlabel('Category', fontsize=11)
plt.ylabel('Price', fontsize=11)
plt.title('Price Distribution by Category', fontweight='bold', fontsize=12)
plt.suptitle('')  # Remove auto title
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('01b_eda_purchase_conversion.png', dpi=150, bbox_inches='tight')
print("\n✅ Đã lưu: 01b_eda_purchase_conversion.png")

# ============================================================================
# BƯỚC 4: TÍNH PRICE ELASTICITY OF CONVERSION RATE
# ============================================================================

print("\n" + "="*80)
print("BƯỚC 2: TÍNH PRICE ELASTICITY OF CONVERSION RATE")
print("="*80)

# Phương pháp 1: Logistic Regression
print("\n📐 PHƯƠNG PHÁP 1: LOGISTIC REGRESSION")
print("-"*80)
print("   Model: P(Purchase=1) = 1 / (1 + e^-(β₀ + β₁×Price))")
print()

X = df[['Price']].values
y = df['Purchase Probability'].values

# Standardize Price để có coefficient dễ hiểu hơn
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

logreg = LogisticRegression()
logreg.fit(X_scaled, y)

# Coefficient
price_coef = logreg.coef_[0][0]
intercept = logreg.intercept_[0]

y_pred_proba = logreg.predict_proba(X_scaled)[:, 1]
auc = roc_auc_score(y, y_pred_proba)

print(f"   Coefficient (β₁): {price_coef:.4f}")
print(f"   Intercept (β₀): {intercept:.4f}")
print(f"   AUC Score: {auc:.4f}")
print()

# Tính Marginal Effect tại giá trung bình
mean_price = df['Price'].mean()
mean_price_scaled = scaler.transform([[mean_price]])[0][0]

# P(purchase) at mean price
prob_at_mean = 1 / (1 + np.exp(-(intercept + price_coef * mean_price_scaled)))

# Marginal effect = β₁ × P × (1-P) (derivative of logistic function)
marginal_effect = price_coef * prob_at_mean * (1 - prob_at_mean)

# Elasticity = (dP/dPrice) × (Price/P)
elasticity_logistic = marginal_effect * (mean_price / prob_at_mean)

print(f"📊 TẠI GIÁ TRUNG BÌNH (${mean_price:.2f}):")
print(f"   Xác suất mua: {prob_at_mean*100:.2f}%")
print(f"   Marginal Effect: {marginal_effect:.6f}")
print(f"   Elasticity: {elasticity_logistic:.4f}")
print()

# Phương pháp 2: Linear Probability Model (OLS)
print("\n📐 PHƯƠNG PHÁP 2: LINEAR PROBABILITY MODEL (OLS)")
print("-"*80)
print("   Model: P(Purchase) = β₀ + β₁×ln(Price)")
print()

df_ols = df[['Price', 'Purchase Probability']].copy()
df_ols['ln_Price'] = np.log(df_ols['Price'])

X_ols = df_ols[['ln_Price']].values
y_ols = df_ols['Purchase Probability'].values

lm = LinearRegression()
lm.fit(X_ols, y_ols)

beta_1 = lm.coef_[0]
beta_0 = lm.intercept_
r2 = r2_score(y_ols, lm.predict(X_ols))

print(f"   Coefficient (β₁): {beta_1:.4f}")
print(f"   Intercept (β₀): {beta_0:.4f}")
print(f"   R² Score: {r2:.4f}")
print()

print(f"🎯 PRICE ELASTICITY OF CONVERSION RATE: {beta_1:.4f}")
print()

# Phân tích
if beta_1 < 0:
    print("📈 PHÂN TÍCH:")
    print(f"   ✅ Hệ số âm ({beta_1:.4f}) → Giá tăng, conversion giảm (như mong đợi!)")
    print(f"   📊 Ý nghĩa: Tăng giá 1% → Giảm conversion rate ~{abs(beta_1):.2f}%")

    if abs(beta_1) > 1:
        print(f"   🔸 ELASTIC: Conversion rất nhạy cảm với giá")
        print(f"   💡 Chiến lược: GIẢM GIÁ để tăng số lượng khách mua")
    else:
        print(f"   🔸 INELASTIC: Conversion ít nhạy cảm với giá")
        print(f"   💡 Chiến lược: CÓ THỂ TĂNG GIÁ mà không mất quá nhiều khách")
else:
    print("⚠️  BẤT THƯỜNG: Hệ số dương (giá càng cao, conversion càng cao?)")
    print("   → Có thể do: Premium products effect, hoặc confounding factors")

# ============================================================================
# BƯỚC 5: ELASTICITY THEO PHÂN KHÚC SẢN PHẨM
# ============================================================================

print("\n" + "="*80)
print("BƯỚC 3: ELASTICITY THEO CATEGORY")
print("="*80)

categories = df['Product_Category'].unique()
elasticity_by_category = {}

for cat in categories:
    df_cat = df[df['Product_Category'] == cat].copy()

    if len(df_cat) > 30:  # Đủ data
        df_cat['ln_Price'] = np.log(df_cat['Price'])

        X_cat = df_cat[['ln_Price']].values
        y_cat = df_cat['Purchase Probability'].values

        lm_cat = LinearRegression()
        lm_cat.fit(X_cat, y_cat)

        elast = lm_cat.coef_[0]
        elasticity_by_category[cat] = elast

        print(f"\n   {cat}:")
        print(f"      Elasticity: {elast:.4f}")
        print(f"      Avg Price: ${df_cat['Price'].mean():.2f}")
        print(f"      Conversion: {df_cat['Purchase Probability'].mean()*100:.1f}%")
        print(f"      Type: {'Elastic' if abs(elast) > 1 else 'Inelastic'}")

# Visualization Elasticity by Category
if elasticity_by_category:
    plt.figure(figsize=(10, 6))
    categories_list = list(elasticity_by_category.keys())
    elasticities = list(elasticity_by_category.values())

    colors = ['green' if e < -1 else 'orange' if e < 0 else 'red' for e in elasticities]

    plt.barh(categories_list, elasticities, color=colors, alpha=0.7, edgecolor='black')
    plt.xlabel('Price Elasticity of Conversion Rate', fontsize=12)
    plt.ylabel('Product Category', fontsize=12)
    plt.title('Elasticity Comparison by Category', fontweight='bold', fontsize=13)
    plt.axvline(x=-1, color='blue', linestyle='--', linewidth=2, label='|E| = 1')
    plt.axvline(x=0, color='black', linestyle='-', linewidth=1)
    plt.legend()
    plt.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.savefig('01b_elasticity_by_category.png', dpi=150, bbox_inches='tight')
    print("\n✅ Đã lưu: 01b_elasticity_by_category.png")

# ============================================================================
# BƯỚC 5B: ANOVA TEST - SO SÁNH ELASTICITY GIỮA CATEGORIES
# ============================================================================

print("\n" + "="*80)
print("BƯỚC 4: ANOVA F-TEST (Kiểm định sự khác biệt giữa categories)")
print("="*80)

# Phương pháp: Test interaction term Category × ln(Price)
# H0: Tất cả categories có cùng elasticity
# H1: Ít nhất 1 category khác biệt

df_anova = df.copy()
# Thêm cột ln_Price (nếu chưa có)
if 'ln_Price' not in df_anova.columns:
    df_anova['ln_Price'] = np.log(df_anova['Price'])

# Model 1: NULL model (No interaction) - Overall elasticity
X_null = df_anova[['ln_Price']].values
y_anova = df_anova['Purchase Probability'].values

lm_null = LinearRegression()
lm_null.fit(X_null, y_anova)
y_pred_null = lm_null.predict(X_null)
ss_res_null = np.sum((y_anova - y_pred_null)**2)

# Model 2: FULL model (With Category × ln(Price) interaction)
category_dummies = pd.get_dummies(df_anova['Product_Category'], prefix='Cat', drop_first=True)
df_anova = pd.concat([df_anova, category_dummies], axis=1)

# Create interaction terms
interaction_cols = []
for col in category_dummies.columns:
    interaction_col = f'ln_Price_x_{col}'
    df_anova[interaction_col] = df_anova['ln_Price'] * df_anova[col]
    interaction_cols.append(interaction_col)

X_full = df_anova[['ln_Price'] + interaction_cols].values

lm_full = LinearRegression()
lm_full.fit(X_full, y_anova)
y_pred_full = lm_full.predict(X_full)
ss_res_full = np.sum((y_anova - y_pred_full)**2)

# Calculate F-statistic
n = len(df_anova)
k_null = 1  # Only ln(Price)
k_full = 1 + len(interaction_cols)  # ln(Price) + interactions
df_numerator = k_full - k_null  # Number of interaction terms
df_denominator = n - k_full - 1

f_stat = ((ss_res_null - ss_res_full) / df_numerator) / (ss_res_full / df_denominator)

# Calculate p-value
p_value = 1 - f_dist.cdf(f_stat, df_numerator, df_denominator)

print(f"\n📊 ANOVA F-TEST RESULTS:")
print(f"   Null Model SS(Residual): {ss_res_null:.4f}")
print(f"   Full Model SS(Residual): {ss_res_full:.4f}")
print(f"   Improvement: {ss_res_null - ss_res_full:.4f}")
print()
print(f"   F-statistic: F({df_numerator}, {df_denominator}) = {f_stat:.2f}")
print(f"   p-value: {p_value:.4f}")
print()

if p_value < 0.05:
    print(f"   ✅ REJECT H0 (p = {p_value:.4f} < 0.05)")
    print(f"   → CÓ sự khác biệt có ý nghĩa thống kê giữa categories")
else:
    print(f"   ❌ FAIL TO REJECT H0 (p = {p_value:.4f} >= 0.05)")
    print(f"   → KHÔNG có sự khác biệt có ý nghĩa")

# Practical significance check
elast_min = min(elasticities)
elast_max = max(elasticities)
elast_range = abs(elast_max - elast_min)

print(f"\n💡 PRACTICAL SIGNIFICANCE:")
print(f"   Range: {abs(elast_min):.4f} to {abs(elast_max):.4f}")
print(f"   Difference: {elast_range:.4f}")

if elast_range < 0.05:
    print(f"   ⚠️  Range < 0.05 → Practical significance LIMITED")
    print(f"   → All categories remain in {'INELASTIC' if abs(elast_max) < 1 else 'ELASTIC'} range")

# Save ANOVA results
anova_results = pd.DataFrame({
    'Test': ['ANOVA F-Test'],
    'F_Statistic': [f_stat],
    'df_numerator': [df_numerator],
    'df_denominator': [df_denominator],
    'p_value': [p_value],
    'Significant_at_0.05': [p_value < 0.05],
    'Elasticity_Range': [elast_range],
    'Min_Elasticity': [elast_min],
    'Max_Elasticity': [elast_max]
})

anova_results.to_csv('01b_anova_test.csv', index=False)
print(f"\n✅ Đã lưu: 01b_anova_test.csv")

print(f"\n📝 FOR THESIS:")
print(f"   'ANOVA testing confirms {'significant differences' if p_value < 0.05 else 'no significant differences'}")
print(f"    in elasticity across categories (F({df_numerator}, {df_denominator}) = {f_stat:.2f}, p = {p_value:.4f}),")
print(f"    {'though the practical significance is limited given the narrow range' if elast_range < 0.05 else ''}")
print(f"    of variation ({abs(elast_min):.4f} to {abs(elast_max):.4f}).'")

# ============================================================================
# BƯỚC 6: TỔNG KẾT & LƯU KẾT QUẢ
# ============================================================================

print("\n" + "="*80)
print("TỔNG KẾT")
print("="*80)

results = {
    'Metric': [
        'Overall Conversion Rate',
        'Price Elasticity (OLS)',
        'Elasticity Type',
        'AUC Score (Logistic)',
        'R² Score (OLS)',
        'Mean Price',
        'Recommendation'
    ],
    'Value': [
        f"{df['Purchase Probability'].mean()*100:.2f}%",
        f"{beta_1:.4f}",
        'Elastic' if abs(beta_1) > 1 else 'Inelastic',
        f"{auc:.4f}",
        f"{r2:.4f}",
        f"${mean_price:.2f}",
        'Decrease Price' if abs(beta_1) > 1 else 'Can Increase Price'
    ]
}

results_df = pd.DataFrame(results)
results_df.to_csv('01b_elasticity_results.csv', index=False)
print("\n✅ Đã lưu: 01b_elasticity_results.csv")

print("\n📋 KẾT QUẢ CHÍNH:")
print(results_df.to_string(index=False))

# Lưu elasticity by category
if elasticity_by_category:
    cat_df = pd.DataFrame({
        'Category': categories_list,
        'Elasticity': elasticities,
        'Type': ['Elastic' if abs(e) > 1 else 'Inelastic' for e in elasticities]
    })
    cat_df.to_csv('01b_elasticity_by_category.csv', index=False)
    print("\n✅ Đã lưu: 01b_elasticity_by_category.csv")

print("\n" + "="*80)
print("✅ HOÀN THÀNH PHÂN TÍCH PRICE ELASTICITY!")
print("="*80)

print("\n📁 FILES ĐÃ TẠO:")
print("   ✅ 01b_eda_purchase_conversion.png")
print("   ✅ 01b_elasticity_by_category.png")
print("   ✅ 01b_elasticity_results.csv")
print("   ✅ 01b_elasticity_by_category.csv")
print("   ✅ 01b_anova_test.csv")

print("\n🎯 CHO BÁO CÁO TIỂU LUẬN:")
print("   - Metric: Price Elasticity of Purchase Conversion Rate")
print("   - Method: Log-Linear Regression (OLS) + Logistic Regression")
print("   - Finding: Elasticity = {:.4f}".format(beta_1))
print("   - Business Implication: {}".format(
    "Giảm giá sẽ tăng conversion rate đáng kể" if abs(beta_1) > 1
    else "Có thể tăng giá mà không ảnh hưởng nhiều đến conversion"))

print("\n" + "="*80)

plt.show()
