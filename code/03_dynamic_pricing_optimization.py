"""
================================================================================
TIỂU LUẬN: DYNAMIC PRICING & PRICE ELASTICITY
================================================================================
BƯỚC 3: DYNAMIC PRICING OPTIMIZATION MODEL
- Sử dụng Demand Forecasting Model để dự đoán conversion tại các mức giá
- Tối ưu hóa giá để maximize Profit = (Price - Cost) × Predicted_Demand
- Phân tích theo Product Category
- Tạo pricing recommendations

Tác giả: Business Forecasting Models - UEL
Ngày: 2026-10-13
================================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import joblib
from scipy.optimize import minimize_scalar, differential_evolution
import warnings
warnings.filterwarnings('ignore')

# Thiết lập
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
pd.set_option('display.max_columns', None)

print("="*80)
print("DYNAMIC PRICING OPTIMIZATION MODEL")
print("="*80)
print("Goal: Maximize Profit = (Price - Cost) × Predicted_Demand")
print("="*80)

# ============================================================================
# BƯỚC 1: LOAD MODEL VÀ DỮ LIỆU
# ============================================================================

print("\n" + "="*80)
print("BƯỚC 1: LOAD TRAINED MODEL & DATA")
print("="*80)

# Load trained model
model = joblib.load('02_best_model.pkl')
scaler = joblib.load('02_scaler.pkl')

print("✅ Đã load model: Logistic Regression")
print("✅ Đã load scaler")

# Load data
df = pd.read_csv('ecommerce_pricing_data.csv')

print(f"\n📊 Dataset: {len(df):,} transactions")
print(f"   Current Avg Price: ${df['Price'].mean():.2f}")
print(f"   Current Conversion: {df['Purchase Probability'].mean()*100:.1f}%")

# ============================================================================
# BƯỚC 2: THIẾT LẬP PARAMETERS
# ============================================================================

print("\n" + "="*80)
print("BƯỚC 2: THIẾT LẬP PARAMETERS")
print("="*80)

# Cost structure (giả định)
# Trong thực tế, cost khác nhau theo category
COST_STRUCTURE = {
    'Books': 0.40,           # 40% of price
    'Electronics': 0.60,     # 60% of price
    'Clothing': 0.45,        # 45% of price
    'Home & Kitchen': 0.50   # 50% of price
}

# Traffic assumptions (số lượng visitors mỗi ngày)
DAILY_TRAFFIC = 1000

# Price bounds (min/max price theo category)
PRICE_BOUNDS = {
    'Books': (10, 300),
    'Electronics': (50, 1000),
    'Clothing': (20, 800),
    'Home & Kitchen': (15, 900)
}

print("\n💰 COST STRUCTURE (% of Price):")
for cat, pct in COST_STRUCTURE.items():
    print(f"   {cat:<20} {pct*100:.0f}%")

print(f"\n👥 DAILY TRAFFIC: {DAILY_TRAFFIC:,} visitors")

print("\n💵 PRICE BOUNDS:")
for cat, bounds in PRICE_BOUNDS.items():
    print(f"   {cat:<20} ${bounds[0]:.0f} - ${bounds[1]:.0f}")

# ============================================================================
# BƯỚC 3: HÀM DỰ ĐOÁN CONVERSION RATE
# ============================================================================

print("\n" + "="*80)
print("BƯỚC 3: TẠO PREDICTION FUNCTION")
print("="*80)

def create_features_for_price(price, category, discount_pct=0.10,
                               customer_age=35, review_rating=3.5,
                               gender='Male', payment='Credit Card',
                               shipping='Standard'):
    """
    Tạo feature vector cho một mức giá cụ thể
    """
    # Encode category
    category_map = {'Books': 0, 'Clothing': 1, 'Electronics': 2, 'Home & Kitchen': 3}
    gender_map = {'Female': 0, 'Male': 1, 'Other': 2}
    payment_map = {'Credit Card': 0, 'Debit Card': 1, 'Net Banking': 2, 'PayPal': 3}
    shipping_map = {'Express': 0, 'Standard': 1}

    # Calculate derived features
    discount = price * discount_pct
    final_price = price - discount
    price_log = np.log(price)
    price_squared = price ** 2

    # Price bins (manual encoding based on training)
    if price < 200:
        price_bin = 0
    elif price < 400:
        price_bin = 1
    elif price < 600:
        price_bin = 2
    elif price < 800:
        price_bin = 3
    else:
        price_bin = 4

    # Discount level
    if discount_pct < 0.05:
        discount_level = 0
    elif discount_pct < 0.15:
        discount_level = 1
    elif discount_pct < 0.25:
        discount_level = 2
    else:
        discount_level = 3

    # Age group
    if customer_age < 25:
        age_group = 0
    elif customer_age < 35:
        age_group = 1
    elif customer_age < 50:
        age_group = 2
    else:
        age_group = 3

    # Interaction features
    price_x_discount = price * discount_pct
    price_x_rating = price * review_rating

    # Feature vector (phải đúng thứ tự như khi training!)
    features = [
        price,                              # Price
        price_log,                          # Price_Log
        price_squared,                      # Price_Squared
        final_price,                        # Final_Price
        price_bin,                          # Price_Bin_Encoded
        discount,                           # Discount
        discount_pct * 100,                 # Discount_Pct
        discount_level,                     # Discount_Level_Encoded
        category_map.get(category, 0),      # Category_Encoded
        review_rating,                      # Review_Rating
        customer_age,                       # Customer_Age
        age_group,                          # Age_Group_Encoded
        gender_map.get(gender, 0),          # Gender_Encoded
        payment_map.get(payment, 0),        # Payment_Encoded
        shipping_map.get(shipping, 0),      # Shipping_Encoded
        price_x_discount,                   # Price_x_Discount
        price_x_rating                      # Price_x_Rating
    ]

    return np.array(features).reshape(1, -1)

def predict_conversion_rate(price, category, **kwargs):
    """
    Dự đoán conversion rate cho một mức giá
    """
    features = create_features_for_price(price, category, **kwargs)
    features_scaled = scaler.transform(features)
    conversion_prob = model.predict_proba(features_scaled)[0][1]
    return conversion_prob

print("\n✅ Đã tạo prediction functions!")

# Test prediction
test_price = 500
test_category = 'Electronics'
test_conversion = predict_conversion_rate(test_price, test_category)
print(f"\n🧪 TEST: Price=${test_price}, Category={test_category}")
print(f"   Predicted Conversion: {test_conversion*100:.2f}%")

# ============================================================================
# BƯỚC 4: HÀM TỐI ƯU HÓA GIÁ
# ============================================================================

print("\n" + "="*80)
print("BƯỚC 4: TẠO OPTIMIZATION FUNCTION")
print("="*80)

def calculate_profit(price, category, traffic=DAILY_TRAFFIC, **kwargs):
    """
    Tính profit tại một mức giá
    Profit = (Price - Cost) × Conversion_Rate × Traffic
    """
    # Cost
    cost_pct = COST_STRUCTURE[category]
    cost = price * cost_pct

    # Predicted conversion rate
    conversion_rate = predict_conversion_rate(price, category, **kwargs)

    # Demand = conversion_rate × traffic
    demand = conversion_rate * traffic

    # Profit
    profit = (price - cost) * demand

    return profit, conversion_rate, demand

def optimize_price(category, traffic=DAILY_TRAFFIC, **kwargs):
    """
    Tìm giá tối ưu để maximize profit cho một category
    """
    min_price, max_price = PRICE_BOUNDS[category]

    # Hàm objective (negative profit vì scipy minimize)
    def objective(price):
        profit, _, _ = calculate_profit(price, category, traffic, **kwargs)
        return -profit  # Negative because we minimize

    # Optimization
    result = minimize_scalar(objective, bounds=(min_price, max_price), method='bounded')

    optimal_price = result.x
    max_profit = -result.fun  # Convert back to positive

    # Get conversion and demand at optimal price
    _, optimal_conversion, optimal_demand = calculate_profit(optimal_price, category, traffic, **kwargs)

    return {
        'category': category,
        'optimal_price': optimal_price,
        'max_profit': max_profit,
        'conversion_rate': optimal_conversion,
        'demand': optimal_demand
    }

print("\n✅ Đã tạo optimization functions!")

# ============================================================================
# BƯỚC 5: TÌM GIÁ TỐI ƯU CHO TỪNG CATEGORY
# ============================================================================

print("\n" + "="*80)
print("BƯỚC 5: TÌM GIÁ TỐI ƯU CHO TỪNG CATEGORY")
print("="*80)

categories = ['Books', 'Electronics', 'Clothing', 'Home & Kitchen']
optimal_results = []

for category in categories:
    print(f"\n🔍 Optimizing for {category}...")

    result = optimize_price(category)
    optimal_results.append(result)

    print(f"   Optimal Price: ${result['optimal_price']:.2f}")
    print(f"   Expected Conversion: {result['conversion_rate']*100:.2f}%")
    print(f"   Expected Demand: {result['demand']:.0f} units/day")
    print(f"   Expected Profit: ${result['max_profit']:.2f}/day")

# Tạo DataFrame
optimal_df = pd.DataFrame(optimal_results)

print("\n" + "="*80)
print("📊 BẢNG GIÁ TỐI ƯU")
print("="*80)
print(optimal_df.to_string(index=False))

optimal_df.to_csv('03_optimal_prices.csv', index=False)
print("\n✅ Đã lưu: 03_optimal_prices.csv")

# ============================================================================
# BƯỚC 6: PHÂN TÍCH SENSITIVITY - GIÁ VS PROFIT
# ============================================================================

print("\n" + "="*80)
print("BƯỚC 6: SENSITIVITY ANALYSIS - PRICE vs PROFIT")
print("="*80)

sensitivity_results = {}

for category in categories:
    print(f"\n📈 Analyzing {category}...")

    min_price, max_price = PRICE_BOUNDS[category]
    prices = np.linspace(min_price, max_price, 100)

    profits = []
    conversions = []
    demands = []
    revenues = []

    for price in prices:
        profit, conversion, demand = calculate_profit(price, category)
        revenue = price * demand

        profits.append(profit)
        conversions.append(conversion * 100)
        demands.append(demand)
        revenues.append(revenue)

    sensitivity_results[category] = {
        'prices': prices,
        'profits': profits,
        'conversions': conversions,
        'demands': demands,
        'revenues': revenues
    }

print("✅ Đã hoàn thành sensitivity analysis!")

# ============================================================================
# BƯỚC 7: SO SÁNH VỚI GIÁ HIỆN TẠI
# ============================================================================

print("\n" + "="*80)
print("BƯỚC 7: SO SÁNH GIÁ TỐI ƯU vs GIÁ HIỆN TẠI")
print("="*80)

comparison_results = []

for category in categories:
    # Current average price
    current_price = df[df['Product_Category'] == category]['Price'].mean()

    # Current performance
    current_profit, current_conversion, current_demand = calculate_profit(current_price, category)

    # Optimal performance
    optimal_row = optimal_df[optimal_df['category'] == category].iloc[0]
    optimal_price = optimal_row['optimal_price']
    optimal_profit = optimal_row['max_profit']

    # Comparison
    price_change = ((optimal_price - current_price) / current_price) * 100
    profit_improvement = ((optimal_profit - current_profit) / current_profit) * 100

    comparison_results.append({
        'Category': category,
        'Current_Price': current_price,
        'Optimal_Price': optimal_price,
        'Price_Change_%': price_change,
        'Current_Profit': current_profit,
        'Optimal_Profit': optimal_profit,
        'Profit_Improvement_%': profit_improvement
    })

    print(f"\n📊 {category}:")
    print(f"   Current Price: ${current_price:.2f}")
    print(f"   Optimal Price: ${optimal_price:.2f} ({price_change:+.1f}%)")
    print(f"   Current Profit: ${current_profit:.2f}/day")
    print(f"   Optimal Profit: ${optimal_profit:.2f}/day")
    print(f"   Profit Improvement: {profit_improvement:+.1f}%")

comparison_df = pd.DataFrame(comparison_results)
comparison_df.to_csv('03_price_comparison.csv', index=False)
print("\n✅ Đã lưu: 03_price_comparison.csv")

# ============================================================================
# BƯỚC 8: VISUALIZATION
# ============================================================================

print("\n" + "="*80)
print("BƯỚC 8: TRỰC QUAN HÓA KẾT QUẢ")
print("="*80)

fig = plt.figure(figsize=(20, 12))

# Plot 1-4: Price vs Profit curves for each category
for idx, category in enumerate(categories, 1):
    ax = plt.subplot(3, 4, idx)

    data = sensitivity_results[category]
    prices = data['prices']
    profits = data['profits']

    # Plot profit curve
    plt.plot(prices, profits, linewidth=2, color='steelblue', label='Profit')

    # Mark optimal point
    optimal_row = optimal_df[optimal_df['category'] == category].iloc[0]
    opt_price = optimal_row['optimal_price']
    opt_profit = optimal_row['max_profit']

    plt.scatter([opt_price], [opt_profit], color='red', s=100, zorder=5,
               label=f'Optimal: ${opt_price:.0f}')

    # Mark current price
    current_price = df[df['Product_Category'] == category]['Price'].mean()
    current_profit, _, _ = calculate_profit(current_price, category)
    plt.scatter([current_price], [current_profit], color='orange', s=100, zorder=5,
               label=f'Current: ${current_price:.0f}')

    plt.xlabel('Price ($)', fontsize=10)
    plt.ylabel('Daily Profit ($)', fontsize=10)
    plt.title(f'{category}', fontweight='bold', fontsize=11)
    plt.legend(fontsize=8)
    plt.grid(True, alpha=0.3)

# Plot 5: Optimal Prices by Category
ax5 = plt.subplot(3, 4, 5)
plt.barh(optimal_df['category'], optimal_df['optimal_price'],
        color='coral', alpha=0.7, edgecolor='black')
plt.xlabel('Optimal Price ($)', fontsize=10)
plt.title('Optimal Prices by Category', fontweight='bold', fontsize=11)
plt.grid(True, alpha=0.3, axis='x')

# Plot 6: Expected Profit by Category
ax6 = plt.subplot(3, 4, 6)
plt.barh(optimal_df['category'], optimal_df['max_profit'],
        color='green', alpha=0.7, edgecolor='black')
plt.xlabel('Expected Daily Profit ($)', fontsize=10)
plt.title('Expected Daily Profit by Category', fontweight='bold', fontsize=11)
plt.grid(True, alpha=0.3, axis='x')

# Plot 7: Conversion Rate at Optimal Price
ax7 = plt.subplot(3, 4, 7)
plt.barh(optimal_df['category'], optimal_df['conversion_rate'] * 100,
        color='purple', alpha=0.7, edgecolor='black')
plt.xlabel('Conversion Rate (%)', fontsize=10)
plt.title('Expected Conversion Rate', fontweight='bold', fontsize=11)
plt.grid(True, alpha=0.3, axis='x')

# Plot 8: Expected Demand
ax8 = plt.subplot(3, 4, 8)
plt.barh(optimal_df['category'], optimal_df['demand'],
        color='orange', alpha=0.7, edgecolor='black')
plt.xlabel('Expected Daily Sales (units)', fontsize=10)
plt.title('Expected Daily Demand', fontweight='bold', fontsize=11)
plt.grid(True, alpha=0.3, axis='x')

# Plot 9: Price Change Recommendation
ax9 = plt.subplot(3, 4, 9)
colors_change = ['green' if x > 0 else 'red' for x in comparison_df['Price_Change_%']]
plt.barh(comparison_df['Category'], comparison_df['Price_Change_%'],
        color=colors_change, alpha=0.7, edgecolor='black')
plt.xlabel('Price Change (%)', fontsize=10)
plt.title('Recommended Price Change', fontweight='bold', fontsize=11)
plt.axvline(x=0, color='black', linestyle='-', linewidth=1)
plt.grid(True, alpha=0.3, axis='x')

# Plot 10: Profit Improvement
ax10 = plt.subplot(3, 4, 10)
plt.barh(comparison_df['Category'], comparison_df['Profit_Improvement_%'],
        color='gold', alpha=0.7, edgecolor='black')
plt.xlabel('Profit Improvement (%)', fontsize=10)
plt.title('Expected Profit Improvement', fontweight='bold', fontsize=11)
plt.grid(True, alpha=0.3, axis='x')

# Plot 11: Current vs Optimal Price
ax11 = plt.subplot(3, 4, 11)
x = np.arange(len(categories))
width = 0.35

plt.bar(x - width/2, comparison_df['Current_Price'], width,
       label='Current', color='lightblue', alpha=0.7, edgecolor='black')
plt.bar(x + width/2, comparison_df['Optimal_Price'], width,
       label='Optimal', color='lightcoral', alpha=0.7, edgecolor='black')

plt.xlabel('Category', fontsize=10)
plt.ylabel('Price ($)', fontsize=10)
plt.title('Current vs Optimal Prices', fontweight='bold', fontsize=11)
plt.xticks(x, categories, rotation=15, ha='right', fontsize=9)
plt.legend()
plt.grid(True, alpha=0.3, axis='y')

# Plot 12: Revenue vs Profit comparison for Books
ax12 = plt.subplot(3, 4, 12)
data_books = sensitivity_results['Books']
plt.plot(data_books['prices'], data_books['revenues'], linewidth=2,
        color='blue', label='Revenue')
plt.plot(data_books['prices'], data_books['profits'], linewidth=2,
        color='green', label='Profit')
plt.xlabel('Price ($)', fontsize=10)
plt.ylabel('Amount ($)', fontsize=10)
plt.title('Revenue vs Profit (Books)', fontweight='bold', fontsize=11)
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('03_dynamic_pricing_optimization.png', dpi=150, bbox_inches='tight')
print("\n✅ Đã lưu: 03_dynamic_pricing_optimization.png")

# ============================================================================
# BƯỚC 9: TẠO PRICING RECOMMENDATIONS
# ============================================================================

print("\n" + "="*80)
print("BƯỚC 9: PRICING RECOMMENDATIONS")
print("="*80)

recommendations = []

for _, row in comparison_df.iterrows():
    category = row['Category']
    price_change = row['Price_Change_%']
    profit_improvement = row['Profit_Improvement_%']

    if price_change > 5:
        action = "TĂNG GIÁ"
        reason = f"Có thể tăng giá {price_change:.1f}% để tăng profit {profit_improvement:.1f}%"
    elif price_change < -5:
        action = "GIẢM GIÁ"
        reason = f"Nên giảm giá {abs(price_change):.1f}% để tăng demand và profit {profit_improvement:.1f}%"
    else:
        action = "GIỮ NGUYÊN"
        reason = "Giá hiện tại gần tối ưu"

    recommendations.append({
        'Category': category,
        'Action': action,
        'Current_Price': f"${row['Current_Price']:.2f}",
        'Recommended_Price': f"${row['Optimal_Price']:.2f}",
        'Expected_Improvement': f"+{profit_improvement:.1f}%",
        'Reason': reason
    })

recommendations_df = pd.DataFrame(recommendations)

print("\n📋 PRICING RECOMMENDATIONS:")
print("="*80)
for _, row in recommendations_df.iterrows():
    print(f"\n🏷️  {row['Category']}:")
    print(f"   Action: {row['Action']}")
    print(f"   {row['Current_Price']} → {row['Recommended_Price']}")
    print(f"   Expected Improvement: {row['Expected_Improvement']}")
    print(f"   Reason: {row['Reason']}")

recommendations_df.to_csv('03_pricing_recommendations.csv', index=False)
print("\n✅ Đã lưu: 03_pricing_recommendations.csv")

# ============================================================================
# TỔNG KẾT
# ============================================================================

print("\n" + "="*80)
print("✅ HOÀN THÀNH DYNAMIC PRICING OPTIMIZATION!")
print("="*80)

print("\n📁 FILES ĐÃ TẠO:")
print("   ✅ 03_optimal_prices.csv - Bảng giá tối ưu")
print("   ✅ 03_price_comparison.csv - So sánh current vs optimal")
print("   ✅ 03_pricing_recommendations.csv - Recommendations")
print("   ✅ 03_dynamic_pricing_optimization.png - 12 biểu đồ phân tích")

total_current_profit = comparison_df['Current_Profit'].sum()
total_optimal_profit = comparison_df['Optimal_Profit'].sum()
total_improvement = ((total_optimal_profit - total_current_profit) / total_current_profit) * 100

print(f"\n💰 TỔNG KẾT PROFIT:")
print(f"   Current Total Profit: ${total_current_profit:.2f}/day")
print(f"   Optimal Total Profit: ${total_optimal_profit:.2f}/day")
print(f"   Total Improvement: +{total_improvement:.1f}%")

print("\n🎯 CHO BÁO CÁO TIỂU LUẬN:")
print("   - Đã optimize giá cho 4 categories")
print("   - Model dự đoán conversion rate với AUC = 0.9999")
print("   - Pricing strategy giúp tăng profit đáng kể")
print("   - Có sensitivity analysis và recommendations cụ thể")

print("\n" + "="*80)

plt.show()
