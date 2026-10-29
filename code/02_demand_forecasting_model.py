"""
================================================================================
TIỂU LUẬN: DYNAMIC PRICING & PRICE ELASTICITY
================================================================================
BƯỚC 2: DEMAND FORECASTING MODEL
- Dự báo Purchase Probability (Conversion Rate) dựa trên Price và các features
- Sử dụng Machine Learning: Random Forest & XGBoost
- So sánh performance của các models

Tác giả: Business Forecasting Models - UEL
Ngày: 2026-10-13
================================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, classification_report,
                              confusion_matrix, roc_curve)
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# Thiết lập
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
pd.set_option('display.max_columns', None)

print("="*80)
print("DEMAND FORECASTING MODEL - DỰ BÁO CONVERSION RATE")
print("="*80)
print("Dataset: E-Commerce Pricing Optimization (Kaggle)")
print("Target: Purchase Probability (0/1)")
print("="*80)

# ============================================================================
# BƯỚC 1: LOAD DỮ LIỆU
# ============================================================================

print("\n" + "="*80)
print("BƯỚC 1: LOAD VÀ CHUẨN BỊ DỮ LIỆU")
print("="*80)

df = pd.read_csv('ecommerce_pricing_data.csv')

print(f"\n📊 Dataset: {len(df):,} transactions")
print(f"   Conversion Rate: {df['Purchase Probability'].mean()*100:.1f}%")
print(f"   Features: {df.shape[1]} columns")

# ============================================================================
# BƯỚC 2: FEATURE ENGINEERING
# ============================================================================

print("\n" + "="*80)
print("BƯỚC 2: FEATURE ENGINEERING")
print("="*80)

# Tạo các features bổ sung
print("\n🔧 Đang tạo features...")

# 2.1: Price-based features
df['Price_Log'] = np.log(df['Price'])
df['Price_Squared'] = df['Price'] ** 2
df['Discount_Pct'] = (df['Discount'] / df['Price']) * 100
df['Final_Price'] = df['Price'] - df['Discount']

# 2.2: Price Bins (phân nhóm giá)
df['Price_Bin'] = pd.cut(df['Price'], bins=5, labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
df['Price_Bin_Encoded'] = LabelEncoder().fit_transform(df['Price_Bin'])

# 2.3: Discount Bins
df['Discount_Level'] = pd.cut(df['Discount_Pct'], bins=4, labels=['No Discount', 'Low', 'Medium', 'High'])
df['Discount_Level_Encoded'] = LabelEncoder().fit_transform(df['Discount_Level'])

# 2.4: Customer Age Groups
df['Age_Group'] = pd.cut(df['Customer_Age'], bins=[0, 25, 35, 50, 100],
                         labels=['Young', 'Adult', 'Middle', 'Senior'])
df['Age_Group_Encoded'] = LabelEncoder().fit_transform(df['Age_Group'])

# 2.5: Interaction features (quan trọng!)
df['Price_x_Discount'] = df['Price'] * df['Discount_Pct']
df['Price_x_Rating'] = df['Price'] * df['Review_Rating']

# 2.6: Encode categorical variables
le_category = LabelEncoder()
df['Category_Encoded'] = le_category.fit_transform(df['Product_Category'])

le_gender = LabelEncoder()
df['Gender_Encoded'] = le_gender.fit_transform(df['Customer_Gender'])

le_payment = LabelEncoder()
df['Payment_Encoded'] = le_payment.fit_transform(df['Payment_Method'])

le_shipping = LabelEncoder()
df['Shipping_Encoded'] = le_shipping.fit_transform(df['Shipping_Type'])

print("✅ Đã tạo xong features!")

# ============================================================================
# BƯỚC 3: CHỌN FEATURES CHO MODEL
# ============================================================================

print("\n" + "="*80)
print("BƯỚC 3: CHỌN FEATURES CHO MODEL")
print("="*80)

# Danh sách features
feature_cols = [
    # Price features
    'Price', 'Price_Log', 'Price_Squared', 'Final_Price',
    'Price_Bin_Encoded',

    # Discount features
    'Discount', 'Discount_Pct', 'Discount_Level_Encoded',

    # Product features
    'Category_Encoded', 'Review_Rating',

    # Customer features
    'Customer_Age', 'Age_Group_Encoded', 'Gender_Encoded',

    # Transaction features
    'Payment_Encoded', 'Shipping_Encoded',

    # Interaction features
    'Price_x_Discount', 'Price_x_Rating'
]

X = df[feature_cols]
y = df['Purchase Probability']

print(f"\n📋 Features được chọn: {len(feature_cols)}")
for i, col in enumerate(feature_cols, 1):
    print(f"  {i:2d}. {col}")

print(f"\n📊 Shape:")
print(f"   X: {X.shape}")
print(f"   y: {y.shape}")
print(f"   Class distribution: {y.value_counts().to_dict()}")

# ============================================================================
# BƯỚC 4: SPLIT DATA
# ============================================================================

print("\n" + "="*80)
print("BƯỚC 4: CHIA DỮ LIỆU TRAIN/TEST")
print("="*80)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\n📊 Train set: {X_train.shape[0]:,} samples ({X_train.shape[0]/len(df)*100:.1f}%)")
print(f"📊 Test set:  {X_test.shape[0]:,} samples ({X_test.shape[0]/len(df)*100:.1f}%)")

# Chuẩn hóa features (cho Logistic Regression)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ============================================================================
# BƯỚC 5: TRAINING MODELS
# ============================================================================

print("\n" + "="*80)
print("BƯỚC 5: TRAINING MACHINE LEARNING MODELS")
print("="*80)

models = {}
predictions = {}
probabilities = {}

# Model 1: Logistic Regression
print("\n🤖 Model 1: LOGISTIC REGRESSION")
print("-"*80)

lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train_scaled, y_train)
models['Logistic Regression'] = lr

y_pred_lr = lr.predict(X_test_scaled)
y_proba_lr = lr.predict_proba(X_test_scaled)[:, 1]

predictions['Logistic Regression'] = y_pred_lr
probabilities['Logistic Regression'] = y_proba_lr

print(f"  ✅ Training completed!")
print(f"  Accuracy: {accuracy_score(y_test, y_pred_lr):.4f}")
print(f"  AUC Score: {roc_auc_score(y_test, y_proba_lr):.4f}")

# Model 2: Random Forest
print("\n🌲 Model 2: RANDOM FOREST")
print("-"*80)

rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
models['Random Forest'] = rf

y_pred_rf = rf.predict(X_test)
y_proba_rf = rf.predict_proba(X_test)[:, 1]

predictions['Random Forest'] = y_pred_rf
probabilities['Random Forest'] = y_proba_rf

print(f"  ✅ Training completed!")
print(f"  Accuracy: {accuracy_score(y_test, y_pred_rf):.4f}")
print(f"  AUC Score: {roc_auc_score(y_test, y_proba_rf):.4f}")

# Model 3: Gradient Boosting
print("\n🚀 Model 3: GRADIENT BOOSTING")
print("-"*80)

gb = GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42)
gb.fit(X_train, y_train)
models['Gradient Boosting'] = gb

y_pred_gb = gb.predict(X_test)
y_proba_gb = gb.predict_proba(X_test)[:, 1]

predictions['Gradient Boosting'] = y_pred_gb
probabilities['Gradient Boosting'] = y_proba_gb

print(f"  ✅ Training completed!")
print(f"  Accuracy: {accuracy_score(y_test, y_pred_gb):.4f}")
print(f"  AUC Score: {roc_auc_score(y_test, y_proba_gb):.4f}")

# ============================================================================
# BƯỚC 6: SO SÁNH PERFORMANCE
# ============================================================================

print("\n" + "="*80)
print("BƯỚC 6: SO SÁNH PERFORMANCE CÁC MODELS")
print("="*80)

results = []

for model_name in models.keys():
    y_pred = predictions[model_name]
    y_proba = probabilities[model_name]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    results.append({
        'Model': model_name,
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
        'F1-Score': f1,
        'AUC': auc
    })

results_df = pd.DataFrame(results)
results_df = results_df.sort_values('AUC', ascending=False).reset_index(drop=True)

print("\n📊 BẢNG SO SÁNH PERFORMANCE:")
print("-"*80)
print(results_df.to_string(index=False))

# Lưu kết quả
results_df.to_csv('02_model_comparison.csv', index=False)
print("\n✅ Đã lưu: 02_model_comparison.csv")

# Tìm best model
best_model_name = results_df.iloc[0]['Model']
best_model = models[best_model_name]

print(f"\n🏆 BEST MODEL: {best_model_name}")
print(f"   AUC Score: {results_df.iloc[0]['AUC']:.4f}")

# ============================================================================
# BƯỚC 7: FEATURE IMPORTANCE (Random Forest/Gradient Boosting)
# ============================================================================

print("\n" + "="*80)
print("BƯỚC 7: FEATURE IMPORTANCE ANALYSIS")
print("="*80)

# Lấy feature importance từ Random Forest
feature_importance = pd.DataFrame({
    'Feature': feature_cols,
    'Importance': rf.feature_importances_
}).sort_values('Importance', ascending=False)

print("\n📊 TOP 10 FEATURES QUAN TRỌNG NHẤT (Random Forest):")
print("-"*80)
for idx, row in feature_importance.head(10).iterrows():
    print(f"  {row['Feature']:<30} {row['Importance']:.4f}")

feature_importance.to_csv('02_feature_importance.csv', index=False)
print("\n✅ Đã lưu: 02_feature_importance.csv")

# ============================================================================
# BƯỚC 8: VISUALIZATION
# ============================================================================

print("\n" + "="*80)
print("BƯỚC 8: TRỰC QUAN HÓA KẾT QUẢ")
print("="*80)

fig = plt.figure(figsize=(18, 12))

# Plot 1: Model Comparison - AUC Scores
ax1 = plt.subplot(2, 3, 1)
colors = ['gold' if model == best_model_name else 'steelblue' for model in results_df['Model']]
plt.barh(results_df['Model'], results_df['AUC'], color=colors, alpha=0.7, edgecolor='black')
plt.xlabel('AUC Score', fontsize=11)
plt.title('Model Comparison - AUC Scores', fontweight='bold', fontsize=12)
plt.axvline(x=0.5, color='red', linestyle='--', alpha=0.5)
plt.grid(True, alpha=0.3, axis='x')

# Plot 2: Feature Importance
ax2 = plt.subplot(2, 3, 2)
top_features = feature_importance.head(10)
plt.barh(top_features['Feature'], top_features['Importance'], color='coral', alpha=0.7, edgecolor='black')
plt.xlabel('Importance', fontsize=11)
plt.title('Top 10 Feature Importance (Random Forest)', fontweight='bold', fontsize=12)
plt.grid(True, alpha=0.3, axis='x')

# Plot 3: ROC Curves
ax3 = plt.subplot(2, 3, 3)
for model_name, y_proba in probabilities.items():
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc_score = roc_auc_score(y_test, y_proba)
    plt.plot(fpr, tpr, linewidth=2, label=f'{model_name} (AUC={auc_score:.3f})')

plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Guess')
plt.xlabel('False Positive Rate', fontsize=11)
plt.ylabel('True Positive Rate', fontsize=11)
plt.title('ROC Curves Comparison', fontweight='bold', fontsize=12)
plt.legend(fontsize=9)
plt.grid(True, alpha=0.3)

# Plot 4: Confusion Matrix (Best Model)
ax4 = plt.subplot(2, 3, 4)
y_pred_best = predictions[best_model_name]
cm = confusion_matrix(y_test, y_pred_best)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Not Purchase', 'Purchase'],
            yticklabels=['Not Purchase', 'Purchase'])
plt.xlabel('Predicted', fontsize=11)
plt.ylabel('Actual', fontsize=11)
plt.title(f'Confusion Matrix - {best_model_name}', fontweight='bold', fontsize=12)

# Plot 5: Predicted Probability Distribution
ax5 = plt.subplot(2, 3, 5)
y_proba_best = probabilities[best_model_name]
plt.hist(y_proba_best[y_test == 0], bins=30, alpha=0.5, color='red', label='Not Purchased', density=True)
plt.hist(y_proba_best[y_test == 1], bins=30, alpha=0.5, color='green', label='Purchased', density=True)
plt.xlabel('Predicted Probability', fontsize=11)
plt.ylabel('Density', fontsize=11)
plt.title(f'Predicted Probability Distribution - {best_model_name}', fontweight='bold', fontsize=12)
plt.legend()
plt.grid(True, alpha=0.3, axis='y')

# Plot 6: Metrics Comparison
ax6 = plt.subplot(2, 3, 6)
metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC']
x = np.arange(len(models))
width = 0.15

for i, metric in enumerate(metrics_to_plot):
    values = results_df[metric].values
    plt.bar(x + i*width, values, width, label=metric, alpha=0.7)

plt.xlabel('Models', fontsize=11)
plt.ylabel('Score', fontsize=11)
plt.title('All Metrics Comparison', fontweight='bold', fontsize=12)
plt.xticks(x + width*2, results_df['Model'], rotation=15, ha='right')
plt.ylim([0, 1])
plt.legend(fontsize=9)
plt.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('02_demand_forecasting_results.png', dpi=150, bbox_inches='tight')
print("\n✅ Đã lưu: 02_demand_forecasting_results.png")

# ============================================================================
# BƯỚC 9: DETAILED CLASSIFICATION REPORT
# ============================================================================

print("\n" + "="*80)
print(f"BƯỚC 9: CLASSIFICATION REPORT - {best_model_name}")
print("="*80)

y_pred_best = predictions[best_model_name]
print("\n" + classification_report(y_test, y_pred_best,
                                   target_names=['Not Purchase', 'Purchase']))

# ============================================================================
# BƯỚC 10: SAVE BEST MODEL
# ============================================================================

print("\n" + "="*80)
print("BƯỚC 10: LƯU MODEL & PREDICTIONS")
print("="*80)

# Lưu predictions
predictions_df = pd.DataFrame({
    'Actual': y_test.values,
    'Predicted_Probability': probabilities[best_model_name],
    'Predicted_Class': predictions[best_model_name]
})

# Thêm features quan trọng nhất
important_features = feature_importance.head(5)['Feature'].tolist()
for feat in important_features:
    predictions_df[feat] = X_test[feat].values

predictions_df.to_csv('02_predictions.csv', index=False)
print("✅ Đã lưu: 02_predictions.csv")

# Lưu model (sử dụng joblib)
import joblib
joblib.dump(best_model, '02_best_model.pkl')
joblib.dump(scaler, '02_scaler.pkl')
print(f"✅ Đã lưu model: 02_best_model.pkl")
print(f"✅ Đã lưu scaler: 02_scaler.pkl")

# ============================================================================
# TỔNG KẾT
# ============================================================================

print("\n" + "="*80)
print("✅ HOÀN THÀNH DEMAND FORECASTING MODEL!")
print("="*80)

print("\n📁 FILES ĐÃ TẠO:")
print("   ✅ 02_model_comparison.csv - So sánh performance")
print("   ✅ 02_feature_importance.csv - Feature importance")
print("   ✅ 02_demand_forecasting_results.png - Biểu đồ kết quả")
print("   ✅ 02_predictions.csv - Predictions trên test set")
print("   ✅ 02_best_model.pkl - Trained model")
print("   ✅ 02_scaler.pkl - Feature scaler")

print(f"\n🎯 KẾT QUẢ CHÍNH:")
print(f"   Best Model: {best_model_name}")
print(f"   AUC Score: {results_df.iloc[0]['AUC']:.4f}")
print(f"   Accuracy: {results_df.iloc[0]['Accuracy']:.4f}")
print(f"   F1-Score: {results_df.iloc[0]['F1-Score']:.4f}")

print(f"\n🔝 TOP 5 FEATURES:")
for idx, row in feature_importance.head(5).iterrows():
    print(f"   {row['Feature']:<30} {row['Importance']:.4f}")

print("\n🚀 BƯỚC TIẾP THEO:")
print("   → Xây dựng Dynamic Pricing Model sử dụng demand forecast này!")
print("   → Tối ưu hóa giá để maximize profit!")

print("\n" + "="*80)

plt.show()
