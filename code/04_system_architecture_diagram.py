"""
================================================================================
TIỂU LUẬN: DYNAMIC PRICING & PRICE ELASTICITY
================================================================================
BƯỚC 4: TẠO SYSTEM ARCHITECTURE DIAGRAM
- Vẽ sơ đồ kiến trúc tổng thể của hệ thống Dynamic Pricing
- Minh họa data flow từ raw data → predictions → optimization
- Sử dụng matplotlib và custom drawing

Tác giả: Business Forecasting Models - UEL
Ngày: 2026-10-13
================================================================================
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle
import matplotlib.lines as mlines

# Thiết lập
plt.rcParams['font.size'] = 10
plt.rcParams['font.family'] = 'sans-serif'

print("="*80)
print("SYSTEM ARCHITECTURE DIAGRAM")
print("="*80)
print("Creating comprehensive architecture diagram...")

# Tạo figure
fig, ax = plt.subplots(1, 1, figsize=(20, 14))
ax.set_xlim(0, 20)
ax.set_ylim(0, 14)
ax.axis('off')

# Colors
COLOR_DATA = '#E8F4F8'      # Light blue - Data
COLOR_PROCESS = '#FFF4E6'   # Light orange - Processing
COLOR_MODEL = '#E8F5E9'     # Light green - Models
COLOR_OUTPUT = '#F3E5F5'    # Light purple - Outputs
COLOR_ARROW = '#424242'     # Dark gray - Arrows

# ============================================================================
# LAYER 1: DATA SOURCES (Top)
# ============================================================================

y_start = 12.5

# Title
ax.text(10, 13.5, 'DYNAMIC PRICING SYSTEM ARCHITECTURE',
        ha='center', va='top', fontsize=18, fontweight='bold')

# Data source boxes
data_sources = [
    {'name': 'E-Commerce\nPricing Data', 'x': 2, 'features': 'Price, Discount,\nCategory, Rating'},
    {'name': 'Customer\nData', 'x': 6, 'features': 'Age, Gender,\nPayment Method'},
    {'name': 'Transaction\nHistory', 'x': 10, 'features': 'Purchase\nProbability'},
    {'name': 'Market\nData', 'x': 14, 'features': 'Competitor Prices,\nTraffic'},
    {'name': 'Cost\nStructure', 'x': 18, 'features': 'Product Costs,\nMargins'}
]

for ds in data_sources:
    # Box
    box = FancyBboxPatch((ds['x']-0.9, y_start-0.8), 1.8, 1.2,
                         boxstyle="round,pad=0.1",
                         facecolor=COLOR_DATA, edgecolor='black', linewidth=2)
    ax.add_patch(box)

    # Text
    ax.text(ds['x'], y_start-0.3, ds['name'],
           ha='center', va='center', fontsize=9, fontweight='bold')
    ax.text(ds['x'], y_start-0.6, ds['features'],
           ha='center', va='center', fontsize=7, style='italic')

# Layer label
ax.text(-0.5, y_start-0.2, 'DATA\nSOURCES',
       ha='center', va='center', fontsize=11, fontweight='bold',
       bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))

# ============================================================================
# LAYER 2: DATA INTEGRATION & PREPROCESSING
# ============================================================================

y_process = 10

# Large processing box
process_box = FancyBboxPatch((1, y_process-0.9), 18, 1.5,
                            boxstyle="round,pad=0.1",
                            facecolor=COLOR_PROCESS, edgecolor='black', linewidth=2)
ax.add_patch(process_box)

ax.text(10, y_process+0.3, 'DATA INTEGRATION & FEATURE ENGINEERING',
       ha='center', va='center', fontsize=12, fontweight='bold')

# Sub-processes
sub_processes = [
    'Data Cleaning',
    'Missing Value\nImputation',
    'Feature Creation\n(Log, Squared, Interaction)',
    'Encoding\n(Label, One-Hot)',
    'Normalization\n(StandardScaler)'
]

x_positions = [3, 6, 10, 14, 17]
for i, (process, x) in enumerate(zip(sub_processes, x_positions)):
    ax.text(x, y_process-0.3, process,
           ha='center', va='center', fontsize=8,
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# Arrows from data sources to processing
for ds in data_sources:
    arrow = FancyArrowPatch((ds['x'], y_start-0.8), (ds['x'], y_process+0.6),
                           arrowstyle='->', mutation_scale=20, linewidth=1.5,
                           color=COLOR_ARROW, alpha=0.6)
    ax.add_patch(arrow)

ax.text(-0.5, y_process, 'DATA\nPROCESSING',
       ha='center', va='center', fontsize=11, fontweight='bold',
       bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))

# ============================================================================
# LAYER 3: ANALYTICS & MODELING
# ============================================================================

y_model = 7

# Three main components
components = [
    {
        'name': 'PRICE ELASTICITY\nANALYSIS',
        'x': 3.5,
        'methods': ['Log-Linear\nRegression', 'Logistic\nRegression'],
        'output': 'Elasticity: -0.18'
    },
    {
        'name': 'DEMAND FORECASTING\nMODEL',
        'x': 10,
        'methods': ['Random Forest', 'Gradient Boosting', 'Logistic Regression'],
        'output': 'AUC: 0.9999'
    },
    {
        'name': 'DYNAMIC PRICING\nOPTIMIZATION',
        'x': 16.5,
        'methods': ['Scipy Optimize', 'Profit Function', 'Constraints'],
        'output': 'Optimal Prices'
    }
]

for comp in components:
    # Main box
    box = FancyBboxPatch((comp['x']-1.3, y_model-1), 2.6, 2.5,
                        boxstyle="round,pad=0.1",
                        facecolor=COLOR_MODEL, edgecolor='black', linewidth=2.5)
    ax.add_patch(box)

    # Title
    ax.text(comp['x'], y_model+1.2, comp['name'],
           ha='center', va='center', fontsize=10, fontweight='bold')

    # Methods
    y_method = y_model+0.6
    for method in comp['methods']:
        ax.text(comp['x'], y_method, f'• {method}',
               ha='center', va='center', fontsize=7)
        y_method -= 0.35

    # Output box
    output_box = FancyBboxPatch((comp['x']-0.9, y_model-0.8), 1.8, 0.4,
                               boxstyle="round,pad=0.05",
                               facecolor='yellow', edgecolor='black', linewidth=1, alpha=0.6)
    ax.add_patch(output_box)
    ax.text(comp['x'], y_model-0.6, comp['output'],
           ha='center', va='center', fontsize=8, fontweight='bold')

# Arrow from processing to models
arrow = FancyArrowPatch((10, y_process-0.9), (10, y_model+1.5),
                       arrowstyle='->', mutation_scale=25, linewidth=2.5,
                       color=COLOR_ARROW)
ax.add_patch(arrow)

# Arrows between model components
arrow1 = FancyArrowPatch((4.8, y_model), (8.7, y_model),
                        arrowstyle='->', mutation_scale=20, linewidth=2,
                        color=COLOR_ARROW)
ax.add_patch(arrow1)
ax.text(6.8, y_model+0.3, 'Feed into', ha='center', fontsize=8, style='italic')

arrow2 = FancyArrowPatch((11.3, y_model), (15.2, y_model),
                        arrowstyle='->', mutation_scale=20, linewidth=2,
                        color=COLOR_ARROW)
ax.add_patch(arrow2)
ax.text(13.2, y_model+0.3, 'Predict &\nOptimize', ha='center', fontsize=8, style='italic')

ax.text(-0.5, y_model+0.5, 'ANALYTICS\n& MODELS',
       ha='center', va='center', fontsize=11, fontweight='bold',
       bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))

# ============================================================================
# LAYER 4: VALIDATION & EVALUATION
# ============================================================================

y_valid = 4

valid_box = FancyBboxPatch((4, y_valid-0.6), 12, 1.2,
                          boxstyle="round,pad=0.1",
                          facecolor='#FFF9E6', edgecolor='black', linewidth=2)
ax.add_patch(valid_box)

ax.text(10, y_valid+0.3, 'MODEL VALIDATION & PERFORMANCE EVALUATION',
       ha='center', va='center', fontsize=11, fontweight='bold')

metrics = [
    'Cross-Validation',
    'AUC Score',
    'Confusion Matrix',
    'Feature Importance',
    'Sensitivity Analysis'
]

x_metrics = [5.5, 7.5, 10, 12.5, 14.5]
for metric, x in zip(metrics, x_metrics):
    ax.text(x, y_valid-0.2, metric, ha='center', va='center', fontsize=7,
           bbox=dict(boxstyle='round', facecolor='white', edgecolor='gray'))

# Arrow from models to validation
arrow = FancyArrowPatch((10, y_model-1), (10, y_valid+0.6),
                       arrowstyle='->', mutation_scale=20, linewidth=2,
                       color=COLOR_ARROW)
ax.add_patch(arrow)

ax.text(-0.5, y_valid, 'VALIDATION',
       ha='center', va='center', fontsize=11, fontweight='bold',
       bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))

# ============================================================================
# LAYER 5: OUTPUTS & RECOMMENDATIONS
# ============================================================================

y_output = 1.5

outputs = [
    {
        'name': 'Elasticity\nReport',
        'x': 3,
        'icon': '📊',
        'details': 'PED by Category'
    },
    {
        'name': 'Demand\nForecast',
        'x': 7,
        'icon': '📈',
        'details': 'Conversion Prediction'
    },
    {
        'name': 'Optimal\nPrices',
        'x': 11,
        'icon': '💰',
        'details': 'Price by Category'
    },
    {
        'name': 'Pricing\nStrategy',
        'x': 15,
        'icon': '🎯',
        'details': 'Recommendations'
    },
    {
        'name': 'Profit\nAnalysis',
        'x': 18,
        'icon': '📉',
        'details': 'ROI & Impact'
    }
]

for out in outputs:
    # Box
    box = FancyBboxPatch((out['x']-0.8, y_output-0.6), 1.6, 1.4,
                        boxstyle="round,pad=0.1",
                        facecolor=COLOR_OUTPUT, edgecolor='black', linewidth=2)
    ax.add_patch(box)

    # Icon
    ax.text(out['x'], y_output+0.5, out['icon'], ha='center', va='center', fontsize=20)

    # Name
    ax.text(out['x'], y_output+0.05, out['name'],
           ha='center', va='center', fontsize=8, fontweight='bold')

    # Details
    ax.text(out['x'], y_output-0.35, out['details'],
           ha='center', va='center', fontsize=6, style='italic')

# Arrows from validation to outputs
arrow = FancyArrowPatch((10, y_valid-0.6), (10, y_output+0.8),
                       arrowstyle='->', mutation_scale=25, linewidth=2.5,
                       color=COLOR_ARROW)
ax.add_patch(arrow)

ax.text(-0.5, y_output+0.2, 'OUTPUTS\n& INSIGHTS',
       ha='center', va='center', fontsize=11, fontweight='bold',
       bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))

# ============================================================================
# FEEDBACK LOOP
# ============================================================================

# Feedback arrow from outputs back to data
feedback_arrow = FancyArrowPatch((18.5, y_output+0.5), (18.5, y_start-0.3),
                                arrowstyle='->', mutation_scale=20, linewidth=2,
                                color='red', linestyle='--', alpha=0.7)
ax.add_patch(feedback_arrow)

ax.text(19.2, y_start/2, 'Continuous\nMonitoring\n& Updates',
       ha='center', va='center', fontsize=8, fontweight='bold',
       color='red', rotation=-90)

# ============================================================================
# LEGEND
# ============================================================================

legend_elements = [
    mpatches.Patch(facecolor=COLOR_DATA, edgecolor='black', label='Data Sources'),
    mpatches.Patch(facecolor=COLOR_PROCESS, edgecolor='black', label='Processing'),
    mpatches.Patch(facecolor=COLOR_MODEL, edgecolor='black', label='Models & Analytics'),
    mpatches.Patch(facecolor=COLOR_OUTPUT, edgecolor='black', label='Outputs'),
    mlines.Line2D([], [], color='red', linestyle='--', linewidth=2, label='Feedback Loop')
]

ax.legend(handles=legend_elements, loc='lower left', fontsize=9,
         framealpha=0.9, edgecolor='black')

# ============================================================================
# FOOTER
# ============================================================================

ax.text(10, 0.2, 'Dynamic Pricing & Price Elasticity System | Business Forecasting Models - UEL | 2026',
       ha='center', va='center', fontsize=9, style='italic', color='gray')

# ============================================================================
# SAVE
# ============================================================================

plt.tight_layout()
plt.savefig('04_system_architecture.png', dpi=300, bbox_inches='tight', facecolor='white')
print("\n✅ Đã lưu: 04_system_architecture.png (300 DPI)")

# Tạo version PDF cho báo cáo
plt.savefig('04_system_architecture.pdf', bbox_inches='tight', facecolor='white')
print("✅ Đã lưu: 04_system_architecture.pdf (High Quality)")

print("\n" + "="*80)
print("✅ HOÀN THÀNH SYSTEM ARCHITECTURE DIAGRAM!")
print("="*80)

print("\n📋 MÔ TẢ CÁC LAYER:")
print("-"*80)
print("1. DATA SOURCES: 5 nguồn dữ liệu chính")
print("2. DATA PROCESSING: Feature engineering & preprocessing")
print("3. ANALYTICS & MODELS:")
print("   - Price Elasticity Analysis (Elasticity = -0.18)")
print("   - Demand Forecasting Model (AUC = 0.9999)")
print("   - Dynamic Pricing Optimization (Optimal Prices)")
print("4. VALIDATION: Cross-validation & performance metrics")
print("5. OUTPUTS: Reports, forecasts, recommendations")
print("6. FEEDBACK LOOP: Continuous improvement")

print("\n🎯 CHO BÁO CÁO:")
print("   → Insert diagram vào Section 3: METHODOLOGY")
print("   → Giải thích từng layer trong text")
print("   → Reference diagram khi mô tả data flow")

print("\n" + "="*80)

plt.show()
