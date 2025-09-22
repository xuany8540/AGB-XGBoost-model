import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# ===== 设置路径 =====
output_dir = r".results\XGBoost"
os.makedirs(output_dir, exist_ok=True)

# ===== 1. 读取数据 =====
file_path = r".results\Feature selection\Lasso(VIF).xlsx"
df = pd.read_excel(file_path)

# 特征与目标
target_column = '生物量'
X = df.drop(target_column, axis=1)
y = df[target_column]

# 数据划分
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ===== 2. 建立模型 =====
model = XGBRegressor(
    n_estimators=500,
    learning_rate=0.1,
    max_depth=10,
    subsample=0.9,
    colsample_bytree=0.8,
    random_state=80
)

# 交叉验证
cv_scores = cross_val_score(model, X_train, y_train, cv=10, scoring='r2')

# 模型训练与预测
start_time = time.time()
model.fit(X_train, y_train)
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)
end_time = time.time()

# ===== 特征重要性排序 =====
importance = model.feature_importances_
importance_df = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': importance
}).sort_values(by='Importance', ascending=False)

# 保存特征重要性图
plt.figure(figsize=(10, 7))
sns.barplot(x='Importance', y='Feature', data=importance_df, palette='viridis')
plt.title('XGBoost Feature Importance', fontsize=14)
plt.xlabel('Importance Score')
plt.ylabel('Feature')
plt.tight_layout()
feature_plot_path = os.path.join(output_dir, "XGB_Feature_Importance.png")
plt.savefig(feature_plot_path, dpi=300)
plt.close()

# ===== 模型评估 =====
train_r2 = r2_score(y_train, y_train_pred)
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
train_mse = mean_squared_error(y_train, y_train_pred)
train_aic = len(y_train) * np.log(train_mse) + 2 * (X_train.shape[1] + 1)
train_bic = len(y_train) * np.log(train_mse) + np.log(len(y_train)) * (X_train.shape[1] + 1)

test_r2 = r2_score(y_test, y_test_pred)
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
test_mse = mean_squared_error(y_test, y_test_pred)
test_aic = len(y_test) * np.log(test_mse) + 2 * (X_test.shape[1] + 1)
test_bic = len(y_test) * np.log(test_mse) + np.log(len(y_test)) * (X_test.shape[1] + 1)

elapsed_time = end_time - start_time

# 保存模型评估结果到TXT
eval_path = os.path.join(output_dir, "XGB_Model_Evaluation.txt")
with open(eval_path, "w", encoding="utf-8") as f:
    f.write("===== XGBoost Model Evaluation =====\n")
    f.write(f"Training Time: {elapsed_time:.2f} seconds\n")
    f.write("\n--- Training Set ---\n")
    f.write(f"R²: {train_r2:.4f}\n")
    f.write(f"RMSE: {train_rmse:.4f}\n")
    f.write(f"AIC: {train_aic:.2f}\n")
    f.write(f"BIC: {train_bic:.2f}\n")
    f.write("\n--- Testing Set ---\n")
    f.write(f"R²: {test_r2:.4f}\n")
    f.write(f"RMSE: {test_rmse:.4f}\n")
    f.write(f"AIC: {test_aic:.2f}\n")
    f.write(f"BIC: {test_bic:.2f}\n")
print(f"✅ 模型评估结果已保存至：{eval_path}")

# ===== 全局残差分布图 =====
residuals = y_test - y_test_pred
plt.figure(figsize=(9, 6))
sns.histplot(residuals, kde=True, bins=30, color='skyblue', edgecolor='black')
plt.axvline(0, color='red', linestyle='--', label='Zero Error Line')
plt.title('Distribution of Prediction Residuals')
plt.xlabel('Residuals (Observed - Predicted)')
plt.ylabel('Frequency')
plt.legend()
plt.tight_layout()
resid_plot_path = os.path.join(output_dir, "XGB_Residual_Distribution.png")
plt.savefig(resid_plot_path, dpi=300)
plt.close()

# ===== AGB区间残差箱线图 =====
bins = [0, 50, 100, 150, 200]
labels = ['0–50', '50–100', '100–150', '150–200']

mask = (y_test >= 0) & (y_test <= 200)
y_test_filtered = y_test[mask]
residuals_filtered = residuals[mask]
y_test_bins = pd.cut(y_test_filtered, bins=bins, labels=labels, right=False)

box_data = pd.DataFrame({
    'AGB Bin': y_test_bins,
    'Residuals': residuals_filtered
})

plt.figure(figsize=(9, 6))
sns.boxplot(x='AGB Bin', y='Residuals', data=box_data, palette='pastel')
plt.axhline(0, color='red', linestyle='--', label='Zero Error Line')
plt.title('Prediction Residuals Across AGB Ranges')
plt.xlabel('AGB Range (Mg/ha)')
plt.ylabel('Residuals (Observed - Predicted)')
plt.legend()
plt.tight_layout()
box_plot_path = os.path.join(output_dir, "XGB_Residuals_by_AGB_Bin.png")
plt.savefig(box_plot_path, dpi=300)
plt.close()
