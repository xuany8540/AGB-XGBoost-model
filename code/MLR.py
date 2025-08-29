import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import time
import os

# Output folder
output_dir = r".results\MLR"
os.makedirs(output_dir, exist_ok=True)

# Set matplotlib font to support Chinese if needed
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# Load input Excel file
file_path = r".results\Feature selection\Lasso(VIF).xlsx"
df = pd.read_excel(file_path)

# Split into features and target
target_column = '生物量'  # If needed, rename to 'AGB' in original Excel
X = df.drop(target_column, axis=1)
y = df[target_column]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Cross-validation
model = LinearRegression()
cv_scores = cross_val_score(model, X_train, y_train, cv=10, scoring='r2')

# Timer start
start_time = time.time()

# Fit the model
model.fit(X_train, y_train)

# Predictions
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Timer end
end_time = time.time()
print(f"\n⏱ Training and prediction time: {end_time - start_time:.4f} seconds")

# Evaluation metrics
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

# Print results
print("Training set evaluation:")
print(f"R^2: {train_r2:.4f}")
print(f"MSE: {train_mse:.4f}")
print(f"RMSE: {train_rmse:.4f}")
print(f"AIC: {train_aic:.4f}")
print(f"BIC: {train_bic:.4f}")

print("\nTest set evaluation:")
print(f"R^2: {test_r2:.4f}")
print(f"MSE: {test_mse:.4f}")
print(f"RMSE: {test_rmse:.4f}")
print(f"AIC: {test_aic:.4f}")
print(f"BIC: {test_bic:.4f}")

output_dir = r"results"
os.makedirs(output_dir, exist_ok=True)
output_file = os.path.join(output_dir, "MLR_Model_Evaluation.txt")

evaluation_text = f"""
===================== 模型评估报告（MLR）=====================

📌 训练集评估指标：
- R^2       : {train_r2:.4f}
- MSE       : {train_mse:.4f}
- RMSE      : {train_rmse:.4f}
- AIC       : {train_aic:.4f}
- BIC       : {train_bic:.4f}

📌 测试集评估指标：
- R^2       : {test_r2:.4f}
- MSE       : {test_mse:.4f}
- RMSE      : {test_rmse:.4f}
- AIC       : {test_aic:.4f}
- BIC       : {test_bic:.4f}

📁 文件保存路径：{output_file}
📅 项目说明：本模型基于 VIF.xlsx 特征构建，采用 Multiple Linear Regression，使用 10 折交叉验证，训练集与测试集比例为 8:2。

===========================================================
"""

with open(output_file, 'w', encoding='utf-8') as f:
    f.write(evaluation_text)

print(f"\n✅ 模型评估结果已保存为 TXT 文件：\n{output_file}")

# Scatter plot: Observed vs Predicted
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_train, y=y_train_pred, label='Train', alpha=0.5)
sns.scatterplot(x=y_test, y=y_test_pred, label='Test', alpha=0.5)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2, label='1:1 Line')
plt.annotate(f'$R^2$: {test_r2:.2f}\nMSE: {test_mse:.2f}\nRMSE: {test_rmse:.2f}',
             xy=(0.85, 0.75), xycoords='axes fraction',
             bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.5, edgecolor='black'))
plt.legend()
plt.xlabel('Observed AGB')
plt.ylabel('Predicted AGB')
plt.title('Observed vs Predicted AGB')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'AGB_scatter_MLR.png'))
plt.show()

# Violin plot
data = pd.DataFrame({
    'AGB': np.concatenate([y_test, y_test_pred]),
    'Group': np.repeat(['Observed', 'Predicted'], len(y_test))
})
plt.figure(figsize=(8, 6))
sns.violinplot(x='Group', y='AGB', data=data)
plt.title('Distribution of Observed vs Predicted AGB')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'AGB_violin_MLR.png'))
plt.show()

# Residual plot
residuals = y_test - y_test_pred
plt.figure(figsize=(8, 6))
sns.histplot(residuals, kde=True, bins=30, color='skyblue', edgecolor='black')
plt.axvline(0, color='red', linestyle='--', label='Zero error line')
plt.title('Residual Distribution (Test Set)')
plt.xlabel('Residuals (Observed - Predicted)')
plt.ylabel('Frequency')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'Residuals_hist_MLR.png'))
plt.show()
