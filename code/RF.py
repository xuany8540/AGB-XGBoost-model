import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import time  # ✅ 用于计时
import os

# 设置matplotlib的字体为支持中文的字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体为黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号'-'显示为方块的问题

# 读取Excel文件
file_path = r".results\Feature selection\Lasso(VIF).xlsx"
df = pd.read_excel(file_path)

# 指定目标列（生物量）并分离出特征列
target_column = '生物量'
X = df.drop(target_column, axis=1)
y = df[target_column]

# 数据集按照8:2进行分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 创建随机森林模型
model = RandomForestRegressor(n_estimators=500, random_state=80)

# 进行10折交叉验证
cv_scores = cross_val_score(model, X_train, y_train, cv=10, scoring='r2')

# ✅ 计时开始
start_time = time.time()

# 训练模型
model.fit(X_train, y_train)

# 预测测试集
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# ✅ 计时结束
end_time = time.time()
print(f"\n⏱ 模型训练 + 预测耗时：{end_time - start_time:.4f} 秒")

# 模型评估
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

# 打印评估结果
print("训练集评估结果:")
print(f"R^2: {train_r2:.4f}")
print(f"MSE: {train_mse:.4f}")
print(f"RMSE: {train_rmse:.4f}")
print(f"AIC: {train_aic:.4f}")
print(f"BIC: {train_bic:.4f}")

print("\n测试集评估结果:")
print(f"R^2: {test_r2:.4f}")
print(f"MSE: {test_mse:.4f}")
print(f"RMSE: {test_rmse:.4f}")
print(f"AIC: {test_aic:.4f}")
print(f"BIC: {test_bic:.4f}")
# 散点图：真实值 vs 预测值，并添加y=x辅助线
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_train, y=y_train_pred, label='训练集', alpha=0.5)
sns.scatterplot(x=y_test, y=y_test_pred, label='测试集', alpha=0.5)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2, label='1:1线')  # 添加y=x辅助线，红色
plt.annotate(f'$R^2$: {test_r2:.2f}\nMSE: {test_mse:.2f}\nRMSE: {test_rmse:.2f}',
             xy=(0.85, 0.75), xycoords='axes fraction',  # 坐标位置(0,0)是左下角，(1,1)是右上角
             bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.5, edgecolor='black'))
plt.legend()
plt.xlabel('真实值')
plt.ylabel('预测值')
# 保存散点图
output_dir = r".results\RF"
scatter_output_path = os.path.join(output_dir, "RF_ScatterPlot_True_vs_Predicted.png")
plt.savefig(scatter_output_path, dpi=300, bbox_inches='tight')
print(f"✅ 散点图已保存至：{scatter_output_path}")
plt.show()

# 小提琴图：真实值 vs 预测值
data = pd.DataFrame({'值': np.concatenate([y_test, y_test_pred]), '小提琴图': np.repeat(['真实值', '预测值'], len(y_test))})
plt.figure(figsize=(8, 6))
sns.violinplot(x='小提琴图', y='值', data=data)
# 保存小提琴图
output_dir = r"G:\Github\AGB-XGBoost-model\results"
violin_output_path = os.path.join(output_dir, "RF_ViolinPlot_True_vs_Predicted.png")
plt.savefig(violin_output_path, dpi=300, bbox_inches='tight')
print(f"✅ 小提琴图已保存至：{violin_output_path}")
plt.show()

output_dir = r"G:\Github\AGB-XGBoost-model\results"
os.makedirs(output_dir, exist_ok=True)
output_file = os.path.join(output_dir, "RF_Model_Evaluation.txt")  # ✅ 文件名为随机森林模型结果

evaluation_text = f"""

⏱ 模型训练 + 预测耗时：{end_time - start_time:.4f} 秒

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
📅 项目说明：本模型基于 VIF.xlsx 特征构建，采用 Random Forest 算法，n_estimators=500，使用 10 折交叉验证，训练集与测试集比例为 9:1。

===========================================================
"""

with open(output_file, 'w', encoding='utf-8') as f:
    f.write(evaluation_text)

print(f"\n✅ 模型评估结果已保存为 TXT 文件：\n{output_file}")
