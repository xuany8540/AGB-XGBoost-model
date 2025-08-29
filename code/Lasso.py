import os
import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
import warnings

# 忽略Lasso收敛警告（可选）
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# 读取数据路径
file_path = r'.data\Sample_data.xlsx'
df = pd.read_excel(file_path)

# 排除 FID 列
if 'FID' in df.columns:
    df = df.drop('FID', axis=1)

# 设置目标列
target_column = '生物量'
X = df.drop(target_column, axis=1)
y = df[target_column]

# 标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Lasso 模型训练
alpha = 0.061
lasso = Lasso(alpha=alpha, max_iter=1000)
lasso.fit(X_scaled, y)

# 筛选特征
coefficients = lasso.coef_
selected_features = X.columns[coefficients != 0].tolist()
print("Lasso特征选择后保留的特征：", selected_features)

# 导出结果
df_selected = df[selected_features + [target_column]]

# 创建保存目录
output_dir = r'.results\Feature selection'
os.makedirs(output_dir, exist_ok=True)

# 写入Excel
output_file_path = os.path.join(output_dir, 'Lasso.xlsx')
df_selected.to_excel(output_file_path, index=False)

print("Lasso特征选择完成，数据已保存至:", output_file_path)
