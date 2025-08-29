import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor

# 读取Excel文件
file_path = r'.data\Sample_data.xlsx'
df = pd.read_excel(file_path)

# 排除第一列序号，假设序号列名为'Index'或类似名称
# 如果您的序号列名不是'Index'，请替换为实际的列名
if 'FID' in df.columns:
    df = df.drop('FID', axis=1)

# 指定目标列（生物量）并分离出特征列
target_column = '生物量'
features = df.columns.drop(target_column)

# 计算VIF
vif_data = pd.DataFrame()
vif_data["feature"] = features
vif_data["VIF"] = [variance_inflation_factor(df[features].values, i) for i in range(len(features))]

# 显示VIF值
print("VIF值：")
print(vif_data)

# 根据VIF值进行特征选择，通常VIF值大于5或10认为是有问题的
vif_threshold = 10
selected_features = vif_data[vif_data["VIF"] <= vif_threshold]["feature"].tolist()

# 打印选中的特征
print("根据VIF特征选择后保留的特征：", selected_features)

# 可以选择保存选中的特征列和目标列到新的DataFrame
df_selected = df[selected_features + [target_column]]

# 保存处理后的数据到新的Excel文件
output_file_path = r'.results\Feature selection'
df_selected.to_excel(output_file_path, index=False)

print("VIF特征选择完成，数据已保存至:", output_file_path)