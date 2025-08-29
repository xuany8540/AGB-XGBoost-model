import os
import rasterio
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import shap
import xgboost as xgb
from statsmodels.nonparametric.smoothers_lowess import lowess
from sklearn.model_selection import train_test_split
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ========== 1. 设置输出路径 ==========
save_dir = r".results\shap_plots"
os.makedirs(save_dir, exist_ok=True)

# ========== 2. 读取栅格 ==========
def read_raster(file_path):
    with rasterio.open(file_path) as src:
        data = src.read(1)
        if src.nodata is not None:
            data = np.where(data == src.nodata, np.nan, data)
        else:
            data = np.where(data >= 3.40282e+38, np.nan, data)
        return data

# ========== 3. 主函数 ==========
def main():
    # ----- 3.1 加载15年数据 -----
    years = range(2009, 2024)
    biomass_stack = np.stack([read_raster(f".data/biomass/{y}.tif") for y in years], axis=2)
    pre_stack     = np.stack([read_raster(f".data/Pre/{y}.tif") for y in years], axis=2)
    tem_stack     = np.stack([read_raster(f".data/Tem/{y}.tif") for y in years], axis=2)

    biomass_mean = np.nanmean(biomass_stack, axis=2)
    pre_mean     = np.nanmean(pre_stack, axis=2)
    tem_mean     = np.nanmean(tem_stack, axis=2)

    # ----- 3.2 构建DataFrame -----
    df = pd.DataFrame({
        "biomass": biomass_mean.flatten(),
        "precipitation": pre_mean.flatten(),
        "temperature": tem_mean.flatten()
    }).dropna()

    # ======================== 图①：LOWESS 拟合图 ========================
    sample_df = df.sample(n=1000, random_state=42)

    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=sample_df["temperature"], y=sample_df["biomass"], color='blue', label='Temperature vs Biomass')
    lowess_temp = lowess(sample_df["biomass"], sample_df["temperature"], frac=0.2)
    plt.plot(lowess_temp[:, 0], lowess_temp[:, 1], color='red', label='LOWESS: Temp')

    sns.scatterplot(x=sample_df["precipitation"], y=sample_df["biomass"], color='green', label='Precipitation vs Biomass')
    lowess_pre = lowess(sample_df["biomass"], sample_df["precipitation"], frac=0.2)
    plt.plot(lowess_pre[:, 0], lowess_pre[:, 1], color='orange', label='LOWESS: Precip')

    plt.xlabel("Temperature / Precipitation")
    plt.ylabel("Biomass")
    plt.title("LOWESS: Climate vs Biomass")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "lowess_temp_precip_vs_biomass.png"), dpi=300)
    plt.close()

    # ======================== 图②：SHAP 全局贡献图（不含交互） ========================
    # 特征与标签
    X_basic = df[['temperature', 'precipitation']]
    y = df['biomass']

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X_basic, y, test_size=0.2, random_state=42)

    # 构建基础模型（无交互项）
    model_basic = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        objective="reg:squarederror"
    )
    model_basic.fit(X_train, y_train)

    # SHAP 解释器与计算
    explainer_basic = shap.Explainer(model_basic, X_train)
    shap_values_basic = explainer_basic(X_test)

    # 绘制 SHAP summary dot plot（不包含交互项）
    shap.summary_plot(
        shap_values_basic,
        X_test,
        plot_type="dot",  # 注意：用 dot 替代 bar
        feature_names=X_basic.columns,
        show=False
    )
    plt.title("SHAP Summary Plot (No Interaction Terms)")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "Fig2_SHAP_Summary_NoInteraction.png"), dpi=300)
    plt.close()

    # ======================== 图③：SHAP 交互依赖图（含交互项） ========================
    df['tem_pre_interaction'] = df['temperature'] * df['precipitation']
    X_all = df[['temperature', 'precipitation', 'tem_pre_interaction']]
    y = df['biomass']  # 注意补充 y 的定义

    X_train, X_test, y_train, y_test = train_test_split(X_all, y, test_size=0.2, random_state=42)

    model_interact = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        objective="reg:squarederror"
    )
    model_interact.fit(X_train, y_train)

    # SHAP 解释器与交互值
    explainer_interact = shap.Explainer(model_interact, X_train)
    shap_values_interact = explainer_interact(X_test)

    # 温度主导、降水为交互因子
    shap.dependence_plot(
        'temperature',
        shap_values_interact.values,
        X_test,
        interaction_index='precipitation',
        show=False
    )
    plt.title("SHAP Dependence: Temperature × Precipitation")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "Fig3_SHAP_Dep_Temp_Precip.png"), dpi=300)
    plt.close()

    # 降水主导、温度为交互因子
    shap.dependence_plot(
        'precipitation',
        shap_values_interact.values,
        X_test,
        interaction_index='temperature',
        show=False
    )
    plt.title("SHAP Dependence: Precipitation × Temperature")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "Fig3_SHAP_Dep_Precip_Temp.png"), dpi=300)
    plt.close()


if __name__ == '__main__':
    main()
