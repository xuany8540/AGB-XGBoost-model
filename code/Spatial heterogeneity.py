# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
import rasterio
import shap
import xgboost as xgb
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd

np.seterr(divide='ignore', invalid='ignore')

# ---------- 输出路径设置 ----------
output_dir = r".results\Spatial heterogeneity"
os.makedirs(output_dir, exist_ok=True)

def save_figure(name):
    path = os.path.join(output_dir, f"{name}.png")
    plt.savefig(path, dpi=300)
    print(f"[Saved] 图像已保存：{path}")

# ---------- 栅格读取 ----------
def read_raster(file_path):
    with rasterio.open(file_path) as src:
        data = src.read(1)
        if src.nodata is not None:
            data = np.where(data == src.nodata, np.nan, data)
        else:
            data = np.where(data >= 3.40282e+38, np.nan, data)
        return data, src.meta

# ---------- 分区构建 ----------
def define_zones(df, temp_col='temperature', precip_col='precipitation'):
    df = df.copy()
    t_q = pd.qcut(df[temp_col], q=3,
                  labels=['Low Temp','Medium Temp','High Temp'],
                  duplicates='drop')
    p_q = pd.qcut(df[precip_col], q=3,
                  labels=['Low Precip','Medium Precip','High Precip'],
                  duplicates='drop')
    if hasattr(t_q, 'cat') and t_q.cat.categories.size != 3:
        k = t_q.cat.categories.size
        t_q = pd.qcut(df[temp_col], q=k,
                      labels=[f'Temp_{i+1}' for i in range(k)],
                      duplicates='drop')
    if hasattr(p_q, 'cat') and p_q.cat.categories.size != 3:
        k = p_q.cat.categories.size
        p_q = pd.qcut(df[precip_col], q=k,
                      labels=[f'Precip_{i+1}' for i in range(k)],
                      duplicates='drop')
    df['temp_zone'] = t_q
    df['precip_zone'] = p_q
    df['zone'] = df['temp_zone'].astype(str) + " + " + df['precip_zone'].astype(str)
    return df

# ---------- ANOVA + Tukey 输出为表格 ----------
def run_anova_and_tukey_and_collect(sub_df, var_label):
    result_frames = []

    zones = sub_df['zone'].unique()
    if len(zones) < 2:
        print(f"\n[Skip] {var_label}: fewer than 2 zones with n>=2.")
        return None

    try:
        model = ols('shap ~ C(zone)', data=sub_df).fit()
        anova_tbl = sm.stats.anova_lm(model, typ=2)
        ss_between = anova_tbl.loc['C(zone)', 'sum_sq']
        ss_total = ss_between + anova_tbl.loc['Residual', 'sum_sq']
        eta_sq = ss_between / ss_total if ss_total > 0 else np.nan

        anova_tbl = anova_tbl.reset_index().rename(columns={'index': 'Source'})
        anova_tbl['Variable'] = var_label
        anova_tbl['Effect_Size(eta_sq)'] = [eta_sq if s == 'C(zone)' else np.nan for s in anova_tbl['Source']]
        result_frames.append(('{}_ANOVA'.format(var_label.replace(" ", "_")), anova_tbl))

        tukey = pairwise_tukeyhsd(endog=sub_df['shap'], groups=sub_df['zone'], alpha=0.05)
        tukey_df = pd.DataFrame(data=tukey.summary().data[1:], columns=tukey.summary().data[0])
        tukey_df.insert(0, 'Variable', var_label)
        result_frames.append(('{}_Tukey'.format(var_label.replace(" ", "_")), tukey_df))

        return result_frames

    except Exception as e:
        print(f"\n[Error] {var_label}: {e}")
        return None

# ---------- 主分析函数 ----------
def zone_analysis(df, shap_values, feature_names, X_test):
    df_test = df.loc[X_test.index].copy()
    df_test = define_zones(df_test)

    sv = shap_values.values if hasattr(shap_values, "values") else shap_values
    df_test['shap_temp']        = sv[:, 0]
    df_test['shap_precip']      = sv[:, 1]
    df_test['shap_interaction'] = sv[:, 2]

    # ---------- 图1：生物量分区柱状图 ----------
    zone_stats = df_test.groupby('zone', dropna=False).agg(
        mean_biomass=('biomass','mean'),
        std_biomass=('biomass','std'),
        count=('biomass','size')
    ).reset_index()

    plt.figure(figsize=(12, 6))
    sns.barplot(data=zone_stats, x='zone', y='mean_biomass', errorbar=None, palette='viridis')
    plt.xticks(rotation=30, ha='right')
    plt.xlabel('Zone (Temperature + Precipitation)')
    plt.ylabel('Mean Biomass')
    plt.title('Biomass Distribution by Zone')
    plt.tight_layout()
    save_figure("Biomass_Distribution_by_Zone")
    plt.show()

    # ---------- 图2：SHAP敏感性柱状图 ----------
    shap_zone_stats = df_test.groupby('zone', dropna=False).agg(
        shap_temp_mean=('shap_temp','mean'),
        shap_precip_mean=('shap_precip','mean'),
        shap_interaction_mean=('shap_interaction','mean'),
        n=('shap_temp','size')
    ).reset_index()

    plt.figure(figsize=(12, 6))
    mlt = shap_zone_stats.melt(
        id_vars=['zone'],
        value_vars=['shap_temp_mean','shap_precip_mean','shap_interaction_mean'],
        var_name='Variable', value_name='Mean SHAP Value'
    )
    mlt['Variable'] = mlt['Variable'].replace({
        'shap_temp_mean':'Temperature Impact',
        'shap_precip_mean':'Precipitation Impact',
        'shap_interaction_mean':'Interaction Impact'
    })
    sns.barplot(data=mlt, x='zone', y='Mean SHAP Value', hue='Variable', palette='viridis')
    plt.xticks(rotation=30, ha='right')
    plt.xlabel('Zone (Temperature + Precipitation)')
    plt.ylabel('Mean SHAP Value')
    plt.title('SHAP Sensitivity by Zone')
    plt.legend(frameon=False)
    plt.tight_layout()
    save_figure("SHAP_Sensitivity_by_Zone")
    plt.show()

    # ---------- 图3：SHAP箱线图 ----------
    long_df = df_test.melt(
        id_vars='zone',
        value_vars=['shap_temp','shap_precip','shap_interaction'],
        var_name='var', value_name='shap'
    ).dropna(subset=['zone','shap'])
    long_df = long_df[np.isfinite(long_df['shap'])]

    plt.figure(figsize=(12, 3.6))
    for i, v in enumerate(['shap_temp','shap_precip','shap_interaction']):
        ax = plt.subplot(1, 3, i+1)
        sub = long_df[long_df['var']==v]
        if sub.empty:
            ax.set_title(v + " (no data)")
            ax.axis('off')
            continue
        sns.boxplot(data=sub, x='zone', y='shap')
        plt.xticks(rotation=30, ha='right', fontsize=9)
        ax.set_xlabel('Zone')
        ax.set_ylabel('SHAP value')
        ax.set_title(v.replace('shap_','').capitalize())
    plt.tight_layout()
    save_figure("SHAP_Boxplots_By_Zone")
    plt.show()

    # ---------- ANOVA + Tukey 保存 ----------
    counts = long_df.groupby(['var','zone']).size().rename('n').reset_index()
    keep = counts[counts['n'] >= 2][['var','zone']]
    long_df = long_df.merge(keep, on=['var','zone'], how='inner')

    all_results = []
    for var_name, var_label in zip(
        ['shap_temp', 'shap_precip', 'shap_interaction'],
        ['Temperature SHAP', 'Precipitation SHAP', 'Interaction SHAP']
    ):
        sub_df = long_df[long_df['var'] == var_name]
        res = run_anova_and_tukey_and_collect(sub_df, var_label)
        if res is not None:
            all_results.extend(res)

    if all_results:
        out_path = os.path.join(output_dir, "SHAP_Anova_Tukey.xlsx")
        with pd.ExcelWriter(out_path) as writer:
            for sheet_name, df in all_results:
                df.to_excel(writer, index=False, sheet_name=sheet_name[:31])
        print(f"[Saved] ANOVA + Tukey 已保存为：{out_path}")

# ---------- 主程序 ----------
def main():
    years = range(2009, 2024)
    file_paths = {
        'biomass': [f'.data\\biomass\\{y}.tif' for y in years],
        'pre':     [f'.data\\Pre\\{y}.tif'     for y in years],
        'tem':     [f'.data\\Tem\\{y}.tif'     for y in years],
    }

    rasters = {k: np.array([read_raster(fp)[0] for fp in v]).transpose(1,2,0)
               for k, v in file_paths.items()}

    biomass       = rasters['biomass'].mean(axis=2)
    precipitation = rasters['pre'].mean(axis=2)
    temperature   = rasters['tem'].mean(axis=2)

    df = pd.DataFrame({
        'biomass': biomass.flatten(),
        'precipitation': precipitation.flatten(),
        'temperature':   temperature.flatten()
    }).dropna()
    df['temp_precip_interaction'] = df['temperature'] * df['precipitation']

    X = df[['temperature','precipitation','temp_precip_interaction']]
    y = df['biomass']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = xgb.XGBRegressor(
        n_estimators=100, max_depth=3, learning_rate=0.1,
        objective='reg:squarederror', random_state=42
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Test MSE: {mse:.4f}")

    explainer   = shap.Explainer(model, X_train)
    shap_values = explainer(X_test)

    zone_analysis(df, shap_values, ['temperature','precipitation','temp_precip_interaction'], X_test)

if __name__ == "__main__":
    main()
