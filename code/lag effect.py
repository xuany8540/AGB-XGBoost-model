# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
import rasterio
import statsmodels.api as sm
import matplotlib.pyplot as plt

# ================== 全局参数 ==================
SAMPLE_N = 5000
SEED = 42
P_ADJUST = 'bonferroni'
SAVE_CSV = False
OUT_DIR = r".results\Lag effect analysis"
os.makedirs(OUT_DIR, exist_ok=True)

# ================== 工具函数 ==================
def read_raster_to_array(path):
    with rasterio.open(path) as src:
        arr = src.read(1).astype(np.float32)
    return arr

def tidy_df(arr_x, arr_y, xname, yname, nodata=-3.402823e+38, fillna_mode='drop'):
    df = pd.DataFrame({xname: arr_x.flatten(), yname: arr_y.flatten()})
    df.replace(nodata, np.nan, inplace=True)
    if fillna_mode == 'zero':
        df = df.fillna(0)
    else:
        df = df.dropna()
    return df

def build_results_table(res, terms, p_adjust='bonferroni'):
    out = pd.DataFrame({
        'term': terms,
        'coef': res.params[terms].values,
        'std err': res.bse[terms].values,
        't': res.tvalues[terms].values,
        'p_value': res.pvalues[terms].values
    })
    ci = res.conf_int()
    out['CI_low'] = ci.loc[terms, 0].values
    out['CI_high'] = ci.loc[terms, 1].values

    m = out.shape[0]
    if p_adjust == 'bonferroni':
        out['p_adj'] = np.minimum(out['p_value'] * m, 1.0)
    elif p_adjust == 'fdr_bh':
        order = np.argsort(out['p_value'].values)
        ranks = np.empty_like(order)
        ranks[order] = np.arange(1, m+1)
        out['p_adj'] = np.minimum(out['p_value'] * m / ranks, 1.0)
    else:
        out['p_adj'] = out['p_value']

    def star(p):
        if p < 0.001: return '***'
        if p < 0.01:  return '**'
        if p < 0.05:  return '*'
        return 'ns'
    out['sig'] = out['p_adj'].apply(star)
    return out

def fit_two_tables_and_plot(df, y_col, base_col, var_label,
                            sample_n=SAMPLE_N, seed=SEED,
                            p_adjust=P_ADJUST, save_csv=SAVE_CSV,
                            out_dir=OUT_DIR, keep_fig=True):
    for i in range(1, 11):
        df[f'Lagged_{base_col}_{i}'] = df[base_col].shift(i)
    df2 = df.dropna().copy()
    df_sample = df2.sample(n=sample_n, random_state=seed) if len(df2) > sample_n else df2.copy()

    lag_cols = [f'Lagged_{base_col}_{i}' for i in range(1, 11)]
    X = df_sample[lag_cols].copy()
    y = df_sample[y_col].copy()

    X_unstd = sm.add_constant(X, has_constant='add')
    model_unstd = sm.OLS(y, X_unstd).fit()
    terms_unstd = ['const'] + lag_cols if 'const' in model_unstd.params.index else lag_cols
    table_unstd = build_results_table(model_unstd, terms_unstd, p_adjust=p_adjust)

    print(f"\n[表1] {var_label} — 未标准化 OLS（{p_adjust} 校正）")
    print(table_unstd.to_string(index=False))

    if save_csv:
        out_csv1 = os.path.join(out_dir, f"{var_label}_lag_OLS_unstandardized.csv")
        table_unstd.to_csv(out_csv1, index=False, encoding='utf-8-sig')

    Xz = X.apply(lambda s: (s - s.mean()) / (s.std(ddof=0) if s.std(ddof=0)!=0 else 1.0))
    yz = (y - y.mean()) / (y.std(ddof=0) if y.std(ddof=0)!=0 else 1.0)

    Xz_std = sm.add_constant(Xz, has_constant='add')
    model_std = sm.OLS(yz, Xz_std).fit()
    terms_std = ['const'] + lag_cols if 'const' in model_std.params.index else lag_cols
    table_std = build_results_table(model_std, terms_std, p_adjust=p_adjust)

    print(f"\n[表2] {var_label} — 标准化 OLS（标准化系数 β；{p_adjust} 校正）")
    print(table_std.to_string(index=False))

    if save_csv:
        out_csv2 = os.path.join(out_dir, f"{var_label}_lag_OLS_standardized.csv")
        table_std.to_csv(out_csv2, index=False, encoding='utf-8-sig')

    if keep_fig:
        selected_lags = list(range(1, 11))
        coef_series = model_unstd.params[[f'Lagged_{base_col}_{i}' for i in selected_lags]]

        plt.figure(figsize=(10, 6))
        plt.plot(selected_lags, coef_series, marker='o', linestyle='-', color='b', label='Lagged Response')
        plt.axhline(0, color='gray', linestyle='--', linewidth=1)
        plt.xlabel('Lagged Years', fontsize=12)
        plt.ylabel('Coefficient (Impact on Biomass)', fontsize=12)
        plt.title(f'Lagged Response of {var_label} on Biomass (10 Years)', fontsize=14)
        plt.xticks(selected_lags)
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(out_dir, f"Lag_Effect_{var_label}.png"), dpi=300)
        plt.show()

    return table_unstd, table_std, model_unstd, model_std

# ================== 文件路径 ==================
temperature_path  = r".data\Lag effect analysis\Tem.tif"
precipitation_path= r".data\Lag effect analysis\Pre.tif"
biomass_path      = r".data\Lag effect analysis\生物量.tif"

# ================== 读取数据 ==================
temperature = read_raster_to_array(temperature_path)
precipitation = read_raster_to_array(precipitation_path)
biomass = read_raster_to_array(biomass_path)

# ================== 温度分析 ==================
df_tem = tidy_df(temperature, biomass, xname='temperature', yname='biomass',
                 nodata=-3.402823e+38, fillna_mode='zero')
tem_unstd, tem_std, tem_model_unstd, tem_model_std = fit_two_tables_and_plot(
    df_tem, y_col='biomass', base_col='temperature', var_label='Temperature'
)

# ================== 降水分析 ==================
df_pre = tidy_df(precipitation, biomass, xname='precipitation', yname='biomass',
                 nodata=-3.402823e+38, fillna_mode='drop')
pre_unstd, pre_std, pre_model_unstd, pre_model_std = fit_two_tables_and_plot(
    df_pre, y_col='biomass', base_col='precipitation', var_label='Precipitation'
)

# ================== 保存所有表格为Excel ==================
excel_path = os.path.join(OUT_DIR, "Lag_Effect_Tables.xlsx")
with pd.ExcelWriter(excel_path) as writer:
    tem_unstd.to_excel(writer, sheet_name='Temp_Unstandardized', index=False)
    tem_std.to_excel(writer, sheet_name='Temp_Standardized', index=False)
    pre_unstd.to_excel(writer, sheet_name='Precip_Unstandardized', index=False)
    pre_std.to_excel(writer, sheet_name='Precip_Standardized', index=False)

print(f"\n✅ 所有图表和结果表格已保存至：{OUT_DIR}")
