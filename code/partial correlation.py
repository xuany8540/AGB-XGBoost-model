import rasterio
import numpy as np
from sklearn.linear_model import LinearRegression
from multiprocessing import Pool
import itertools
from scipy.stats import ttest_1samp
import os

# 忽略除以零等运行时警告
np.seterr(divide='ignore', invalid='ignore')

# 定义读取栅格数据的函数，处理 NoData 值
def read_raster(file_path):
    with rasterio.open(file_path) as src:
        data = src.read(1)
        # 将NoData值或3.40282e+38替换为np.nan
        if src.nodata is not None:
            data = np.where(data == src.nodata, np.nan, data)
        else:
            data = np.where(data >= 3.40282e+38, np.nan, data)
        return data, src.meta

# 定义计算偏相关系数的函数
def calculate_partial_correlation(x, y, controls):
    # 检查NaN值
    if np.isnan(x).any() or np.isnan(y).any() or np.isnan(controls).any():
        return np.nan
    # 构建模型并预测残差
    model = LinearRegression().fit(controls, y)
    residuals_y = y - model.predict(controls)
    model = LinearRegression().fit(controls, x)
    residuals_x = x - model.predict(controls)
    # 计算相关系数
    return np.corrcoef(residuals_x, residuals_y)[0, 1]

# worker函数：每个像素的计算
# worker函数：每个像素的计算
def worker(pixel_data):
    i, j, data = pixel_data
    x_biomass = data['biomass']
    y_pre = data['pre']
    y_tem = data['tem']

    # 计算偏相关系数
    controls_tem = np.column_stack([y_tem])
    pre_corr = calculate_partial_correlation(x_biomass, y_pre, controls_tem)
    controls_pre = np.column_stack([y_pre])
    tem_corr = calculate_partial_correlation(x_biomass, y_tem, controls_pre)

    # 初始化T检验结果
    pre_p_value, tem_p_value = np.nan, np.nan

    # 进行单样本T检验：检查有效样本量是否足够
    if np.count_nonzero(~np.isnan(y_pre)) >= 2:  # 有效样本量至少为2
        _, pre_p_value = ttest_1samp(y_pre, 0, nan_policy='omit')
    if np.count_nonzero(~np.isnan(y_tem)) >= 2:  # 有效样本量至少为2
        _, tem_p_value = ttest_1samp(y_tem, 0, nan_policy='omit')

    return i, j, pre_corr, pre_p_value, tem_corr, tem_p_value


# 主函数
def main():
    # 输入文件路径
    years = range(2009, 2024)  # 2009-2023
    file_paths = {
        'biomass': [f'.data\\biomass\\{year}.tif' for year in years],
        'pre': [f'.data\\\Pre\\{year}.tif' for year in years],
        'tem': [f'.data\\Tem\\{year}.tif' for year in years]
    }

    # 读取所有栅格数据，确保数据对齐
    rasters = {
        var: np.array([read_raster(fp)[0] for fp in fps]).transpose(1, 2, 0)
        for var, fps in file_paths.items()
    }
    _, meta = read_raster(file_paths['biomass'][0])

    # 确保栅格数据形状一致
    shape = rasters['biomass'].shape[:2]

    # 组织每个像素数据
    pixel_data = [
        (i, j, {k: v[i, j, :] for k, v in rasters.items()})
        for i, j in itertools.product(range(shape[0]), range(shape[1]))
    ]

    # 初始化结果矩阵
    results = {
        'biomass_pre_corr': np.full(shape, np.nan),
        'biomass_pre_p': np.full(shape, np.nan),
        'biomass_tem_corr': np.full(shape, np.nan),
        'biomass_tem_p': np.full(shape, np.nan)
    }

    # 并行计算
    with Pool() as pool:
        for idx, (i, j, pre_corr, pre_p, tem_corr, tem_p) in enumerate(pool.imap(worker, pixel_data), 1):
            results['biomass_pre_corr'][i, j] = pre_corr
            results['biomass_pre_p'][i, j] = pre_p
            results['biomass_tem_corr'][i, j] = tem_corr
            results['biomass_tem_p'][i, j] = tem_p
            if idx % 1000 == 0:
                print(f'Processed {idx} pixels')

    # 输出结果路径
    results_dir = '.results\\partial correlation analysis'
    os.makedirs(results_dir, exist_ok=True)

    # 保存结果到TIF文件
    meta.update(dtype='float32', count=1, nodata=-9999)
    for var, matrix in results.items():
        output_path = os.path.join(results_dir, f'{var}.tif')
        with rasterio.open(output_path, 'w', **meta) as dst:
            dst.write(np.nan_to_num(matrix, nan=-9999), 1)

if __name__ == '__main__':
    main()
