import rasterio
import numpy as np
import pandas as pd
import os

# 忽略运行时除以零的警告
np.seterr(divide='ignore', invalid='ignore')

# 输出目录
output_dir = r".results\Spatial heterogeneity"
os.makedirs(output_dir, exist_ok=True)

# 读取栅格文件并处理 NoData 值
def read_raster(file_path):
    with rasterio.open(file_path) as src:
        data = src.read(1)
        if src.nodata is not None:
            data = np.where(data == src.nodata, np.nan, data)
        return data

# 主函数
def main():
    # 输入年份范围和文件路径
    years = range(2009, 2024)
    file_paths = {
        'pre': [f'.data\\Pre\\{year}.tif' for year in years],
        'tem': [f'.data\\Tem\\{year}.tif' for year in years]
    }

    # 批量读取栅格数据
    rasters = {
        var: np.array([read_raster(fp) for fp in fps]).transpose(1, 2, 0)
        for var, fps in file_paths.items()
    }

    # 多年平均
    mean_precipitation = np.nanmean(rasters['pre'], axis=2)
    mean_temperature = np.nanmean(rasters['tem'], axis=2)

    # 创建有效值掩码
    valid_mask = ~np.isnan(mean_precipitation) & ~np.isnan(mean_temperature)
    valid_precip = mean_precipitation[valid_mask]
    valid_temp = mean_temperature[valid_mask]

    # 计算分位数
    temp_bins = np.percentile(valid_temp, [33, 66])
    precip_bins = np.percentile(valid_precip, [33, 66])

    print(f"Temperature bins (分位数范围): {temp_bins}")
    print(f"Precipitation bins (分位数范围): {precip_bins}")

    # 使用 digitize 分区（0=低，1=中，2=高）
    temp_class = np.digitize(mean_temperature, temp_bins)
    precip_class = np.digitize(mean_precipitation, precip_bins)

    # 分区标签定义
    temp_labels = ["Low Tem", "Medium Tem", "High Tem"]
    precip_labels = ["Low Pre", "Medium Pre", "High Pre"]

    # 获取每个分区的范围字符串
    def get_range(value, bins, kind):
        if kind == 'low':
            return f"< {bins[0]:.2f}"
        elif kind == 'medium':
            return f"{bins[0]:.2f} ≤ x ≤ {bins[1]:.2f}"
        elif kind == 'high':
            return f"> {bins[1]:.2f}"

    # 构建结果表
    rows = []
    for t_idx in range(3):
        for p_idx in range(3):
            zone_mask = (temp_class == t_idx) & (precip_class == p_idx)
            count = int(np.sum(zone_mask))

            zone_name = f"{temp_labels[t_idx]} + {precip_labels[p_idx]}"

            temp_range = get_range(None, temp_bins, ['low','medium','high'][t_idx])
            precip_range = get_range(None, precip_bins, ['low','medium','high'][p_idx])

            rows.append({
                "Zone Name": zone_name,
                "Annual Mean Temperature Range（°C）": f"Tem {temp_range}",
                "Annual Mean Precipitation Range（mm）": f"Pre {precip_range}",
                "Number of Pixels": count
            })

    df = pd.DataFrame(rows)
    print("\n=== 分区统计表 ===")
    print(df)

    # 保存为 Excel
    output_file = os.path.join(output_dir, "Zone_TemPre_Stats.xlsx")
    df.to_excel(output_file, index=False)
    print(f"\n[Saved] 表格已保存到：{output_file}")

if __name__ == "__main__":
    main()
