import numpy as np
import pandas as pd

# 配置路径与阈值
DARK_CSV_PATH = "/data/zm/2026.1.12_testdata/2.5PINN_Result/noise_data/dark_covered.csv"  # 替换为你的暗场数据路径
SAVE_MASK_PATH = "/data/zm/2026.1.12_testdata/3.2NEW_RESULT/Hot_pixel/hot_pixel_mask.npy"
FREQ_THRESHOLD = 5.0  # 频率阈值：大于 10 Hz 认定为固定坏点


def generate_hot_pixel_mask():
    print("正在读取暗场数据...")
    df = pd.read_csv(DARK_CSV_PATH, header=None, names=['row', 'col', 't_in', 't_off'])

    # 限制在实际光路覆盖的 ROI 范围内进行统计
    df = df[(df['row'] >= 400) & (df['row'] <= 499) & (df['col'] <= 767)].copy()

    if len(df) == 0:
        print("警告：ROI 范围内没有检测到任何事件。")
        return

    # 计算总录制时长(秒) - 假设 t_in 的单位是微秒 (us)
    duration_s = (df['t_in'].max() - df['t_in'].min()) / 1e6
    print(f"暗场录制总时长: {duration_s:.2f} 秒")

    # 统计每个坐标的触发次数
    counts = df.groupby(['row', 'col']).size().reset_index(name='count')

    # 计算触发频率 (Hz)
    counts['frequency'] = counts['count'] / duration_s

    # 筛选高频坏点
    hot_pixels = counts[counts['frequency'] > FREQ_THRESHOLD]
    print(f"识别出频率大于 {FREQ_THRESHOLD}Hz 的坏点数量: {len(hot_pixels)}")

    # 生成并保存全局掩膜（依然生成 800x1280 的形状，保证在 Dataset 中直接索引不越界）
    mask = np.zeros((800, 1280), dtype=bool)
    mask[hot_pixels['row'].values, hot_pixels['col'].values] = True

    np.save(SAVE_MASK_PATH, mask)
    print(f"坏点掩膜已保存至: {SAVE_MASK_PATH}")


if __name__ == "__main__":
    generate_hot_pixel_mask()