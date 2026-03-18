import os

# 限制底层 C++ 库的线程数，防止多进程评估时死锁卡死
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'

import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import MinkowskiEngine as ME

from dataset import CelexBloodFlowDataset, sequence_sparse_collate
from model import SNN_CNN_Hybrid


def evaluate_model():
    # ================= 1. 基础设置与数据路径 =================
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    BATCH_SIZE = 1  # 测试时 batch_size 设为 1，方便逐个样本提取
    MASK_PATH = "/data/zm/Moshaboli/new_data/other_data/hot_pixel_mask_strict.npy"
    MODEL_WEIGHTS = "/data/zm/Moshaboli/new_data/Model/best_hybrid_model_strictmask.pth"

    # 【可视化核心参数】：统一全局颜色标尺
    # 假设你的流速范围在 0 到 2.0 mm/s 之间，你可以根据实际最大流速修改 vmax
    GLOBAL_VMIN = 0.0
    GLOBAL_VMAX = 2.5

    # 替换为你全新的、网络没见过的泛化测试集路径
    test_config = {
        "/data/zm/Moshaboli/new_data/no3": 0.01973,
    }

    print("正在加载泛化测试集...")
    test_dataset = CelexBloodFlowDataset(data_config=test_config, mask_path=MASK_PATH)
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=sequence_sparse_collate,
        num_workers=0
    )
    print(f"泛化测试集加载完成，共 {len(test_dataset)} 个样本。")

    # ================= 2. 加载训练好的模型 =================
    model = SNN_CNN_Hybrid().to(DEVICE)
    if os.path.exists(MODEL_WEIGHTS):
        model.load_state_dict(torch.load(MODEL_WEIGHTS, map_location=DEVICE))
        print(f"成功加载模型权重: {MODEL_WEIGHTS}")
    else:
        print(f"错误：找不到模型权重文件 {MODEL_WEIGHTS}！")
        return

    model.eval()

    # ================= 3. 指标统计与推理 =================
    total_mse = 0.0
    total_mae = 0.0
    total_mape = 0.0

    # 【核心修改】：创建一个计数器和用于收集多帧的字典
    velocity_counts = {}
    vis_samples = {}

    # 设定参数
    SKIP_FRAMES = 50  # 跳过前 50 帧，避开注射泵/平移台起步的机械不稳定期
    NUM_AVG_FRAMES = 20  # 连续收集 10 帧用来做时间平均 (可以根据需要改为 5 或 20)

    with torch.no_grad():
        for batch_idx, (seq_data, y_true, d_values) in enumerate(test_loader):
            x_seq = []
            for b_coords, b_feats in seq_data:
                sparse_tensor = ME.SparseTensor(features=b_feats.to(DEVICE),
                                                coordinates=b_coords.to(DEVICE))
                x_seq.append(sparse_tensor)

            y_true = y_true.to(DEVICE)
            d_values = d_values.to(DEVICE)

            # 前向传播
            fc_pred = model(x_seq, actual_batch_size=len(y_true))
            d_values_expanded = d_values.view(-1, 1, 1, 1)

            # 得到全场预测图 (此时已经是纯净的 100x368 中心区域)
            v_pred_map = d_values_expanded * fc_pred

            # 直接对整张图求平均
            v_pred_mean = v_pred_map.mean(dim=(1, 2, 3))

            # 计算误差指标（此时使用的是剔除了边缘误差后的纯净平均值）
            mse = torch.nn.functional.mse_loss(v_pred_mean, y_true).item()
            mae = torch.abs(v_pred_mean - y_true).mean().item()
            mape = (torch.abs(v_pred_mean - y_true) / (y_true + 1e-8)).mean().item() * 100

            total_mse += mse
            total_mae += mae
            total_mape += mape

            # 获取当前真实流速作为 Key
            y_true_val = round(y_true[0].item(), 4)

            if y_true_val not in velocity_counts:
                velocity_counts[y_true_val] = 0
                vis_samples[y_true_val] = []  # 改为列表，用于存放多帧

            velocity_counts[y_true_val] += 1

            # 【核心逻辑】：如果在稳定窗口内，则将切片后的热力图存入列表
            if SKIP_FRAMES < velocity_counts[y_true_val] <= (SKIP_FRAMES + NUM_AVG_FRAMES):
                vis_samples[y_true_val].append(v_pred_map[0, 0].cpu().numpy())

            # ================= 4. 预测流速场可视化 (时间平均) =================
        print(f"正在为 {len(vis_samples)} 种不同流速生成经过【时间平均】的热力图...")
        for y_val, maps_list in vis_samples.items():
            if len(maps_list) == 0:
                continue

            plt.figure(figsize=(10, 4))

            # 【核心修改】：对收集到的多帧热力图在第 0 维度（帧维度）求平均！
            # 这一步能从物理上极大地抹平散斑的随机涨落噪声
            avg_map = np.mean(maps_list, axis=0)
            avg_mean_val = np.mean(avg_map)

            # 【核心修改 3】：强制锁死颜色标尺 vmin 和 vmax
            img = plt.imshow(avg_map, cmap='jet', aspect='auto', vmin=GLOBAL_VMIN, vmax=GLOBAL_VMAX)



            cbar = plt.colorbar(img)
            cbar.set_label('Velocity (mm/s)', fontsize=12)

            plt.title(
                f"Averaged Velocity Map ({len(maps_list)} frames)\nGround Truth: {y_val:.3f} mm/s | Predicted Mean: {avg_mean_val:.3f} mm/s",
                fontsize=14)
            plt.xlabel('Width (Pixels)')
            plt.ylabel('Height (Pixels)')

            save_path = f"/data/zm/Moshaboli/new_data/evaluate/tvloss0.005/evaluate_velocity_map_{y_val:.3f}mms_averaged.png"
            plt.tight_layout()
            plt.savefig(save_path, dpi=300)
            plt.close()
            print(f"-> 流速 {y_val:.3f} mm/s 的平滑热力图已保存至: {save_path}")

if __name__ == '__main__':
    evaluate_model()
