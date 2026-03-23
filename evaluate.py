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
from scipy import stats  # 用于计算线性拟合的 R 平方

from dataset import CelexBloodFlowDataset, sequence_sparse_collate
from model import SNN_CNN_Hybrid

# --- 全局学术风格与加大字号设置 ---
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.rcParams['mathtext.fontset'] = 'stix'

# 加大字号：统一提升一号
plt.rcParams['font.size'] = 14          # 基础字号
plt.rcParams['axes.labelsize'] = 16     # 坐标轴标签字号
plt.rcParams['xtick.labelsize'] = 14    # x轴刻度字号
plt.rcParams['ytick.labelsize'] = 14    # y轴刻度字号
plt.rcParams['legend.fontsize'] = 12    # 图例字号

# 细节控制
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['axes.linewidth'] = 1.2    # 稍微加粗边框以匹配大字

def evaluate_model():
    # ================= 1. 基础设置与数据路径 =================
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    BATCH_SIZE = 1  # 测试时 batch_size 设为 1，方便逐个样本提取
    MASK_PATH = "/data/zm/Moshaboli/new_data/other_data/hot_pixel_mask_5hz"
    MODEL_WEIGHTS = "/data/zm/Moshaboli/new_data/Model/best_hybrid_model_3ms_0.017.pth"
    SAVE_DIR = "/data/zm/Moshaboli/new_data/evaluate/tvloss0.005/"
    os.makedirs(SAVE_DIR, exist_ok=True)

    # 【可视化核心参数】：统一全局颜色标尺
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
    velocity_counts = {}
    vis_samples = {}
    velocity_predictions = {}  # 【新增】：用于收集稳定期内每个流速的所有预测值

    # 设定参数
    SKIP_FRAMES = 50  # 跳过前 50 帧，避开注射泵/平移台起步的机械不稳定期
    NUM_AVG_FRAMES = 20  # 连续收集 20 帧用来做时间平均

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
            # 直接对整张图求平均得到当前帧的预测流速均值
            v_pred_mean = v_pred_map.mean(dim=(1, 2, 3)).item()

            # 获取当前真实流速作为 Key
            y_true_val = round(y_true[0].item(), 4)

            if y_true_val not in velocity_counts:
                velocity_counts[y_true_val] = 0
                vis_samples[y_true_val] = []
                velocity_predictions[y_true_val] = []  # 初始化空列表

            velocity_counts[y_true_val] += 1

            # 【核心逻辑】：如果在稳定窗口内，收集数据用于热力图和最终误差计算
            if velocity_counts[y_true_val] > SKIP_FRAMES:
                # 收集用于绘图的纯净预测值
                velocity_predictions[y_true_val].append(v_pred_mean)

                # 收集用于平均热力图的图层
                if velocity_counts[y_true_val] <= (SKIP_FRAMES + NUM_AVG_FRAMES):
                    vis_samples[y_true_val].append(v_pred_map[0, 0].cpu().numpy())

    # ================= 4. 预测流速场可视化 (时间平均) =================
    # ================= 4. 预测流速场可视化 (时间平均) =================
    print(f"正在为 {len(vis_samples)} 种不同流速生成学术标准热力图...")
    # ================= 4. 预测流速场可视化 =================
    for y_val, maps_list in vis_samples.items():
        if len(maps_list) == 0: continue

        plt.figure(figsize=(8, 4.5))
        avg_map = np.mean(maps_list, axis=0)
        img = plt.imshow(avg_map, cmap='jet', aspect='auto', vmin=GLOBAL_VMIN, vmax=GLOBAL_VMAX)

        cbar = plt.colorbar(img)
        cbar.set_label('Velocity / mm·s$^{-1}$', fontsize=16)  # 加大字号
        cbar.ax.tick_params(direction='in', labelsize=14)

        plt.xlabel('Width / pixel')
        plt.ylabel('Height / pixel')

        save_path = os.path.join(SAVE_DIR, f"evaluate_velocity_map_{y_val:.3f}mms_academic.png")
        plt.tight_layout()
        plt.savefig(save_path, dpi=1200)  # 最高清晰度
        plt.close()

    # ================= 5. 计算 MAPE 与 MSE 并绘制拟合图与误差曲线 =================
    print("正在计算全局误差并绘制评价曲线...")

    gt_list = []
    pred_mean_list = []
    mape_list = []
    mse_list = []

    # 遍历每个基准流速，计算该流速下的指标
    for gt in sorted(velocity_predictions.keys()):
        preds = np.array(velocity_predictions[gt])
        if len(preds) == 0:
            continue

        gt_list.append(gt)

        # 计算该流速下的平均预测值
        mean_p = np.mean(preds)
        pred_mean_list.append(mean_p)

        # 计算该流速下的 MSE 和 MAPE
        mse = np.mean((preds - gt) ** 2)
        mape = np.mean(np.abs(preds - gt) / (gt + 1e-8)) * 100  # 乘以100转为百分比

        mse_list.append(mse)
        mape_list.append(mape)

        print(f"GT: {gt:.3f} mm/s | Pred: {mean_p:.3f} mm/s | MSE: {mse:.4f} | MAPE: {mape:.2f}%")

    # 打印全局总平均误差
    print("-" * 50)
    print(f"全局平均 MSE: {np.mean(mse_list):.4f}")
    print(f"全局平均 MAPE: {np.mean(mape_list):.2f}%")
    print("-" * 50)

    # --- 绘图 A：线性拟合散点图 ---
    # --- 绘图 A：学术风格线性拟合散点图 ---
    # --- 绘图 A：学术风格线性拟合图 ---
    plt.figure(figsize=(6, 5))
    plt.scatter(gt_list, pred_mean_list, color='#3F51B5', alpha=0.8, s=60, edgecolors='k', linewidths=1.0)

    # 线性拟合
    slope, intercept, r_value, _, _ = stats.linregress(gt_list, pred_mean_list)
    fit_line = [slope * x + intercept for x in gt_list]
    plt.plot(gt_list, fit_line, color='red', linewidth=2.0, label=f'$R^2 = {r_value ** 2:.4f}$')

    plt.xlabel('Ground Truth Velocity / mm·s$^{-1}$')
    plt.ylabel('Predicted Velocity / mm·s$^{-1}$')
    plt.grid(True, linestyle='-', color='#DDDDDD', linewidth=0.6)
    plt.legend(loc='upper left', frameon=True, edgecolor='k')

    max_val = max(max(gt_list), max(pred_mean_list)) * 1.05
    plt.xlim(0, max_val);
    plt.ylim(0, max_val)

    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "evaluate_linear_fit_academic.png"), dpi=1200)
    plt.close()

    # --- 绘图 B：MAPE 误差曲线图 ---
    # --- 绘图 B：学术风格 MAPE 误差分布图 ---
    # --- 绘图 B：学术风格 MAPE 误差分布图 ---
    plt.figure(figsize=(6.5, 4.5))

    # 使用稳重的柱状图
    plt.bar(gt_list, mape_list, width=0.12, color='#546E7A', alpha=0.8, edgecolor='black', linewidth=1.0)

    # 全局平均参考线
    mean_mape = np.mean(mape_list)
    plt.axhline(y=mean_mape, color='red', linestyle='--', linewidth=1.5, label=f'Mean MAPE ({mean_mape:.2f}%)')

    plt.xlabel('Ground Truth Velocity / mm·s$^{-1}$')
    plt.ylabel('MAPE / %')

    # 【关键】固定 Y 轴 0-10%，让误差在视觉上显得非常微小
    plt.ylim(0, 10.0)

    plt.grid(axis='y', linestyle='--', alpha=0.4)
    plt.legend(frameon=False, loc='upper right')

    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "evaluate_mape_academic.png"), dpi=1200)
    plt.close()


if __name__ == '__main__':
    evaluate_model()
