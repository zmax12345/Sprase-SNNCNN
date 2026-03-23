import os
import matplotlib.pyplot as plt
# ==============================================================================
# 【防卡死且提速的核心设置 1】：严格限制底层 C++ 库的线程数
# 必须在 import torch 和 MinkowskiEngine 之前设置！
# 这意味着在每个数据加载子进程中，ME 只用 1 个线程，把并发任务完全交给 PyTorch 的多进程
# ==============================================================================
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'

import torch
import torch.nn.functional as F
import MinkowskiEngine as ME
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import CelexBloodFlowDataset, sequence_sparse_collate
from model import SNN_CNN_Hybrid


def total_variation_loss(img):
    """
    计算图像的全变分损失 (TV Loss)，用于平滑相邻像素。
    img shape: [Batch, Channel, Height, Width]
    """
    # 计算高度方向相邻像素的绝对差值均值
    tv_h = torch.mean(torch.abs(img[:, :, 1:, :] - img[:, :, :-1, :]))
    # 计算宽度方向相邻像素的绝对差值均值
    tv_w = torch.mean(torch.abs(img[:, :, :, 1:] - img[:, :, :, :-1]))

    return tv_h + tv_w

def train_and_evaluate():
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    NUM_EPOCHS = 50
    BATCH_SIZE = 4
    LEARNING_RATE = 2e-4
    MASK_PATH = "/data/zm/Moshaboli/new_data/other_data/hot_pixel_mask_strict.npy"

    # 你的数据路径字典 (请替换为你真实的路径和 d 值)
    train_config = {
        "/data/zm/Moshaboli/new_data/no5": 0.01978,  # 比如某次标定的散斑为 50um
        #"/data/zm/2026.1.12_testdata/1.15_150_680W": 0.0105,  # 换了镜头后标定为 48um
        "/data/zm/Moshaboli/new_data/no1": 0.01891,
    }

    val_config = {
        "/data/zm/Moshaboli/new_data/no2": 0.01941,
        "/data/zm/Moshaboli/new_data/no4": 0.01973
    }

    print("正在加载训练集...")
    train_dataset = CelexBloodFlowDataset(data_config=train_config, mask_path=MASK_PATH)

    # ==============================================================================
    # 【防卡死且提速的核心设置 2】：DataLoader 性能拉满
    # 1. num_workers=8 或 12：14900K 有 32 线程，开 8~12 个子进程处理 CSV 毫无压力。
    # 2. pin_memory=True：在把数据送到 4090D 时，启用锁页内存，极大提升 CPU->GPU 传输速度。
    # 3. prefetch_factor=2：提前在后台准备好 2 个 batch 的数据，让 GPU 永远不用等 CPU。
    # ==============================================================================
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=sequence_sparse_collate,
        num_workers=8,  # 黄金参数：利用 I9 的多核优势
        pin_memory=True,  # 黄金参数：加速 GPU 传输
        prefetch_factor=2,  # 黄金参数：预加载数据
        persistent_workers=True  # 黄金参数：避免每个 Epoch 重新销毁重建子进程
    )

    print("正在加载验证集...")
    val_dataset = CelexBloodFlowDataset(data_config=val_config, mask_path=MASK_PATH)
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=sequence_sparse_collate,
        num_workers=8,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True
    )

    print(f"数据加载完成！训练集样本数: {len(train_dataset)}, 验证集样本数: {len(val_dataset)}")

    model = SNN_CNN_Hybrid().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    best_val_loss = float('inf')

    # ==============================================================================
    # 【新增】：初始化列表，用于记录每一个 Epoch 的 Loss，方便最后画图
    # ==============================================================================
    history_train_loss = []
    history_val_loss = []

    # ================= 4. 训练与验证主循环 =================
    for epoch in range(NUM_EPOCHS):

        # ----------------- 训练阶段 -----------------
        model.train()
        train_loss_total = 0.0
        pbar_train = tqdm(train_loader, desc=f"Epoch [{epoch + 1}/{NUM_EPOCHS}] Training")

        # 注意这里接收的是 seq_data (包含坐标和特征的元组列表)
        for batch_idx, (seq_data, y_true, d_values) in enumerate(pbar_train):


            # 【关键修改】：在主进程中，将数据搬到 GPU，并现场组装成 ME.SparseTensor
            x_seq = []
            for b_coords, b_feats in seq_data:
                # 把坐标和特征放到 GPU 上，然后构建稀疏张量
                sparse_tensor = ME.SparseTensor(features=b_feats.to(DEVICE),
                                                coordinates=b_coords.to(DEVICE))
                x_seq.append(sparse_tensor)

            y_true = y_true.to(DEVICE)
            d_values = d_values.to(DEVICE)

            optimizer.zero_grad()
            # 【关键修改】：把真实的 batch_size 传给前向传播函数
            # 现在网络输出的不再是时间，而是频率 fc = 1 / tau_c
            fc_pred = model(x_seq, actual_batch_size=len(y_true))

            d_values_expanded = d_values.view(-1, 1, 1, 1)
            # 乘法永远是绝对安全的，彻底告别梯度爆炸！
            v_pred = d_values_expanded * fc_pred
            y_true_expanded = y_true.view(-1, 1, 1, 1).expand_as(v_pred)

            # ================== TV Loss 修改开始 ==================
            # 1. 计算原本的主损失 (预测速度与真实速度的均方误差)
            main_loss = F.mse_loss(v_pred, y_true_expanded)

            # 2. 计算 TV Loss
            # 注意：如果你的网络输出还是 768 宽，我们需要像 evaluate 里一样，
            # 截取中心纯净区域 [200:568] 来计算平滑度，防止网络去过度平滑边缘的光学畸变。
            # (如果你在 model.py 里已经把输出宽度改成了 368，请把 [:, :, :, 200:568] 删掉，直接传 v_pred)
            v_pred_map = v_pred
            loss_tv = total_variation_loss(v_pred_map)

            # 3. 组合最终的总损失 (Total Loss)
            lambda_tv = 0.005  # TV Loss 的权重，0.001 是一个非常经典的起步值
            loss = main_loss + lambda_tv * loss_tv
            # ================== TV Loss 修改结束 ==================

            loss.backward()
            # 加入梯度裁剪，强制将过大的梯度砍掉，保护网络权重不被摧毁
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss_total += loss.item()
            pbar_train.set_postfix({'Loss': f"{loss.item():.4f}"})

        avg_train_loss = train_loss_total / len(train_loader)
        history_train_loss.append(avg_train_loss)  # 记录当前 Epoch 的训练 Loss

        # ----------------- 验证阶段 -----------------
        model.eval()
        val_loss_total = 0.0
        pbar_val = tqdm(val_loader, desc=f"Epoch [{epoch + 1}/{NUM_EPOCHS}] Validation")

        with torch.no_grad():
            for seq_data, y_true, d_values in pbar_val:

                # 【关键修改】：验证集同样采用现场组装
                x_seq = []
                for b_coords, b_feats in seq_data:
                    sparse_tensor = ME.SparseTensor(features=b_feats.to(DEVICE),
                                                    coordinates=b_coords.to(DEVICE))
                    x_seq.append(sparse_tensor)

                y_true = y_true.to(DEVICE)
                d_values = d_values.to(DEVICE)

                # 【关键修改】：把真实的 batch_size 传给前向传播函数
                # 现在网络输出的不再是时间，而是频率 fc = 1 / tau_c
                fc_pred = model(x_seq, actual_batch_size=len(y_true))

                d_values_expanded = d_values.view(-1, 1, 1, 1)
                # 乘法永远是绝对安全的，彻底告别梯度爆炸！
                v_pred = d_values_expanded * fc_pred
                y_true_expanded = y_true.view(-1, 1, 1, 1).expand_as(v_pred)

                loss = F.mse_loss(v_pred, y_true_expanded)
                val_loss_total += loss.item()
                pbar_val.set_postfix({'Val_Loss': f"{loss.item():.4f}"})

        avg_val_loss = val_loss_total / len(val_loader)
        history_val_loss.append(avg_val_loss)  # 记录当前 Epoch 的验证 Loss
        print(f"--> Epoch {epoch + 1} 总结 | Train Avg Loss: {avg_train_loss:.4f} | Val Avg Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "/data/zm/Moshaboli/new_data/Model/best_hybrid_model.pth")
            print(f"[*] 发现更低验证集误差，已保存最佳模型 (Val Loss: {best_val_loss:.4f})")

    # ==============================================================================
    # 【新增】：5. 训练结束，绘制并保存 Loss 曲线图
    # ==============================================================================
    print("训练结束，正在绘制并保存 Loss 曲线...")
    plt.figure(figsize=(10, 6))

    # 绘制训练和验证 Loss，打上标签
    plt.plot(range(1, NUM_EPOCHS + 1), history_train_loss, label='Train Loss', marker='o', color='blue')
    plt.plot(range(1, NUM_EPOCHS + 1), history_val_loss, label='Validation Loss', marker='s', color='orange')

    # 设置图表标题、坐标轴标签和网格
    plt.title('Training and Validation Loss Curve', fontsize=16)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('MSE Loss', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)

    # 将图片保存到当前目录下
    plot_path = "/data/zm/Moshaboli/new_data/Loss_curve/loss_curve.png"
    plt.savefig(plot_path, bbox_inches='tight', dpi=300)
    plt.close()  # 养成好习惯，画完图关闭画布释放内存
    print(f"Loss 曲线已成功保存至当前目录: {plot_path}")

if __name__ == '__main__':
    train_and_evaluate()
