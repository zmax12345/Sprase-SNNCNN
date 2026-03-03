import os

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
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import CelexBloodFlowDataset, sequence_sparse_collate
from model import SNN_CNN_Hybrid


def train_and_evaluate():
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    NUM_EPOCHS = 50
    BATCH_SIZE = 4
    LEARNING_RATE = 1e-3
    MASK_PATH = "/data/zm/2026.1.12_testdata/3.2NEW_RESULT/Hot_pixel/hot_pixel_mask.npy"

    # 你的数据路径字典 (请替换为你真实的路径和 d 值)
    train_config = {
        "/data/zm/2026.1.12_testdata/gaoyuzhi": 0.014611,  # 比如某次标定的散斑为 50um
        "/data/zm/2026.1.12_testdata/1.15_150_680W": 0.0105,  # 换了镜头后标定为 48um
        "/data/zm/2026.1.12_testdata/2.3": 0.010099,
    }

    val_config = {
        "/data/zm/2026.1.12_testdata/1.15_150_580W": 0.0114853,
        "/data/zm/2026.1.12_testdata/1.22data": 0.010154,
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
            tau_c_pred = model(x_seq)

            d_values_expanded = d_values.view(-1, 1, 1, 1)
            v_pred = d_values_expanded / (tau_c_pred + 1e-8)
            y_true_expanded = y_true.view(-1, 1, 1, 1).expand_as(v_pred)

            loss = F.mse_loss(v_pred, y_true_expanded)
            loss.backward()
            optimizer.step()

            train_loss_total += loss.item()
            pbar_train.set_postfix({'Loss': f"{loss.item():.4f}"})

        avg_train_loss = train_loss_total / len(train_loader)

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

                tau_c_pred = model(x_seq)

                d_values_expanded = d_values.view(-1, 1, 1, 1)
                v_pred = d_values_expanded / (tau_c_pred + 1e-8)
                y_true_expanded = y_true.view(-1, 1, 1, 1).expand_as(v_pred)

                loss = F.mse_loss(v_pred, y_true_expanded)
                val_loss_total += loss.item()
                pbar_val.set_postfix({'Val_Loss': f"{loss.item():.4f}"})

        avg_val_loss = val_loss_total / len(val_loader)
        print(f"--> Epoch {epoch + 1} 总结 | Train Avg Loss: {avg_train_loss:.4f} | Val Avg Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "best_hybrid_model.pth")
            print(f"[*] 发现更低验证集误差，已保存最佳模型 (Val Loss: {best_val_loss:.4f})")


if __name__ == '__main__':
    train_and_evaluate()