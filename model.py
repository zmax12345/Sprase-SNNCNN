import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import MinkowskiEngine as ME
from sparse_spk_layers import SparseSpikingConv2D


# --- 网络模型定义 ---
class SNN_CNN_Hybrid(nn.Module):
    def __init__(self):
        super(SNN_CNN_Hybrid, self).__init__()
        # SNN 输入: 100x768。经过两次 stride=2 的卷积
        # 第1层输出: 50x384
        self.snn_enc1 = SparseSpikingConv2D(in_channels=1, out_channels=16, kernel=(5, 5), out_shape=(50, 384),
                                            stride=(2, 2))
        # 第2层输出: 25x192
        self.snn_enc2 = SparseSpikingConv2D(in_channels=16, out_channels=32, kernel=(3, 3), out_shape=(25, 192),
                                            stride=(2, 2), return_dense=True)

        # CNN 解码器
        self.cnn_dec = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=3, padding=1),  # 输出1通道: tau_c
            nn.Upsample(size=(100, 768), mode='bilinear', align_corners=False),  # 恢复到ROI大小
            nn.Softplus()  # 确保 tau_c 恒为正数，防止除零错误
        )

    def forward(self, x_seq):
        batch_size = int(torch.max(x_seq[0].C[:, 0])) + 1
        mem1, mem2 = None, None

        for x_sparse in x_seq:
            out1, mem1 = self.snn_enc1(x_sparse, mem=mem1, bs=batch_size)
            out2, mem2 = self.snn_enc2(out1, mem=mem2, bs=batch_size)

        # 输出每个像素点的去相关时间 tau_c, shape: [Batch, 1, 100, 768]
        tau_c = self.cnn_dec(mem2)
        return tau_c


# --- 序列合并函数 ---
def sequence_sparse_collate(batch):
    seq_len = len(batch[0][0])
    batched_seq = []
    for t in range(seq_len):
        coords_t = [sample[0][t][0] for sample in batch]
        feats_t = [sample[0][t][1] for sample in batch]
        b_coords, b_feats = ME.utils.batch_sparse_collate(coords_t, feats_t)
        batched_seq.append(ME.SparseTensor(features=b_feats, coordinates=b_coords))

    labels = torch.tensor([sample[1] for sample in batch], dtype=torch.float32)
    return batched_seq, labels


def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 定义你的多组数据配置
    experiment_config = {
        "/data/zm/2026.1.12_testdata/gaoyuzhi": 0.014611,  # 比如某次标定的散斑为 50um
        "/data/zm/2026.1.12_testdata/1.15_150_680W": 0.0105,  # 换了镜头后标定为 48um
        "/data/zm/2026.1.12_testdata/1.15_150_580W": 0.0114853,
    }

    # 传入配置字典
    dataset = CelexBloodFlowDataset(data_config=experiment_config)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=sequence_sparse_collate, num_workers=4)

    model = SNN_CNN_Hybrid().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(50):
        model.train()
        # 注意这里接收到了 d_values
        for batch_idx, (x_seq, y_true, d_values) in enumerate(dataloader):
            x_seq = [x.to(device) for x in x_seq]
            y_true = y_true.to(device)  # shape: [Batch]
            d_values = d_values.to(device)  # shape: [Batch]

            optimizer.zero_grad()

            # 前向计算得到去相关时间 tau_c_pred，尺寸为 [Batch, 1, 100, 1280]
            tau_c_pred = model(x_seq)

            # 关键张量操作：将 d_values 从 [Batch] 扩展为 [Batch, 1, 1, 1] 以便进行逐像素除法
            d_values_expanded = d_values.view(-1, 1, 1, 1)

            # 物理映射：每个样本用它自己的专属散斑尺寸计算流速！
            v_pred = d_values_expanded / (tau_c_pred + 1e-8)

            # 将真值扩展至全图尺寸
            y_true_expanded = y_true.view(-1, 1, 1, 1).expand_as(v_pred)

            loss = F.mse_loss(v_pred, y_true_expanded)

            loss.backward()
            optimizer.step()

            print(f"Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item():.4f}")


if __name__ == '__main__':
    train()