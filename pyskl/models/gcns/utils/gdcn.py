import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphDiffusionConvolution(nn.Module):
    def __init__(self, in_channels, out_channels, diffusion_steps=3, loop_weight=0.1):
        super(GraphDiffusionConvolution, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.diffusion_steps = diffusion_steps  # 图扩散的步数
        self.loop_weight = loop_weight  # 自环的权重

        # 定义用于卷积的权重矩阵
        self.weight = nn.Parameter(torch.FloatTensor(in_channels, out_channels))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x, adjacency_matrix):
        # 添加自环到邻接矩阵
        identity_matrix = torch.eye(adjacency_matrix.size(0), device=adjacency_matrix.device)
        A_hat = adjacency_matrix + self.loop_weight * identity_matrix

        # 计算节点的度矩阵
        degree_matrix = torch.diag(A_hat.sum(dim=1))
        degree_inv = torch.inverse(degree_matrix)

        # 计算随机游走转移矩阵
        T_rw = torch.mm(degree_inv, A_hat)

        # 计算广义图扩散矩阵 S
        diffusion_matrix = torch.zeros_like(T_rw)
        current_step = torch.eye(T_rw.size(0), device=T_rw.device)
        
        for k in range(self.diffusion_steps):
            diffusion_matrix += current_step
            current_step = torch.mm(current_step, T_rw)
        
        diffusion_matrix /= self.diffusion_steps  # 归一化处理

        # 执行图卷积
        out = torch.mm(diffusion_matrix, x)
        out = torch.mm(out, self.weight)

        return out

# 示例模型使用
class GDCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, diffusion_steps=3):
        super(GDCN, self).__init__()
        self.gdc1 = GraphDiffusionConvolution(in_channels, hidden_channels, diffusion_steps)
        self.gdc2 = GraphDiffusionConvolution(hidden_channels, out_channels, diffusion_steps)

    def forward(self, x, adjacency_matrix):
        x = F.relu(self.gdc1(x, adjacency_matrix))
        x = self.gdc2(x, adjacency_matrix)
        return F.log_softmax(x, dim=1)

# 示例输入
node_features = torch.randn(5, 10)  # 5个节点，每个节点10个特征
adjacency_matrix = torch.tensor([[0, 1, 0, 0, 1],
                                 [1, 0, 1, 0, 0],
                                 [0, 1, 0, 1, 0],
                                 [0, 0, 1, 0, 1],
                                 [1, 0, 0, 1, 0]], dtype=torch.float32)

model = GDCN(in_channels=10, hidden_channels=16, out_channels=2, diffusion_steps=3)
output = model(node_features, adjacency_matrix)
print(output)
