import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import torchvision.models as models
import sys
sys.path.append("/workspace/code/RadarSimulator2")
from Network.modules import PointNet2MSG
import math

device = torch.device("cuda:2")

class KLDivergenceLoss(nn.Module):
    def __init__(self):
        super(KLDivergenceLoss, self).__init__()

    def forward(self, P_pred, P_true):

        # 将预测分布和真实分布归一化
        P_pred = P_pred / P_pred.sum(dim=(2, 3), keepdim=True)
        P_true = P_true / P_true.sum(dim=(2, 3), keepdim=True)

        # 为避免log(0)的问题，对 P_pred 进行平滑处理
        P_pred = torch.clamp(P_pred, min=1e-10)
        P_true = torch.clamp(P_true, min=1e-10)
        # 对每个像素计算KL散度
        kl_div = P_true * torch.log(P_true / P_pred)

        # 对图像的所有像素的KL散度进行求和，然后对batch取平均
        loss = torch.sum(kl_div, dim=(1, 2, 3)).mean()

        return loss

class MultiTaskLoss(nn.Module):
    def __init__(self, model, weight_1=0.5, weight_2=0.5):
        super(MultiTaskLoss, self).__init__()
        self.model = model
        self.weight_1 = weight_1
        self.weight_2 = weight_2

    def forward(self, loss_1, loss_2):
        # 获取需要梯度的参数
        params = [p for p in self.model.parameters() if p.requires_grad]

        # 计算梯度，允许未使用的参数
        grad_1 = torch.autograd.grad(loss_1, params, retain_graph=True, create_graph=True, allow_unused=True)
        grad_2 = torch.autograd.grad(loss_2, params, retain_graph=True, create_graph=True, allow_unused=True)

        # 计算范数，忽略 None 值
        norm_1 = sum(torch.norm(g) for g in grad_1 if g is not None)
        norm_2 = sum(torch.norm(g) for g in grad_2 if g is not None)

        # 动态调整权重
        total_norm = norm_1 + norm_2
        if total_norm > 0:  # 防止零除
            self.weight_1 = norm_2 / total_norm
            self.weight_2 = norm_1 / total_norm

        # 总损失
        total_loss = self.weight_1 * loss_1 + self.weight_2 * loss_2
        return total_loss

class NormalizedLoss(nn.Module):
    def __init__(self, epsilon=1e-8):
        super(NormalizedLoss, self).__init__()
        self.epsilon = epsilon

    def forward(self, losses):
        """
        :param losses: 一个列表，包含每个任务的损失值 [L1, L2, ..., Ln]
        :return: 加权后的总损失
        """
        normalized_weights = [1.0 / (loss + self.epsilon) for loss in losses]
        total_loss = sum(w * l for w, l in zip(normalized_weights, losses))
        return total_loss


class SelfAttention(nn.Module):
    def __init__(self, embed_dim, heads):
        """
        初始化 Self-Attention 模块。
        :param embed_dim: 输入特征维度。
        :param heads: 注意力头的数量。
        """
        super(SelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.heads = heads
        self.head_dim = embed_dim // heads
        
        assert self.head_dim * heads == embed_dim, "Embed dimension must be divisible by the number of heads"
        
        # 定义线性变换
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.fc_out = nn.Linear(embed_dim, embed_dim)
    
    def forward(self, x):
        batch_size, seq_length, embed_dim = x.shape
        
        # 计算 Q, K, V
        Q = self.query(x)  # (batch_size, seq_length, embed_dim)
        K = self.key(x)    # (batch_size, seq_length, embed_dim)
        V = self.value(x)  # (batch_size, seq_length, embed_dim)
        
        # 分解成多头
        Q = Q.view(batch_size, seq_length, self.heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, seq_length, self.heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, seq_length, self.heads, self.head_dim).permute(0, 2, 1, 3)
        
        # Scaled Dot-Product Attention
        energy = torch.matmul(Q, K.permute(0, 1, 3, 2))  # (batch_size, heads, seq_length, seq_length)
        scale = torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))
        attention = torch.softmax(energy / scale, dim=-1)
        
        # 加权求和
        out = torch.matmul(attention, V)  # (batch_size, heads, seq_length, head_dim)
        
        # 合并多头
        out = out.permute(0, 2, 1, 3).contiguous()
        out = out.view(batch_size, seq_length, embed_dim)
        
        return self.fc_out(out)


class SelfAttentionWithPE(nn.Module):
    def __init__(self, embed_dim, heads):
        """
        初始化 Self-Attention 模块，加入位置编码。
        :param embed_dim: 输入特征维度。
        :param heads: 注意力头的数量。
        :param height: 输入图像的高度。
        :param width: 输入图像的宽度。
        """
        super(SelfAttentionWithPE, self).__init__()
        self.embed_dim = embed_dim
        self.heads = heads
        self.head_dim = embed_dim // heads
        
        assert self.head_dim * heads == embed_dim, "Embed dimension must be divisible by the number of heads"
        
        # 定义线性变换
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.fc_out = nn.Linear(embed_dim, embed_dim)

    def create_positional_encoding(self, height, width, embed_dim):

        pos_h = torch.arange(height).unsqueeze(1).repeat(1, width).unsqueeze(-1)  # (height, width, 1)
        pos_w = torch.arange(width).unsqueeze(0).repeat(height, 1).unsqueeze(-1)  # (height, width, 1)
        pos = torch.cat([pos_h, pos_w], dim=-1)  # (height, width, 2)
        
        # 映射到 embed_dim
        pos = pos.view(-1, 2).float()  # (height * width, 2)
        pos = nn.Linear(2, embed_dim)(pos)  # (height * width, embed_dim)
        return pos.view(height, width, embed_dim)  # (height, width, embed_dim)

    def forward(self, x):

        b, c, h, w = x.shape
        positional_encoding = self.create_positional_encoding(h, w, self.embed_dim)
        
        # 展平并添加位置编码
        x = x.permute(0, 2, 3, 1).reshape(b, h * w, c)  # (b, h * w, c)
        pos = positional_encoding.view(1, h * w, -1).to(x.device)  # (1, h * w, embed_dim)
        x = x + pos
        
        # Self-Attention
        Q = self.query(x)  # (b, h * w, embed_dim)
        K = self.key(x)    # (b, h * w, embed_dim)
        V = self.value(x)  # (b, h * w, embed_dim)

        # 分解成多头
        Q = Q.view(b, h * w, self.heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(b, h * w, self.heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(b, h * w, self.heads, self.head_dim).permute(0, 2, 1, 3)

        # Scaled Dot-Product Attention
        energy = torch.matmul(Q, K.permute(0, 1, 3, 2))  # (b, heads, h * w, h * w)
        scale = torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))
        attention = torch.softmax(energy / scale, dim=-1)

        # 加权求和
        out = torch.matmul(attention, V)  # (b, heads, h * w, head_dim)

        # 合并多头
        out = out.permute(0, 2, 1, 3).contiguous()
        out = out.view(b, h * w, self.embed_dim)

        return self.fc_out(out).view(b, h, w, -1).permute(0, 3, 1, 2)



class Regression(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Regression, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)

        self.fc_velo = nn.Linear(1, hidden_dim)

        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim*2, 1)
        # self.fc2 = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()
        self.to_mlp = nn.Sequential(
            nn.Conv2d(512, 1, kernel_size=3, stride=2, padding=0),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )

        #self.edge_mlp = nn.Linear(6832,hidden_dim)

    def forward(self, x, ego_velo):
        x = self.to_mlp(x)
        x = x.reshape(len(ego_velo), -1)
        x = self.fc1(x)
        x = self.relu(x)

        ego_velo = self.fc_velo(ego_velo)
        ego_velo = self.relu(ego_velo)
        x = torch.cat((x, ego_velo), dim = 1)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x


class Regression_with_F(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Regression_with_F, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        #self.bn1 = nn.BatchNorm1d(hidden_dim)

        self.fc_velo = nn.Linear(1, hidden_dim)

        #self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim*2, 1)
        self.sigmoid = nn.Sigmoid()
        self.to_mlp = nn.Sequential(
            nn.Conv2d(512, 1, kernel_size=3, stride=2, padding=0),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        # self.rgbe_layer = nn.Sequential(
        #     nn.Conv2d(4, 1, kernel_size=3, stride=2, padding=0),
        #     nn.BatchNorm2d(1),
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool2d(2, 2)
        # )

        #self.edge_mlp = nn.Linear(6832,hidden_dim)
        self.edge_mlp = nn.Linear(54656, hidden_dim)



    def forward(self, edge, x, ego_velo):
        edge = edge.reshape(len(ego_velo),-1)
        edge = self.edge_mlp(edge)
        x = self.to_mlp(x)
        # print(resnet_out.shape)
        x = x.reshape(len(ego_velo), -1)
        x = self.fc1(x)
        #x = self.bn1(x)
        x = self.relu(x)

        ego_velo = self.fc_velo(ego_velo)

        ego_velo = self.relu(ego_velo)
        x = torch.cat((x,  edge, ego_velo), dim = 1)

        x = self.fc2(x)
        x = self.sigmoid(x)
        return x


class Conv(nn.Module):
    def __init__(self,original_size):
        super(Conv, self).__init__()
        self.layer1 = nn.Sequential(
            #nn.ConvTranspose2d(512+32, 256, kernel_size=4, stride=4, padding=0),
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=4, padding=0),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        # 第二层，输出尺寸: 112x242
        self.layer2 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        # 第三层，输出尺寸: 28x61
        self.layer3 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        # 第四层，输出尺寸: 23x49
        self.layer4 = nn.Sequential(
            nn.ConvTranspose2d(64, 1, kernel_size=(2, 3), stride=2, padding=(6, 15)),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True),
        )
        #self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1) 
        self.upsample = nn.Upsample(size=original_size, mode='bilinear', align_corners=True)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.upsample(x)
        #x = self.sigmoid(x)
        b,c,h,w = x.shape
        x = x.reshape(b,-1)
        x = self.softmax(x)
        x = x.reshape(b,c,h,w)

        return x



class Dis_Est_Head(nn.Module):
    def __init__(self,original_size):
        super(Dis_Est_Head, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, stride=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, stride=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((20, 53)),
        )

        self.layer4 = nn.Sequential(
            nn.Conv2d(64, 1, kernel_size=1, stride=1),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True),
        )
        self.rgb_conv = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=1, stride=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 128, kernel_size=1, stride=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 64, kernel_size=1, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d((20, 53))
        )

        self.lidar2d_conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=1, stride=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),

            nn.Conv2d(16, 32, kernel_size=1, stride=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 64, kernel_size=1, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d((20, 53))
        )

        self.sigmoid = nn.Sigmoid()
        self.original_size = original_size
        #self.upsample = nn.Upsample(size=original_size, mode='bilinear', align_corners=True)

    def forward(self, x, lidar_2d=None, rgb=None):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        #x = self.upsample(x)
        if lidar_2d is not None and rgb is not None:
            lidar_2d = self.lidar2d_conv(lidar_2d)
            rgb = self.rgb_conv(rgb)
            b,c,h,w = lidar_2d.shape
            lidar_2d = lidar_2d.reshape(b,h*w, c)
            rgb = rgb.reshape(b,c,h*w)
            norm1 = torch.norm(lidar_2d, p=2, dim=2, keepdim=True) 
            norm2 = torch.norm(rgb, p=2, dim=1, keepdim=True) 
            heatmap = torch.bmm(lidar_2d, rgb) / (norm1 * norm2 + 1e-7)
            heatmap = F.softmax(heatmap / 0.1, dim=2)

            x = x.reshape(b,-1,c)
            #print(heatmap.shape, x.shape)   #torch.Size([4, 1060, 1060]) torch.Size([4, 1060, 64])
            x =  torch.bmm(heatmap, x)
            #print(x.shape)  #torch.Size([4, 1060, 64])
            x = x.reshape(b,c,h,w)
        x = self.layer4(x)
        x = self.sigmoid(x)
        #print(x.shape)      #torch.Size([4, 1, 20, 53])
        #x = F.interpolate(x, size=self.original_size, mode='nearest')
        #print(x.shape)

        return x


class TransformerEncoderModule(nn.Module):
    def __init__(self, embed_dim, num_heads, num_layers, dim_feedforward, dropout):
        super(TransformerEncoderModule, self).__init__()
        # 定义单个 Transformer Encoder Layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )
        # 堆叠多个 Transformer Encoder Layer
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=num_layers
        )

    def forward(self, x):
        # x.shape: [batch_size, channels, height, width]
        batch_size, channels, height, width = x.shape
        
        # 将 x 转换为序列格式
        x = x.flatten(2)  # [batch_size, channels, height * width]
        x = x.permute(2, 0, 1)  # [seq_len, batch_size, embed_dim]

        # 通过 Transformer Encoder
        x = self.transformer_encoder(x)  # [seq_len, batch_size, embed_dim]

        # 转回原来的维度
        x = x.permute(1, 2, 0).contiguous()  # [batch_size, embed_dim, seq_len]
        x = x.reshape(batch_size, channels, height, width)  # [batch_size, channels, height, width]
        return x


class CrossAttention(nn.Module):
    def __init__(self, input_dim_lidar=128, input_dim_visual=128, npoint = 512):
        super(CrossAttention, self).__init__()
        #self.linear_lidar = nn.Linear(input_dim_lidar, 21*61)
        self.input_dim_lidar = input_dim_lidar
        self.input_dim_visual = input_dim_visual
        self.npoint = npoint
        # self.conv_block = nn.Sequential(
        #     nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=1, stride=1, padding=0),
        #     nn.BatchNorm2d(512),
        #     nn.ReLU()
        # )
        
            
    def forward(self, lidar_features, visual_features):
        batch_size = lidar_features.size(0)
        #lidar_features = lidar_features.reshape(batch_size, self.input_dim_lidar, -1)  # (4, 128, 512)
        '''
        #lidar_features = lidar_features.reshape(batch_size * lidar_features.size(2), -1)  # (4 * 512, 128)
        # Q1 = self.linear_lidar(lidar_features)  
        # Q1 = Q1.reshape(batch_size,self.input_dim_visual,-1)    # (4, 512, 21*61)
        '''
        lidar_features = lidar_features.reshape(batch_size, -1, self.input_dim_lidar)  # (4, 512, 128)
        Q1 = lidar_features
        # Q1 = F.interpolate(lidar_features, size=(21, 61), mode='bilinear', align_corners=False)
        # Q1 = Q1.reshape(batch_size,512,-1)        #b,length, embedding: b, n_point, feature
        # Q1 = lidar_features.reshape(batch_size,-1,self.input_dim_lidar)

        # 视觉特征作为 K（键）和值 V
        # K1 = visual_features.reshape(batch_size, 512, -1)  # (4, 512, 21*61)
        # V1 = K1  # (4, 512, 21*61)
        b,c,h,w = visual_features.shape
        K1 = visual_features.reshape(batch_size, self.npoint, self.input_dim_visual)  # (4, 21*61, 512)
        V1 = K1  # (4, 21*61, 512)

        # 计算注意力分数
        attention_scores1 = torch.matmul(Q1, K1.transpose(-2, -1))  # (4, 512, 21*61) x (4, 512, 21*61) -> (4, 512, 21*61)
        attention_scores1 = attention_scores1 / math.sqrt(self.input_dim_lidar)  # 缩放
        attention_weights1 = torch.softmax(attention_scores1, dim=-1)  # (4, 512, 21*61)

        # 对 V 进行加权求和
        output1 = torch.matmul(attention_weights1, V1)  # (4, 512, 21*61)
        #output1 = output1.reshape(batch_size, 512, 21, 61)  # (4, 512, 21, 61)
        output1 = output1.reshape(batch_size, 128, h, w)  # (4, 512, 4, 32)

        output = output1

        K2 = Q1
        V2 = K2
        Q2 = K1
        # 计算注意力分数
        attention_scores2 = torch.matmul(Q2, K2.transpose(-2, -1))  # (4, 512, 21*61) x (4, 512, 21*61) -> (4, 512, 21*61)
        attention_scores2 = attention_scores2 / math.sqrt(self.input_dim_lidar)  # 缩放
        attention_weights2 = torch.softmax(attention_scores2, dim=-1)  # (4, 512, 21*61)

        # 对 V 进行加权求和
        output2 = torch.matmul(attention_weights2, V2)  # (4, 512, 21*61)
        output2 = output2.reshape(batch_size, self.input_dim_lidar, h, w)  # (4, 512, 21, 61)

        #output = torch.cat((output1,output2),dim = 1)
        output = output1 + output2

        
        return output

class CrossAttention_2d(nn.Module):
    def __init__(self, input_dim_lidar=512, input_dim_visual=512):
        super(CrossAttention_2d, self).__init__()
        #self.linear_lidar = nn.Linear(input_dim_lidar, 21*61)
        self.input_dim_lidar = input_dim_lidar
        self.input_dim_visual = input_dim_visual
        # self.conv_block = nn.Sequential(
        #     nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=1, stride=1, padding=0),
        #     nn.BatchNorm2d(512),
        #     nn.ReLU()
        # )
        
            
    def forward(self, lidar_features, visual_features):
        batch_size = lidar_features.size(0)
        #lidar_features = lidar_features.reshape(batch_size, self.input_dim_lidar, -1)  # (4, 128, 512)
        '''
        #lidar_features = lidar_features.reshape(batch_size * lidar_features.size(2), -1)  # (4 * 512, 128)
        # Q1 = self.linear_lidar(lidar_features)  
        # Q1 = Q1.reshape(batch_size,self.input_dim_visual,-1)    # (4, 512, 21*61)
        '''
        lidar_features = lidar_features.reshape(batch_size, -1, self.input_dim_lidar)  # (4, 512, 128)
        Q1 = lidar_features
        # Q1 = F.interpolate(lidar_features, size=(21, 61), mode='bilinear', align_corners=False)
        # Q1 = Q1.reshape(batch_size,512,-1)        #b,length, embedding: b, n_point, feature
        # Q1 = lidar_features.reshape(batch_size,-1,self.input_dim_lidar)

        # 视觉特征作为 K（键）和值 V
        # K1 = visual_features.reshape(batch_size, 512, -1)  # (4, 512, 21*61)
        # V1 = K1  # (4, 512, 21*61)
        K1 = visual_features.reshape(batch_size, -1, self.input_dim_visual)  # (4, 21*61, 512)
        V1 = K1  # (4, 21*61, 512)

        # 计算注意力分数
        attention_scores1 = torch.matmul(Q1, K1.transpose(-2, -1))  # (4, 512, 21*61) x (4, 512, 21*61) -> (4, 512, 21*61)
        attention_scores1 = attention_scores1 / math.sqrt(self.input_dim_lidar)  # 缩放
        attention_weights1 = torch.softmax(attention_scores1, dim=-1)  # (4, 512, 21*61)

        # 对 V 进行加权求和
        output1 = torch.matmul(attention_weights1, V1)  # (4, 512, 21*61)
        output1 = output1.reshape(batch_size, 512, 21, 61)  # (4, 512, 21, 61)
        # output1 = output1.reshape(batch_size, 128, 16, 32)  # (4, 512, 4, 32)

        output = output1

        K2 = Q1
        V2 = K2
        Q2 = K1
        # 计算注意力分数
        attention_scores2 = torch.matmul(Q2, K2.transpose(-2, -1))  # (4, 512, 21*61) x (4, 512, 21*61) -> (4, 512, 21*61)
        attention_scores2 = attention_scores2 / math.sqrt(self.input_dim_lidar)  # 缩放
        attention_weights2 = torch.softmax(attention_scores2, dim=-1)  # (4, 512, 21*61)

        # 对 V 进行加权求和
        output2 = torch.matmul(attention_weights2, V2)  # (4, 512, 21*61)
        output2 = output2.reshape(batch_size, self.input_dim_lidar, 21, 61)  # (4, 512, 21, 61)

        output = output1 + output2

        
        return output


class Conv_Fusion(nn.Module):
    def __init__(self, input_dim_lidar=128, input_dim_visual=512, output_dim=512):
        super(Conv_Fusion, self).__init__()
        self.linear_lidar = nn.Linear(input_dim_lidar, 21*61)
        self.input_dim_lidar = input_dim_lidar
        self.input_dim_visual = input_dim_visual
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )
    
    def forward(self, lidar_features, visual_features):
        batch_size = lidar_features.size(0)
        lidar_features = lidar_features.reshape(batch_size, self.input_dim_lidar, -1)  # (4, 128, 512)
        lidar_features = lidar_features.reshape(batch_size * lidar_features.size(2), -1)  # (4 * 512, 128)

        lidar_features = self.linear_lidar(lidar_features)  
        lidar_features = lidar_features.reshape(batch_size,self.input_dim_visual,visual_features.shape[2],visual_features.shape[3])    # (4, 512, 21*61)

        fused_fea = torch.cat((visual_features,lidar_features),dim = 1)
        output = self.conv_block1(fused_fea)
       
        
        return output


class Conv_2d(nn.Module):
    def __init__(self, input_dim_lidar=128, input_dim_visual=512, output_dim=512):
        super(Conv_2d, self).__init__()
        self.network = nn.Sequential(
            # Block 1: Input [4, 1, 644, 1935] -> [4, 64, 322, 968]
            nn.Conv2d(1, 64, kernel_size=3, stride=2, padding=1),  # Conv, stride=2 for downsampling
            nn.BatchNorm2d(64),                                    # BatchNorm
            nn.ReLU(inplace=True),                                # Activation
            nn.MaxPool2d(kernel_size=2, stride=2),                # Downsample: [4, 64, 161, 484]
            
            # Block 2: [4, 64, 161, 484] -> [4, 128, 81, 242]
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),                # Downsample: [4, 128, 40, 121]
            
            # Block 3: [4, 128, 40, 121] -> [4, 256, 20, 61]
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),  # Extra conv to learn better features
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            # Block 4: [4, 256, 20, 61] -> [4, 512, 21, 61]
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),  # Increase channels to 512
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            
        )
        self.Up = nn.Upsample(size=(21, 61), mode="bilinear", align_corners=False)  # Resize to match target size

    def forward(self, x):
        x = self.network(x)
        x = self.Up(x)
        return x

class Conv_RGB(nn.Module):
    def __init__(self):
        super(Conv_RGB, self).__init__()
        
        # 使用 nn.Sequential 来按顺序组合层
        self.model = nn.Sequential(
            # 第1层卷积，使用 BatchNorm 和 ReLU
            nn.Conv2d(512, 256, kernel_size=[3, 3], stride=[1, 1], padding=[0, 1]),  # 输出: [4, 256, 11, 61]
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            # 第2层卷积，使用 BatchNorm 和 ReLU
            nn.Conv2d(256, 128, kernel_size=[3, 2], stride=[1, 2], padding=[0, 2]),  # 输出: [4, 128, 6, 31]
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d((16, 32))
    
    def forward(self, x):
        x = self.model(x)
        x = self.pool(x)
        return x

class Conv_RGB_Reverse(nn.Module):
    def __init__(self):
        super(Conv_RGB_Reverse, self).__init__()

        # 使用 nn.Sequential 来按顺序组合层
        self.model = nn.Sequential(
            # 第1层反卷积，恢复到通道数 256
            nn.ConvTranspose2d(128, 256, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1]),  # 输出: [4, 256, 6, 61]
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            # 第2层反卷积，恢复到通道数 512
            nn.ConvTranspose2d(256, 512, kernel_size=[3, 3], stride=[2, 2], padding=[1, 1]),  # 输出: [4, 512, 11, 61]
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d((21, 61))

    def forward(self, x):
        x = self.model(x)
        x = self.pool(x)
        return x


class DepthAwareModule(nn.Module):
    def __init__(self, original_size):
        super(DepthAwareModule, self).__init__()

        embed_dim, heads = 128, 4
        #self.self_attention = SelfAttention(embed_dim, heads)
        self.sa = SelfAttentionWithPE(embed_dim, heads)
        self.conv = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=3, stride=3), 
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(32, 64, kernel_size=3, stride=3), 
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, kernel_size=3, stride=3), 
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d((8, 24))
        )

        self.conv_reverse = nn.Sequential(
            # 第1层反卷积，恢复到通道数 256
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=3),  # 输出: [4, 256, 6, 61]
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # 第2层反卷积，恢复到通道数 512
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=3),  # 输出: [4, 512, 11, 61]
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(32, 3, kernel_size=3, stride=3),  # 输出: [4, 512, 11, 61]
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True),

            nn.Upsample(size=original_size, mode='bilinear', align_corners=True)

        )

    def forward(self, lidar_2d, rgb):
        rgb_clone = rgb.clone()
        x = torch.cat((rgb,lidar_2d),dim=1)
        x = self.conv(x)
        b, c, h, w = x.shape
        #x = self.self_attention(x.reshape(b,h*w,-1))
        x = self.sa(x)
        x = x.reshape(b,-1,h,w)
        x = self.conv_reverse(x)
        x += rgb_clone
        #x = torch.cat((rgb_clone,x),dim=1)
        return x


class DisNet(nn.Module):
    def __init__(self, original_size):
        super(DisNet, self).__init__()
        self.original_size = original_size

        self.resnet = models.resnet18(weights='IMAGENET1K_V1')
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-2])
        # for param in self.resnet.parameters():
        #     param.requires_grad = False

        # in_channels = 4
        # self.resnet = models.resnet18(weights='IMAGENET1K_V1')

        # # 修改第一层的输入通道数
        # self.resnet.conv1 = nn.Conv2d(in_channels, self.resnet.conv1.out_channels, kernel_size=7, stride=2, padding=3,
        #                               bias=False)

        # # 初始化第一层的新通道权重
        # original_conv1_weight = self.resnet.conv1.weight.data.clone()
        # new_conv1_weight = torch.zeros(
        #     (original_conv1_weight.size(0), in_channels, original_conv1_weight.size(2), original_conv1_weight.size(3)))

        # # 复制原始权重至新的通道中
        # new_conv1_weight[:, :3, :, :] = original_conv1_weight[:, :3, :, :]
        # if in_channels > 3:
        #     new_conv1_weight[:, 3:, :, :] = original_conv1_weight[:, :1, :, :]  # 将第一个通道的权重复制到其他通道

        # self.resnet.conv1.weight.data = new_conv1_weight

        # # 锁定除第一层外的所有权重
        # # for name, param in self.resnet.named_parameters():
        # #     if name != 'conv1.weight':
        # #         param.requires_grad = False

        # # 移除 ResNet 的最后的全连接层和池化层
        # self.resnet = nn.Sequential(*list(self.resnet.children())[:-2])


        self.Conv = Conv(original_size)
        #self.Conv = Dis_Est_Head(original_size)


        self.Regression = Regression(90,12)
        self.pool = nn.AdaptiveAvgPool2d((28, 61))
        #self.DownsampleNet = DownsampleNet()
        #embed_dim, heads = 512, 4
        # self.self_attention = SelfAttention(embed_dim, heads)

        # self.transformer = TransformerEncoderModule(
        #     embed_dim=512,      # 嵌入维度，等于 ResNet 的输出通道数
        #     num_heads=8,        # 注意力头的数量
        #     num_layers=6,       # Transformer Encoder 层的数量
        #     dim_feedforward=2048,  # FFN 中间层维度
        #     dropout=0.1         # Dropout 比例
        # )
        # self.DA = DepthAwareModule(original_size)
        # self.pts_extractor = PointNet2MSG(radii_list=[[0.01, 0.02], [0.02,0.04], [0.04,0.08], [0.08,0.16]])
        # self.pts_extractor = self.pts_extractor.to(device)
        # self.cross_attn = CrossAttention()

        #self.cross_attn = CrossAttention_2d()

        # self.conv_rgb = Conv_RGB()
        # self.conv_rgb_r = Conv_RGB_Reverse()


    def forward(self, batch_dict):
        
        local_pic = batch_dict['local_pic']
        local_pic = local_pic.permute(0, 3, 1, 2)  # 重新排列维度以符合卷积的输入要求
        x = local_pic.float().to(device)

        # l_3d = batch_dict['l_3d'].to(device)
        # lidar_fea = self.pts_extractor(l_3d)
        #print(lidar_fea.shape)      #torch.Size([4, 128, 512])

        # local_lidar_depth = batch_dict['local_lidar_depth'].float()
        # local_lidar_depth = local_lidar_depth.unsqueeze(1)
        # local_lidar_depth = local_lidar_depth.to(device)

        # x = torch.cat((x, local_lidar_depth), dim=1)
        #x = self.DA(local_lidar_depth, x)

        x = self.resnet(x)
        #res_out = x.clone()

        #x = self.cross_attn(lidar_fea, x) 

        # x = self.cross_attn(lidar_fea, self.conv_rgb(x)) 
        # x = self.conv_rgb_r(x)
        # fused_out = x.clone()
        # x = x + res_out

        # x = self.Conv_Fusion(lidar_fea,x) + res_out
        # fused_out = x.clone()


        # batch_size, channels, height, width = x.shape
        # seq_length = height * width
        # embed_dim = channels
        # x = x.reshape(batch_size, seq_length, embed_dim)
        # x = self.self_attention(x)  # [batch_size, seq_length, embed_dim]
        # x = x.reshape(batch_size, channels, height, width)

        #x = self.transformer(x) + res_out
        #x = self.transformer(x) + res_out + fused_out
        

        #print(x.shape)  #torch.Size([4, 512, 21, 61])
        x = self.pool(x)
        
        # resnet_out = x.clone()
        #resout_size = x.shape[2:4]


        #print(resnet_out.shape)     #torch.Size([8, 874496])
        ego_velocity = batch_dict['ego_velo'].float().to(device)
        ego_velocity = ego_velocity.reshape(len(ego_velocity),1)

        #out_num = self.Regression_with_F(edge_image, resnet_out,ego_velocity)
        out_num = self.Regression(x, ego_velocity)

        out_gray = self.Conv(x)
        #out_gray_small = self.Conv(x,local_lidar_depth,res_out)
        #out_gray = F.interpolate(out_gray_small, size=self.original_size, mode='nearest')

        return out_gray, out_num

