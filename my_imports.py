# 导入库 神经网络库
import math
import random
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from pylab import mpl
from torch import nn  # torch神经网络库，nn是Neural Network的简称
import torch.nn.functional as F  # 定义了创建神经网络所需要的一些常见处理函数
import torch
# import gym
import torch.optim as optim  # 优化器
from torch.distributions import Categorical  # 概率分布
import time
from copy import deepcopy
import os

# 解决plot不能显示中文的问题
mpl.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 指定默认字体：解决plot不能显示中文问题
__all__ = ['Item', 'SpaceFree', 'DqnPolicy', 'DqnPolicy2', 'DqnPolicy3', 'DqnPolicy4', 'DqnPolicy5', 'DqnPolicy6', 'DqnPolicy6Legacy', 'DqnQNetwork', 'np', 'plt', 'mpl', 'random', 'math', 'calculate_max_area',
           'Poly3DCollection', 'pd', 'nn', 'F', 'torch', 'optim', 'Categorical', 'gym', 'time', 'deepcopy', 'os', 'matplotlib']


class Item:
    # 货物
    def __init__(self, length, width, height):
        self.length = length  # 货物 长
        self.width = width  # 货物 宽
        self.height = height  # 货物 高
        self.volume = length * width * height  # 货物 体积
        self.x = None  # 货物 在空间中的位置
        self.y = None
        self.z = None
        self.placed = False  # 货物 是否被放置
        self.space_free_id = None  # 货物选择空闲空间的编号
        self.place_times_max = False  # 是否达到最大放置次数
        self.place_times = 0  # 记录每个货物被放置的次数


class SpaceFree:
    def __init__(self, x, y, z, length, width, height):
        self.x = x  # 空间 起始位置
        self.y = y
        self.z = z
        self.length = length  # 空间 长度
        self.width = width
        self.height = height
        self.volume = length * width * height  # 空间体积
        self.occupied = False  # 空间 是否 被占用
        self.segment1 = True  # 空间 是否 可以 继续分割
        self.segment2 = True  # 空间 是否 要 继续分割


class DqnPolicy(nn.Module):
    # 初始版本
    def __init__(self, input_dim, output_dim):
        super(DqnPolicy, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        # 输入层input_dim个参数
        # 隐藏层128个神经元
        # 输出层output_dim个参数
        # 定义两个线性层，大小分别为input_dim*128与128*output_dim
        self.fc1 = nn.Linear(input_dim, 128)  # 输入层和隐藏层
        self.fc2 = nn.Linear(128, output_dim)  # 隐藏层和输出层
        # 定义一个dropout层，丢弃比率是60%
        self.drop = nn.Dropout(p=0.6)  # 适当丢弃一些神经元来训练模型，增加模型的鲁棒性

    def forward(self, x, space_free_num_max, space_free):
        # 前向传播函数，输入X
        x = self.fc1(x)  # 线性层
        x = self.drop(x)  # dropout层
        x = F.relu(x)  # 非线性激活层
        x = self.fc2(x)  # 线性层
        # 使用softmax决策最终的行动
        probs = F.softmax(x, dim=1)
        if torch.isnan(probs).any() or torch.isinf(probs).any() or probs.sum() <= 0:
            print('probs有误！')
        # 重新设计probs
        mask = torch.zeros(probs.size(1), dtype=torch.bool, device=probs.device)
        mask[:min(space_free_num_max, len(space_free))] = True
        probs_masked = probs * mask.float()
        probs_masked = probs_masked / probs_masked.sum(dim=-1, keepdim=True)
        probs = probs_masked
        prob_sum = probs_masked.sum(dim=-1, keepdim=True)  # 概率和
        if torch.isnan(probs).any() or torch.isinf(probs).any() or probs.sum() <= 0:
            print('probs有误！')
        if abs(prob_sum - 1) > 1e-5:
            print('probs和有误！')
        return probs


class DqnPolicy2(nn.Module):
    # 训练时固定货物数量,空间数量
    def __init__(self, item_num_max, space_free_num_max):
        super(DqnPolicy2, self).__init__()
        self.item_num_max = item_num_max  # 最大货物数量
        self.space_free_num_max = space_free_num_max  # 最大剩余空间数量

        # 货物特征提取网络
        self.item_encoder = nn.Sequential(
            nn.Linear(3, 64),  # 输入为单个货物的长宽高
            nn.LayerNorm(64),  # 归一化层
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.LayerNorm(128),  # 归一化层
            nn.ReLU()
        )

        # 空间特征提取网络
        self.space_encoder = nn.Sequential(
            nn.Linear(3, 64),  # 输入为单个空间的长宽高
            nn.LayerNorm(64),  # 归一化层
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.LayerNorm(128),  # 归一化层
            nn.ReLU()
        )

        # 融合层
        self.fusion = nn.Sequential(
            nn.Linear((item_num_max + space_free_num_max) * 128, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU()
        )

        # 输出层：分别输出货物选择概率和空间选择概率
        self.item_output = nn.Linear(256, item_num_max)
        self.space_output = nn.Linear(256, space_free_num_max)

    def check_nan_inf(self, tensor, name):
        # 检查张量是否包含NaN，inf
        if torch.isnan(tensor).any():
            print(f"{name}存在nan")
            return True
        if torch.isinf(tensor).any():
            print(f"{name}存在inf")
            return True
        if tensor.sum() <= 0:
            print(f"{name}和有误")
            return True
        '''
        if abs(tensor.sum() - 1) > 1e-5:
            print('和有误！')
        '''
        return False

    def forward(self, item, item_unplaced_order, space_free, device):
        # 前向传播
        # 编码所有货物
        item_features = torch.zeros((self.item_num_max, 3), device=device)
        valid_item = min(self.item_num_max, len(item_unplaced_order))
        for i in range(valid_item):
            item_features[i] = torch.tensor([item[item_unplaced_order[i]].length, item[item_unplaced_order[i]].width, item[item_unplaced_order[i]].height], device=device)
        encoded_item = self.item_encoder(item_features.view(-1, 3))
        encoded_item = encoded_item.view(1, -1)
        self.check_nan_inf(encoded_item, "encoded_item")

        # 检查模型
        for name, param in self.named_parameters():
            if torch.isnan(param).any() or torch.isinf(param).any():
                print(f"参数 {name} 包含 NaN 或 Inf！")

        # 编码所有空间
        space_features = torch.zeros((self.space_free_num_max, 3), device=device)
        valid_space = min(self.space_free_num_max, len(space_free))
        for i in range(valid_space):
            space_features[i] = torch.tensor([space_free[i].length, space_free[i].width, space_free[i].height], device=device)
        # 批量编码
        encoded_space = self.space_encoder(space_features.view(-1, 3))
        encoded_space = encoded_space.view(1, -1)
        self.check_nan_inf(encoded_space, "encoded_space")

        # 融合特征
        fused = torch.cat([encoded_item, encoded_space], dim=1)
        fused = self.fusion(fused)

        # 计算输出概率
        item_logits = self.item_output(fused)
        space_logits = self.space_output(fused)

        # 创建掩码（softmax前应用）
        item_mask = torch.zeros(item_logits.size(1), device=device)
        item_mask[:valid_item] = 1

        space_mask = torch.zeros(space_logits.size(1), device=device)
        space_mask[:valid_space] = 1

        # 应用掩码并计算概率
        item_probs = F.softmax(item_logits * item_mask + (1 - item_mask) * -1e10, dim=1)
        space_probs = F.softmax(space_logits * space_mask + (1 - space_mask) * -1e10, dim=1)

        # 检查数值
        self.check_nan_inf(item_probs, "item_probs")
        self.check_nan_inf(space_probs, "space_probs")
        '''
        if (item_probs > 1e-10).float().sum().item() != len(item_unplaced_order) and len(item_unplaced_order) <= self.item_num_max:
            1
            # print("有误！")
        '''
        return item_probs, space_probs


class DqnPolicy3(nn.Module):
    def __init__(self, item_num_max, space_free_num_max, hidden_dim=128):
        super(DqnPolicy3, self).__init__()
        self.item_num_max = item_num_max  # 最大货物数量
        self.space_free_num_max = space_free_num_max  # 最大剩余空间数量
        self.hidden_dim = hidden_dim

        # 货物特征提取网络
        self.item_encoder = nn.Sequential(
            nn.Linear(3, 64),  # 输入为单个货物的长宽高
            nn.LayerNorm(64),  # 归一化层
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )

        # 空间特征提取网络
        self.space_encoder = nn.Sequential(
            nn.Linear(3, 64),  # 输入为单个空间的长宽高
            nn.LayerNorm(64),  # 归一化层
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        # 注意力聚合层（处理货物特征，忽略填充值）
        self.item_attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),  # 隐藏层的激活函数
            nn.Linear(hidden_dim // 2, 1)  # 输出每个货物的注意力权重
        )

        # 空间注意力聚合层
        self.space_attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )

        # 融合层
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, 512),  # 货物聚合特征 + 空间聚合特征
            nn.ReLU(),
            nn.Dropout(0.2),  # 防止过拟合
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        # 输出层：分别输出货物选择概率和空间选择概率
        self.item_output = nn.Linear(256, item_num_max)
        self.space_output = nn.Linear(256, space_free_num_max)

    def check_nan_inf(self, tensor, name):
        # 检查张量是否包含NaN，inf
        if torch.isnan(tensor).any():
            print(f"{name}存在nan")
            return True
        if torch.isinf(tensor).any():
            print(f"{name}存在inf")
            return True
        if tensor.sum() <= 0:
            # print(f"{name}和有误")
            return True
        '''
        if abs(tensor.sum() - 1) > 1e-5:
            print('和有误！')
        '''
        return False

    def aggregate_features(self, features, attention_net, mask):
        """
        用注意力机制聚合特征（忽略填充的无效特征）
        features: 形状 [batch_size, num_objects, hidden_dim]
        mask: 形状 [batch_size, num_objects]，1表示有效，0表示填充
        """
        # 计算注意力权重
        attn_weights = attention_net(features).squeeze(-1)  # [batch_size, num_objects]
        # 对填充部分施加负无穷掩码，使其注意力权重为0
        attn_weights = attn_weights.masked_fill(mask == 0, -1e10)
        attn_weights = F.softmax(attn_weights, dim=1)  # [batch_size, num_objects]

        # 加权聚合特征
        aggregated = torch.bmm(attn_weights.unsqueeze(1), features).squeeze(1)  # [batch_size, hidden_dim]
        return aggregated

    def forward(self, item, item_unplaced_order, space_free, device):
        # 前向传播
        batch_size = 1
        # #####################################################编码所有货物
        item_features = torch.zeros((batch_size, self.item_num_max, 3), device=device)
        valid_item = min(self.item_num_max, len(item_unplaced_order))
        # 填充有效货物特征
        for i in range(valid_item):
            item_features[0, i] = torch.tensor([item[item_unplaced_order[i]].length, item[item_unplaced_order[i]].width, item[item_unplaced_order[i]].height], device=device)
        # 生成掩码
        item_mask = torch.zeros((batch_size, self.item_num_max), device=device)
        item_mask[0, :valid_item] = 1.0
        # 编码货物
        encoded_item = self.item_encoder(item_features.view(-1, 3)).view(batch_size, self.item_num_max, self.hidden_dim)
        self.check_nan_inf(encoded_item, "encoded_item")
        # 聚合货物特征（注意力机制）
        aggregated_item = self.aggregate_features(encoded_item, self.item_attention, item_mask)
        self.check_nan_inf(aggregated_item, "aggregated_item")

        # 检查模型
        for name, param in self.named_parameters():
            if torch.isnan(param).any() or torch.isinf(param).any():
                print(f"参数 {name} 包含 NaN 或 Inf！")

        # ###################################################################编码所有空间
        space_features = -1 * torch.ones((batch_size, self.space_free_num_max, 3), device=device)
        valid_space = min(self.space_free_num_max, len(space_free))
        # 填充有效空间特征
        for i in range(valid_space):
            space_features[0, i] = torch.tensor([space_free[i].length, space_free[i].width, space_free[i].height], device=device)
        # 生成掩码
        space_mask = torch.zeros((batch_size, self.space_free_num_max), device=device)
        space_mask[0, :valid_space] = 1.0
        # 批量编码
        encoded_space = self.space_encoder(space_features.view(-1, 3)).view(batch_size, self.space_free_num_max, self.hidden_dim)
        self.check_nan_inf(encoded_space, "encoded_space")
        # 聚合空间特征（注意力机制）
        aggregated_space = self.aggregate_features(encoded_space, self.space_attention, space_mask)
        self.check_nan_inf(aggregated_space, "aggregated_space")

        # 融合特征
        fused = torch.cat([aggregated_item, aggregated_space], dim=1)  # [1, 2*hidden_dim]
        fused = self.fusion(fused)

        # 计算输出概率
        item_logits = self.item_output(fused)
        space_logits = self.space_output(fused)

        # 应用掩码并计算概率
        item_probs = F.softmax(item_logits * item_mask + (1 - item_mask) * -1e10, dim=1)
        space_probs = F.softmax(space_logits * space_mask + (1 - space_mask) * -1e10, dim=1)

        # 检查数值
        self.check_nan_inf(item_probs, "item_probs")
        self.check_nan_inf(space_probs, "space_probs")

        return item_probs, space_probs


class DqnPolicy4(nn.Module):
    def __init__(self, hidden_dim=128):
        super(DqnPolicy4, self).__init__()
        self.hidden_dim = hidden_dim

        # 货物特征提取网络
        self.item_encoder = nn.Sequential(
            nn.Linear(3, 64),  # 输入为单个货物的长宽高
            nn.LayerNorm(64),  # 归一化层
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )

        # 空间特征提取网络
        self.space_encoder = nn.Sequential(
            nn.Linear(3, 64),  # 输入为单个空间的长宽高
            nn.LayerNorm(64),  # 归一化层
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )

        # 注意力聚合层（处理货物特征，忽略填充值）
        self.item_attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),  # 隐藏层的激活函数
            nn.Linear(hidden_dim // 2, 1)  # 输出每个货物的注意力权重
        )

        # 空间注意力聚合层
        self.space_attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )

        # 融合层
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, 512),  # 货物聚合特征 + 空间聚合特征
            nn.ReLU(),
            nn.Dropout(0.2),  # 防止过拟合
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        # 动态输出层：为每个对象计算Q值的通用网络
        self.item_q_network = nn.Sequential(
            nn.Linear(256 + hidden_dim, 128),  # 融合特征 + 单个货物特征
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)  # 输出单个Q值
        )

        self.space_q_network = nn.Sequential(
            nn.Linear(256 + hidden_dim, 128),  # 融合特征 + 单个空间特征
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)  # 输出单个Q值
        )

    def check_nan_inf(self, tensor, name):
        # 检查张量是否包含NaN，inf
        if torch.isnan(tensor).any():
            print(f"{name}存在nan")
            return True
        if torch.isinf(tensor).any():
            print(f"{name}存在inf")
            return True
        if tensor.sum() <= 0:
            # print(f"{name}和有误")
            return True
        return False

    def aggregate_features(self, features, attention_net):
        """
        用注意力机制聚合特征（忽略填充的无效特征）
        features: 形状 [batch_size, num_objects, hidden_dim]
        注意：此方法不使用mask，直接对所有特征进行注意力聚合
        """
        # 计算注意力权重
        attn_weights = attention_net(features).squeeze(-1)  # [batch_size, num_objects]
        # 对填充部分施加负无穷掩码，使其注意力权重为0
        attn_weights = F.softmax(attn_weights, dim=1)  # [batch_size, num_objects]

        # 加权聚合特征
        aggregated = torch.bmm(attn_weights.unsqueeze(1), features).squeeze(1)  # [batch_size, hidden_dim]
        return aggregated

    def forward(self, item, item_unplaced_order, space_free, item_num_max, space_free_num_max, device):
        # 前向传播
        batch_size = 1

        # 动态获取实际的货物和空间数量
        actual_item_num = min(len(item_unplaced_order), item_num_max)
        actual_space_num = min(len(space_free), space_free_num_max)

        # #####################################################编码所有货物
        item_features = torch.zeros((batch_size, actual_item_num, 3), device=device)
        # 填充有效货物特征
        for i in range(actual_item_num):
            item_features[0, i] = torch.tensor([item[item_unplaced_order[i]].length, item[item_unplaced_order[i]].width, item[item_unplaced_order[i]].height], device=device)

        # 编码货物
        encoded_item = self.item_encoder(item_features.view(-1, 3)).view(batch_size, actual_item_num, self.hidden_dim)
        self.check_nan_inf(encoded_item, "encoded_item")

        # 聚合货物特征（注意力机制）
        aggregated_item = self.aggregate_features(encoded_item, self.item_attention)
        self.check_nan_inf(aggregated_item, "aggregated_item")

        # ###################################################################编码所有空间
        space_features = torch.zeros((batch_size, actual_space_num, 3), device=device)
        # 填充有效空间特征
        for i in range(actual_space_num):
            space_features[0, i] = torch.tensor([space_free[i].length, space_free[i].width, space_free[i].height], device=device)

        # 批量编码
        encoded_space = self.space_encoder(space_features.view(-1, 3)).view(batch_size, actual_space_num, self.hidden_dim)
        self.check_nan_inf(encoded_space, "encoded_space")

        # 聚合空间特征（注意力机制）
        aggregated_space = self.aggregate_features(encoded_space, self.space_attention)
        self.check_nan_inf(aggregated_space, "aggregated_space")

        # 融合特征
        fused = torch.cat([aggregated_item, aggregated_space], dim=1)  # [1, 2*hidden_dim]
        fused = self.fusion(fused)

        # 动态计算每个货物和空间的Q值
        item_q_values = []
        for i in range(actual_item_num):
            # 将融合特征与单个货物特征结合
            combined_feature = torch.cat([fused, encoded_item[0, i:i + 1]], dim=1)
            q_value = self.item_q_network(combined_feature)
            item_q_values.append(q_value)
        item_logits = torch.cat(item_q_values, dim=1)  # [1, actual_item_num]

        space_q_values = []
        for i in range(actual_space_num):
            # 将融合特征与单个空间特征结合
            combined_feature = torch.cat([fused, encoded_space[0, i:i + 1]], dim=1)
            q_value = self.space_q_network(combined_feature)
            space_q_values.append(q_value)
        space_logits = torch.cat(space_q_values, dim=1)  # [1, actual_space_num]

        # 计算概率分布
        item_probs = F.softmax(item_logits, dim=1)
        space_probs = F.softmax(space_logits, dim=1)

        # 检查数值
        self.check_nan_inf(item_probs, "item_probs")
        self.check_nan_inf(space_probs, "space_probs")

        return item_probs, space_probs


class DqnPolicy5(nn.Module):
    def __init__(self, item_num_max, space_free_num_max):
        super(DqnPolicy5, self).__init__()
        self.item_num_max = item_num_max  # 最大货物数量
        self.space_free_num_max = space_free_num_max  # 最大剩余空间数量

        # 货物特征提取网络
        self.item_encoder = nn.Sequential(
            nn.Linear(3, 64),  # 输入为单个货物的长宽高
            nn.LayerNorm(64),  # 归一化层
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.LayerNorm(128),  # 归一化层
            nn.ReLU()
        )

        # 空间特征提取网络
        self.space_encoder = nn.Sequential(
            nn.Linear(3, 64),  # 输入为单个空间的长宽高
            nn.LayerNorm(64),  # 归一化层
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.LayerNorm(128),  # 归一化层
            nn.ReLU()
        )

        # 融合层
        self.fusion = nn.Sequential(
            nn.Linear((item_num_max + space_free_num_max) * 128, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU()
        )

        # 输出层：分别输出货物选择概率和空间选择概率
        self.item_output = nn.Linear(256, item_num_max)
        self.space_output = nn.Linear(256, space_free_num_max)

    def check_nan_inf(self, tensor, name):
        # 检查张量是否包含NaN，inf
        if torch.isnan(tensor).any():
            print(f"{name}存在nan")
            return True
        if torch.isinf(tensor).any():
            print(f"{name}存在inf")
            return True
        if tensor.sum() <= 0:
            print(f"{name}和有误")
            return True
        return False

    def forward(self, item, item_unplaced_order, space_free, device, space_size=None, space_state=None):
        from can_place_item import can_place_item
        # 前向传播
        # 编码所有货物
        item_features = torch.zeros((self.item_num_max, 3), device=device)
        valid_item = min(self.item_num_max, len(item_unplaced_order))
        for i in range(valid_item):
            item_features[i] = torch.tensor([item[item_unplaced_order[i]].length, item[item_unplaced_order[i]].width, item[item_unplaced_order[i]].height], device=device)
        encoded_item = self.item_encoder(item_features.view(-1, 3))
        encoded_item = encoded_item.view(1, -1)
        self.check_nan_inf(encoded_item, "encoded_item")

        # 检查模型
        for name, param in self.named_parameters():
            if torch.isnan(param).any() or torch.isinf(param).any():
                print(f"参数 {name} 包含 NaN 或 Inf！")

        # 编码所有空间
        space_features = torch.zeros((self.space_free_num_max, 3), device=device)
        valid_space = min(self.space_free_num_max, len(space_free))
        for i in range(valid_space):
            space_features[i] = torch.tensor([space_free[i].length, space_free[i].width, space_free[i].height], device=device)
        encoded_space = self.space_encoder(space_features.view(-1, 3))
        encoded_space = encoded_space.view(1, -1)
        self.check_nan_inf(encoded_space, "encoded_space")

        # 融合特征
        fused = torch.cat([encoded_item, encoded_space], dim=1)
        fused = self.fusion(fused)

        # 计算输出logits
        item_logits = self.item_output(fused)
        space_logits = self.space_output(fused)

        # 创建初始掩码
        item_mask = torch.zeros(item_logits.size(1), device=device)
        item_mask[:valid_item] = 1
        space_mask = torch.zeros(space_logits.size(1), device=device)
        space_mask[:valid_space] = 1

        # 构建兼容性矩阵
        compatibility_mask = torch.zeros((valid_item, valid_space), device=device)
        for i in range(valid_item):
            item_id = item_unplaced_order[i]
            for j in range(valid_space):
                can_place, _ = can_place_item(space_size, item, item_id, space_free, j, space_state)
                compatibility_mask[i, j] = 1 if can_place else 0

        # 计算联合概率分布
        joint_logits = torch.zeros((valid_item, valid_space), device=device)
        for i in range(valid_item):
            for j in range(valid_space):
                if compatibility_mask[i, j] == 1:
                    joint_logits[i, j] = item_logits[0, i] + space_logits[0, j]
                else:
                    joint_logits[i, j] = -1e10  # 不可行动作置为负无穷

        # 展平联合概率分布
        joint_probs = F.softmax(joint_logits.view(-1), dim=0)
        joint_probs = joint_probs.view(valid_item, valid_space)

        # 计算边际概率
        item_probs = joint_probs.sum(dim=1)  # 货物选择概率
        space_probs = joint_probs.sum(dim=0)  # 空间选择概率

        # 扩展到最大维度以兼容train1.py
        item_probs_full = torch.zeros((1, self.item_num_max), device=device)
        space_probs_full = torch.zeros((1, self.space_free_num_max), device=device)
        item_probs_full[0, :valid_item] = item_probs
        space_probs_full[0, :valid_space] = space_probs

        # 确保概率和为1
        item_probs_full = item_probs_full / (item_probs_full.sum() + 1e-10)
        space_probs_full = space_probs_full / (space_probs_full.sum() + 1e-10)

        # 检查数值
        self.check_nan_inf(item_probs_full, "item_probs")
        self.check_nan_inf(space_probs_full, "space_probs")

        return item_probs_full, space_probs_full


class DqnPolicy6Legacy(nn.Module):
    def __init__(self, item_num_max, space_free_num_max):
        super(DqnPolicy6Legacy, self).__init__()
        self.item_num_max = item_num_max
        self.space_free_num_max = space_free_num_max

        # 货物特征提取网络
        self.item_encoder = nn.Sequential(
            nn.Linear(7, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.LayerNorm(128),
            nn.ReLU()
        )

        # 空间特征提取网络（改进：添加坐标与几何/接触特征）
        # 输入特征：坐标、尺度、支撑、体积、基底面积、最长边、对角线与边界接触、卷积掩码统计（20维）
        self.space_encoder = nn.Sequential(
            nn.Linear(20, 64),  # 融合坐标、尺度、支撑特征与几何接触统计
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.LayerNorm(128),
            nn.ReLU()
        )

        # 参照 Online-3D-BPP-DRL 中 CNN 基干网络，构建共享卷积编码器
        self.state_slice_channels = 12
        self.state_channel_count = self.state_slice_channels + 3  # 多层占用切片 + 下一件货物三轴尺度
        self.state_share = nn.Sequential(
            nn.Conv2d(self.state_channel_count, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )
        self.state_global_pool = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.state_context_proj = nn.Sequential(
            nn.Linear(32, 128),
            nn.LayerNorm(128),
            nn.ReLU(inplace=True),
        )
        self.state_mask_head = nn.Sequential(
            nn.Conv2d(64, 16, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=1),
        )

        # 全局几何特征编码（统计空间整体利用情况、层填充度与未放置货物统计）
        self.global_encoder = nn.Sequential(
            nn.Linear(26, 128),
            nn.LayerNorm(128),
            nn.ReLU()
        )

        # 汇聚货物与空间分布统计，减轻 item_num_max 扩大的维度爆炸
        self.item_summary_proj = nn.Sequential(
            nn.Linear(128 * 4, 128),
            nn.LayerNorm(128),
            nn.ReLU()
        )
        self.space_summary_proj = nn.Sequential(
            nn.Linear(128 * 4, 128),
            nn.LayerNorm(128),
            nn.ReLU()
        )

        # 融合上下文向量
        self.context_proj = nn.Sequential(
            nn.Linear(128 * 4, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        # 局部关系建模（借鉴 Online-3D-BPP-DRL 的上下文自注意力聚合）
        # 添加位置编码，使模型能够区分不同位置的item/space
        self.item_pos_encoding = nn.Parameter(torch.randn(item_num_max, 128) * 0.02)
        self.space_pos_encoding = nn.Parameter(torch.randn(space_free_num_max, 128) * 0.02)
        self.item_cls_token = nn.Parameter(torch.randn(1, 1, 128) * 0.02)
        self.space_cls_token = nn.Parameter(torch.randn(1, 1, 128) * 0.02)
        self.item_self_attn = nn.MultiheadAttention(128, 4, dropout=0.1)
        self.space_self_attn = nn.MultiheadAttention(128, 4, dropout=0.1)
        self.item_ffn = nn.Sequential(
            nn.LayerNorm(128),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
        )
        self.space_ffn = nn.Sequential(
            nn.LayerNorm(128),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
        )

        # 货物-空间交互注意力（参考 Online-3D-BPP-DRL 的双向上下文聚合）
        self.item_cross_attn = nn.MultiheadAttention(128, 4, dropout=0.1)
        self.space_cross_attn = nn.MultiheadAttention(128, 4, dropout=0.1)
        self.item_cross_ln = nn.LayerNorm(128)
        self.space_cross_ln = nn.LayerNorm(128)
        self.item_cross_gate = nn.Sequential(
            nn.Linear(128 * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.Sigmoid(),
        )
        self.space_cross_gate = nn.Sequential(
            nn.Linear(128 * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.Sigmoid(),
        )

        # 项-空间兼容性偏置（减少 item_num_max 扩张时的分布稀释）
        self.item_pair_proj = nn.Sequential(
            nn.LayerNorm(128),
            nn.Linear(128, 64),
        )
        self.space_pair_proj = nn.Sequential(
            nn.LayerNorm(128),
            nn.Linear(128, 64),
        )
        self.pair_scale = nn.Parameter(torch.tensor(1.0))
        self.pair_fusion = nn.Sequential(
            nn.LayerNorm(64 * 4),
            nn.Linear(64 * 4, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1),
        )
        self.pair_temperature = nn.Parameter(torch.tensor(1.0))

        # 针对单个货物/空间的打分头（共享上下文，避免输出层尺寸随 item_num_max 线性膨胀）
        self.item_head = nn.Sequential(
            nn.Linear(128 + 256, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1)
        )
        self.space_head = nn.Sequential(
            nn.Linear(128 + 256, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1)
        )

        # 旋转输出（2类：只允许长宽互换，高度保持不变）
        self.rotation_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )

    def check_nan_inf(self, tensor, name):
        if torch.isnan(tensor).any():
            print(f"{name}存在nan")
            return True
        if torch.isinf(tensor).any():
            print(f"{name}存在inf")
            return True
        return False

    def forward(self, item, item_unplaced_order, space_free, device, space_size=None, space_state=None):
        from can_place_item import can_place_item

        # 统一容器尺寸信息（用于特征归一化）
        if space_size is not None:
            container_dims = (
                float(space_size[0]),
                float(space_size[1]),
                float(space_size[2]),
            )
        elif space_state is not None:
            state_shape = np.asarray(space_state).shape
            if len(state_shape) == 3:
                container_dims = (
                    float(state_shape[0]),
                    float(state_shape[1]),
                    float(state_shape[2]),
                )
            else:
                container_dims = (1.0, 1.0, 1.0)
        else:
            container_dims = (1.0, 1.0, 1.0)

        container_volume = max(container_dims[0] * container_dims[1] * container_dims[2], 1.0)
        container_plane_area = max(container_dims[0] * container_dims[1], 1.0)
        denom_x = max(container_dims[0], 1.0)
        denom_y = max(container_dims[1], 1.0)
        denom_z = max(container_dims[2], 1.0)
        max_span = max(denom_x, denom_y, denom_z, 1.0)
        container_diagonal = math.sqrt(
            container_dims[0] ** 2 + container_dims[1] ** 2 + container_dims[2] ** 2
        )
        denom_diag = max(container_diagonal, 1.0)

        space_state_arr = np.asarray(space_state) if space_state is not None else None

        # === 参考 Online-3D-BPP-DRL CNN 结构：构建高度图 + 下一货物尺寸的卷积表征 ===
        if space_state_arr is not None and space_state_arr.ndim == 3:
            map_width = int(space_state_arr.shape[0])
            map_length = int(space_state_arr.shape[1])
        else:
            map_width = max(int(round(container_dims[0])), 1)
            map_length = max(int(round(container_dims[1])), 1)

        state_tensor = torch.zeros(
            (1, self.state_channel_count, map_width, map_length),
            dtype=torch.float32,
            device=device,
        )

        if (
            space_state_arr is not None
            and space_state_arr.size > 0
            and space_state_arr.ndim == 3
        ):
            occupancy = np.clip(space_state_arr.astype(np.float32), 0.0, 1.0)
            z_dim = occupancy.shape[2]
            if z_dim > 0:
                for ch in range(self.state_slice_channels):
                    start_idx = int(np.floor(ch * z_dim / self.state_slice_channels))
                    end_idx = int(np.floor((ch + 1) * z_dim / self.state_slice_channels))
                    if ch == self.state_slice_channels - 1:
                        end_idx = z_dim
                    if end_idx <= start_idx:
                        slice_map = np.zeros((map_width, map_length), dtype=np.float32)
                    else:
                        slice_map = occupancy[:, :, start_idx:end_idx].mean(axis=2)
                    slice_tensor = torch.from_numpy(slice_map).to(
                        device=device, dtype=torch.float32
                    )
                    state_tensor[0, ch, :map_width, :map_length] = torch.clamp(
                        slice_tensor, 0.0, 1.0
                    )

        primary_length_norm = 0.0
        primary_width_norm = 0.0
        primary_height_norm = 0.0
        if len(item_unplaced_order) > 0:
            first_item = item[item_unplaced_order[0]]
            primary_length_norm = float(np.clip(float(first_item.length) / denom_x, 0.0, 1.0))
            primary_width_norm = float(np.clip(float(first_item.width) / denom_y, 0.0, 1.0))
            primary_height_norm = float(np.clip(float(first_item.height) / denom_z, 0.0, 1.0))

        state_tensor[0, self.state_slice_channels + 0].fill_(primary_length_norm)
        state_tensor[0, self.state_slice_channels + 1].fill_(primary_width_norm)
        state_tensor[0, self.state_slice_channels + 2].fill_(primary_height_norm)

        state_feat = self.state_share(state_tensor)
        state_global = self.state_global_pool(state_feat).view(1, -1)
        state_context = self.state_context_proj(state_global)
        state_mask_map = torch.sigmoid(self.state_mask_head(state_feat))
        state_mask_map_np = state_mask_map.detach().cpu().numpy()[0, 0] if state_mask_map.numel() > 0 else None

        def pad_to_max(tensor_2d, target_len, ref_tensor):
            hidden_dim = ref_tensor.size(-1)
            current_len = tensor_2d.size(0)
            if current_len == 0:
                return ref_tensor.new_zeros((target_len, hidden_dim))
            if current_len < target_len:
                pad = ref_tensor.new_zeros((target_len - current_len, hidden_dim))
                return torch.cat([tensor_2d, pad], dim=0)
            return tensor_2d

        # 编码货物（加入归一化尺度与体积特征）
        # 使用-1作为padding标记，而不是0，使模型更容易区分有效和无效位置
        item_features = torch.full((self.item_num_max, 7), -1.0, device=device)
        valid_item = min(self.item_num_max, len(item_unplaced_order))
        for i in range(valid_item):
            cur_item = item[item_unplaced_order[i]]
            length = float(cur_item.length)
            width = float(cur_item.width)
            height = float(cur_item.height)
            volume = float(cur_item.volume)
            longest_edge = max(length, width, height)
            base_area = length * width
            diag = math.sqrt(length ** 2 + width ** 2 + height ** 2)
            item_features[i] = torch.tensor(
                [
                    min(length / denom_x, 1.0),
                    min(width / denom_y, 1.0),
                    min(height / denom_z, 1.0),
                    min(volume / container_volume, 1.0),
                    min(longest_edge / max_span, 1.0),
                    min(base_area / container_plane_area, 1.0),
                    min(diag / denom_diag, 1.0),
                ],
                dtype=torch.float32,
                device=device,
            )
        encoded_item_raw = self.item_encoder(item_features)
        if valid_item > 0:
            encoded_item_with_pos = encoded_item_raw[:valid_item] + self.item_pos_encoding[:valid_item]
            item_attn_input = torch.cat(
                [self.item_cls_token, encoded_item_with_pos.unsqueeze(1)],
                dim=0
            )
        else:
            item_attn_input = self.item_cls_token
        item_attn_out, _ = self.item_self_attn(
            item_attn_input, item_attn_input, item_attn_input, key_padding_mask=None
        )
        item_attn_out = item_attn_out + item_attn_input
        item_ffn_out = self.item_ffn(item_attn_out)
        item_attn_out = item_attn_out + item_ffn_out
        item_attn_out = item_attn_out.squeeze(1)
        item_cls_context = item_attn_out[0:1]
        if valid_item > 0:
            processed_item = item_attn_out[1:valid_item + 1]
        else:
            processed_item = item_attn_out.new_zeros((0, item_attn_out.size(-1)))
        encoded_item = pad_to_max(processed_item, self.item_num_max, encoded_item_raw)

        # 编码空间（改进：添加坐标、支撑与体积特征，并归一化）
        # 使用-1作为padding标记，而不是0
        space_features = torch.full((self.space_free_num_max, 20), -1.0, device=device)
        valid_space = min(self.space_free_num_max, len(space_free))
        norm_x = 1.0 / denom_x
        norm_y = 1.0 / denom_y
        norm_z = 1.0 / denom_z
        norm_length = 1.0 / denom_x
        norm_width = 1.0 / denom_y
        norm_height = 1.0 / denom_z
        support_values = []
        corner_support_values = []
        space_volume_ratios = []
        floor_touch_count = 0
        wall_touch_count = 0
        ceiling_touch_count = 0

        mask_width = state_mask_map_np.shape[0] if state_mask_map_np is not None else 0
        mask_length = state_mask_map_np.shape[1] if state_mask_map_np is not None else 0

        for i in range(valid_space):
            sf = space_free[i]
            support_ratio = 1.0 if sf.z == 0 else 0.0
            corner_support_ratio = 1.0 if sf.z == 0 else 0.0

            if space_state_arr is not None and sf.z > 0:
                x0 = int(sf.x)
                y0 = int(sf.y)
                z0 = int(sf.z)
                L = max(int(round(sf.length)), 0)
                W = max(int(round(sf.width)), 0)
                if (
                    z0 - 1 >= 0
                    and x0 < space_state_arr.shape[0]
                    and y0 < space_state_arr.shape[1]
                    and z0 - 1 < space_state_arr.shape[2]
                ):
                    x1 = min(x0 + L, space_state_arr.shape[0])
                    y1 = min(y0 + W, space_state_arr.shape[1])
                    if x1 > x0 and y1 > y0:
                        support_slice = space_state_arr[x0:x1, y0:y1, z0 - 1]
                        area = max((x1 - x0) * (y1 - y0), 1)
                        support_ratio = float(support_slice.sum()) / area
                        support_ratio = float(np.clip(support_ratio, 0.0, 1.0))
                        if support_slice.size > 0:
                            corner_values = []
                            if support_slice.shape[0] >= 1 and support_slice.shape[1] >= 1:
                                corner_values.append(float(support_slice[0, 0]))
                                corner_values.append(float(support_slice[-1, 0]))
                                corner_values.append(float(support_slice[0, -1]))
                                corner_values.append(float(support_slice[-1, -1]))
                            if corner_values:
                                corner_support_ratio = float(np.mean(corner_values))
                                corner_support_ratio = float(np.clip(corner_support_ratio, 0.0, 1.0))
                            else:
                                corner_support_ratio = support_ratio
                    else:
                        support_ratio = float(np.clip(support_ratio, 0.0, 1.0))
                        corner_support_ratio = support_ratio
                else:
                    support_ratio = float(np.clip(support_ratio, 0.0, 1.0))
                    corner_support_ratio = support_ratio

            volume_ratio = min(float(sf.volume) / container_volume, 1.0)
            base_area_ratio = min(float(sf.length * sf.width) / container_plane_area, 1.0)
            longest_edge_ratio = min(float(max(sf.length, sf.width, sf.height)) / max_span, 1.0)
            diagonal = math.sqrt(float(sf.length) ** 2 + float(sf.width) ** 2 + float(sf.height) ** 2)
            diagonal_ratio = min(diagonal / denom_diag, 1.0)

            touch_floor = 1.0 if float(sf.z) <= 0.0 else 0.0
            touch_ceiling = 1.0 if float(sf.z + sf.height) >= denom_z else 0.0
            touch_wall_x_min = 1.0 if float(sf.x) <= 0.0 else 0.0
            touch_wall_x_max = 1.0 if float(sf.x + sf.length) >= denom_x else 0.0
            touch_wall_y_min = 1.0 if float(sf.y) <= 0.0 else 0.0
            touch_wall_y_max = 1.0 if float(sf.y + sf.width) >= denom_y else 0.0

            support_values.append(support_ratio)
            corner_support_values.append(corner_support_ratio)
            space_volume_ratios.append(volume_ratio)
            if touch_floor > 0.0:
                floor_touch_count += 1
            if touch_wall_x_min > 0.0 or touch_wall_x_max > 0.0 or touch_wall_y_min > 0.0 or touch_wall_y_max > 0.0:
                wall_touch_count += 1
            if touch_ceiling > 0.0:
                ceiling_touch_count += 1

            space_features[i] = torch.tensor(
                [
                    float(np.clip(float(sf.x) * norm_x, 0.0, 1.0)),
                    float(np.clip(float(sf.y) * norm_y, 0.0, 1.0)),
                    float(np.clip(float(sf.z) * norm_z, 0.0, 1.0)),
                    float(np.clip(float(sf.length) * norm_length, 0.0, 1.0)),
                    float(np.clip(float(sf.width) * norm_width, 0.0, 1.0)),
                    float(np.clip(float(sf.height) * norm_height, 0.0, 1.0)),
                    support_ratio,
                    corner_support_ratio,
                    volume_ratio,
                    base_area_ratio,
                    longest_edge_ratio,
                    diagonal_ratio,
                    touch_floor,
                    touch_ceiling,
                    touch_wall_x_min,
                    touch_wall_x_max,
                    touch_wall_y_min,
                    touch_wall_y_max,
                    0.0,
                    0.0,
                ],
                dtype=torch.float32,
                device=device,
            )

            if state_mask_map_np is not None and mask_width > 0 and mask_length > 0:
                x0 = int(np.clip(sf.x, 0, mask_width))
                y0 = int(np.clip(sf.y, 0, mask_length))
                x1 = int(np.clip(sf.x + sf.length, 0, mask_width))
                y1 = int(np.clip(sf.y + sf.width, 0, mask_length))
                if x1 > x0 and y1 > y0:
                    region = state_mask_map_np[x0:x1, y0:y1]
                    if region.size > 0:
                        mask_mean = float(np.clip(region.mean(), 0.0, 1.0))
                        mask_max = float(np.clip(region.max(), 0.0, 1.0))
                        space_features[i, 18] = mask_mean
                        space_features[i, 19] = mask_max
        encoded_space_raw = self.space_encoder(space_features)
        if valid_space > 0:
            encoded_space_with_pos = encoded_space_raw[:valid_space] + self.space_pos_encoding[:valid_space]
            space_attn_input = torch.cat(
                [self.space_cls_token, encoded_space_with_pos.unsqueeze(1)],
                dim=0
            )
        else:
            space_attn_input = self.space_cls_token
        space_attn_out, _ = self.space_self_attn(
            space_attn_input, space_attn_input, space_attn_input, key_padding_mask=None
        )
        space_attn_out = space_attn_out + space_attn_input
        space_ffn_out = self.space_ffn(space_attn_out)
        space_attn_out = space_attn_out + space_ffn_out
        space_attn_out = space_attn_out.squeeze(1)
        space_cls_context = space_attn_out[0:1]
        if valid_space > 0:
            processed_space = space_attn_out[1:valid_space + 1]
        else:
            processed_space = space_attn_out.new_zeros((0, space_attn_out.size(-1)))
        encoded_space = pad_to_max(processed_space, self.space_free_num_max, encoded_space_raw)

        if valid_item > 0 and valid_space > 0:
            # MultiheadAttention期望格式: [seq_len, batch_size, embed_dim]
            item_ctx_input = encoded_item[:valid_item].unsqueeze(1)  # [valid_item, 1, 128]
            space_ctx_input = encoded_space[:valid_space].unsqueeze(1)  # [valid_space, 1, 128]
            # 交叉注意力：item关注space，space关注item
            item_cross_out, _ = self.item_cross_attn(
                item_ctx_input, space_ctx_input, space_ctx_input,
                key_padding_mask=None  # space都是有效的
            )
            space_cross_out, _ = self.space_cross_attn(
                space_ctx_input, item_ctx_input, item_ctx_input,
                key_padding_mask=None  # item都是有效的
            )
            item_cross_norm = self.item_cross_ln(item_cross_out + item_ctx_input)
            space_cross_norm = self.space_cross_ln(space_cross_out + space_ctx_input)
            item_gate = self.item_cross_gate(
                torch.cat([item_ctx_input, item_cross_norm], dim=-1)
            )
            space_gate = self.space_cross_gate(
                torch.cat([space_ctx_input, space_cross_norm], dim=-1)
            )
            fused_item = item_gate * item_cross_norm + (1.0 - item_gate) * item_ctx_input
            fused_space = space_gate * space_cross_norm + (1.0 - space_gate) * space_ctx_input
            processed_item = fused_item.squeeze(1)
            processed_space = fused_space.squeeze(1)
            encoded_item = pad_to_max(processed_item, self.item_num_max, encoded_item)
            encoded_space = pad_to_max(processed_space, self.space_free_num_max, encoded_space)

        # 全局特征编码（全局占用率、层填充统计、未放置货物统计）
        if space_state_arr is not None:
            occupancy = float(space_state_arr.sum())
            if space_state_arr.ndim == 3 and space_state_arr.shape[2] > 0:
                layer_denominator = max(container_dims[0] * container_dims[1], 1.0)
                layer_fill_np = space_state_arr.sum(axis=(0, 1)) / layer_denominator
            else:
                z_dim = max(int(round(container_dims[2])), 1)
                layer_fill_np = np.zeros((z_dim,), dtype=np.float32)
        else:
            occupancy = 0.0
            z_dim = max(int(round(container_dims[2])), 1)
            layer_fill_np = np.zeros((z_dim,), dtype=np.float32)

        layer_fill_np = np.clip(layer_fill_np, 0.0, 1.0)
        if layer_fill_np.size == 0:
            mean_layer_fill = 0.0
            max_layer_fill = 0.0
            min_layer_fill = 0.0
            std_layer_fill = 0.0
        else:
            mean_layer_fill = float(layer_fill_np.mean())
            max_layer_fill = float(layer_fill_np.max())
            min_layer_fill = float(layer_fill_np.min())
            std_layer_fill = float(layer_fill_np.std())

        fill_ratio = occupancy / container_volume
        num_spaces_norm = float(valid_space) / max(self.space_free_num_max, 1)
        if space_volume_ratios:
            mean_space_volume = float(np.mean(space_volume_ratios))
            max_space_volume = float(np.max(space_volume_ratios))
            std_space_volume = float(np.std(space_volume_ratios))
        else:
            mean_space_volume = 0.0
            max_space_volume = 0.0
            std_space_volume = 0.0

        unplaced_volume_ratios = [
            min(float(item[idx].volume) / container_volume, 1.0)
            for idx in item_unplaced_order
        ]
        if unplaced_volume_ratios:
            mean_unplaced_volume_ratio = float(np.mean(unplaced_volume_ratios))
            max_unplaced_volume_ratio = float(np.max(unplaced_volume_ratios))
            min_unplaced_volume_ratio = float(np.min(unplaced_volume_ratios))
            std_unplaced_volume_ratio = float(np.std(unplaced_volume_ratios))
        else:
            mean_unplaced_volume_ratio = 0.0
            max_unplaced_volume_ratio = 0.0
            min_unplaced_volume_ratio = 0.0
            std_unplaced_volume_ratio = 0.0

        support_mean = float(np.mean(support_values)) if support_values else 0.0
        support_max = float(np.max(support_values)) if support_values else 0.0
        corner_mean = float(np.mean(corner_support_values)) if corner_support_values else 0.0
        corner_max = float(np.max(corner_support_values)) if corner_support_values else 0.0

        placed_items = [
            itm
            for itm in item
            if getattr(itm, "placed", False)
            and itm.x is not None
            and itm.y is not None
            and itm.z is not None
        ]
        total_volume = sum(float(itm.volume) for itm in placed_items)
        if total_volume > 0:
            com_x = sum((float(itm.x) + float(itm.length) / 2.0) * float(itm.volume) for itm in placed_items) / total_volume
            com_y = sum((float(itm.y) + float(itm.width) / 2.0) * float(itm.volume) for itm in placed_items) / total_volume
            com_z = sum((float(itm.z) + float(itm.height) / 2.0) * float(itm.volume) for itm in placed_items) / total_volume
            center_mass_x = float(np.clip(com_x / denom_x, 0.0, 1.0))
            center_mass_y = float(np.clip(com_y / denom_y, 0.0, 1.0))
            center_mass_z = float(np.clip(com_z / denom_z, 0.0, 1.0))
        else:
            center_mass_x = 0.0
            center_mass_y = 0.0
            center_mass_z = 0.0

        placed_count_ratio = float(sum(1 for itm in item if getattr(itm, "placed", False))) / max(len(item), 1)
        floor_touch_ratio = float(floor_touch_count) / max(valid_space, 1)
        wall_touch_ratio = float(wall_touch_count) / max(valid_space, 1)
        ceiling_touch_ratio = float(ceiling_touch_count) / max(valid_space, 1)

        global_vector = torch.tensor(
            [
                fill_ratio,
                mean_layer_fill,
                max_layer_fill,
                min_layer_fill,
                std_layer_fill,
                num_spaces_norm,
                mean_space_volume,
                max_space_volume,
                mean_unplaced_volume_ratio,
                max_unplaced_volume_ratio,
                min_unplaced_volume_ratio,
                std_unplaced_volume_ratio,
                support_mean,
                support_max,
                corner_mean,
                corner_max,
                std_space_volume,
                center_mass_x,
                center_mass_y,
                center_mass_z,
                placed_count_ratio,
                floor_touch_ratio,
                wall_touch_ratio,
                ceiling_touch_ratio,
                float(valid_item) / max(self.item_num_max, 1),
                float(len(item_unplaced_order)) / max(len(item), 1),
            ],
            dtype=torch.float32,
            device=device,
        )
        global_vector = torch.clamp(global_vector, 0.0, 1.0)
        global_encoded = self.global_encoder(global_vector.view(1, -1))

        def summarize_embeddings(embeddings, valid_len, cls_vector):
            hidden_dim = cls_vector.size(-1)
            cls_feat = cls_vector.view(-1)
            if valid_len > 0:
                valid_embeddings = embeddings[:valid_len]
                mean_feat = valid_embeddings.mean(dim=0)
                max_feat = valid_embeddings.max(dim=0).values
                std_feat = valid_embeddings.std(dim=0, unbiased=False)
            else:
                zero_feat = cls_vector.new_zeros(hidden_dim)
                mean_feat = zero_feat
                max_feat = zero_feat
                std_feat = zero_feat
            return torch.cat([cls_feat, mean_feat, max_feat, std_feat], dim=0)

        item_summary_vec = summarize_embeddings(processed_item, valid_item, item_cls_context)
        space_summary_vec = summarize_embeddings(processed_space, valid_space, space_cls_context)

        item_summary = self.item_summary_proj(item_summary_vec.view(1, -1))
        space_summary = self.space_summary_proj(space_summary_vec.view(1, -1))

        context_input = torch.cat([item_summary, space_summary, global_encoded, state_context], dim=1)
        context = self.context_proj(context_input)

        context_vec = context.squeeze(0)

        pair_bias = None
        pair_logits_extra = None
        if valid_item > 0 and valid_space > 0:
            item_pair_features = self.item_pair_proj(encoded_item[:valid_item])
            space_pair_features = self.space_pair_proj(encoded_space[:valid_space])
            item_pair_features = F.normalize(item_pair_features, dim=-1)
            space_pair_features = F.normalize(space_pair_features, dim=-1)
            pair_bias = torch.matmul(item_pair_features, space_pair_features.transpose(0, 1))
            pair_bias = pair_bias * self.pair_scale
            pair_bias = torch.clamp(pair_bias, -2.0, 2.0)
            item_expand = item_pair_features.unsqueeze(1).expand(-1, valid_space, -1)
            space_expand = space_pair_features.unsqueeze(0).expand(valid_item, -1, -1)
            pair_diff = item_expand - space_expand
            pair_mul = item_expand * space_expand
            pair_cat = torch.cat([item_expand, space_expand, pair_diff, pair_mul], dim=-1)
            pair_logits_extra = self.pair_fusion(pair_cat).squeeze(-1)
            pair_logits_extra = torch.tanh(pair_logits_extra)
            temp = F.softplus(self.pair_temperature) + 1e-6
            pair_logits_extra = torch.clamp(pair_logits_extra * temp, -4.0, 4.0)
        else:
            pair_bias = None
            pair_logits_extra = None

        # 改进：只对有效位置计算logits，避免padding位置影响
        if self.item_num_max > 0:
            if valid_item > 0:
                # 只对有效item计算logits
                item_context_valid = context_vec.unsqueeze(0).expand(valid_item, -1)
                item_head_input_valid = torch.cat([encoded_item[:valid_item], item_context_valid], dim=1)
                item_logits_valid = self.item_head(item_head_input_valid).view(valid_item)
                item_logits_all = torch.full((1, self.item_num_max), -1e9, device=device)
                item_logits_all[0, :valid_item] = item_logits_valid
            else:
                item_logits_all = torch.full((1, self.item_num_max), -1e9, device=device)
        else:
            item_logits_all = torch.zeros((1, self.item_num_max), device=device)

        if self.space_free_num_max > 0:
            if valid_space > 0:
                # 只对有效space计算logits
                space_context_valid = context_vec.unsqueeze(0).expand(valid_space, -1)
                space_head_input_valid = torch.cat([encoded_space[:valid_space], space_context_valid], dim=1)
                space_logits_valid = self.space_head(space_head_input_valid).view(valid_space)
                space_logits_all = torch.full((1, self.space_free_num_max), -1e9, device=device)
                space_logits_all[0, :valid_space] = space_logits_valid
            else:
                space_logits_all = torch.full((1, self.space_free_num_max), -1e9, device=device)
        else:
            space_logits_all = torch.zeros((1, self.space_free_num_max), device=device)

        # 已经在上面处理了padding掩码，这里不需要再次设置

        rotation_logits = self.rotation_head(context)
        rotation_count = rotation_logits.shape[1]
        item_has_feasible = torch.zeros(self.item_num_max, dtype=torch.bool, device=device)
        space_has_feasible = torch.zeros(self.space_free_num_max, dtype=torch.bool, device=device)
        rotation_has_feasible = torch.zeros(rotation_count, dtype=torch.bool, device=device)

        # 预分配兼容性掩码与对应旋转结果
        feasibility_mask = torch.zeros(
            (self.item_num_max, self.space_free_num_max, rotation_count),
            dtype=torch.bool,
            device=device,
        )
        rotation_dims = torch.zeros(
            (self.item_num_max, self.space_free_num_max, rotation_count, 3),
            dtype=torch.int32,
            device=device,
        )

        # 计算兼容性（仅对有效索引）
        for i in range(valid_item):
            item_id = item_unplaced_order[i]
            for j in range(valid_space):
                for rot in range(rotation_count):
                    can_place, rotation_candidate = can_place_item(
                        space_size,
                        item,
                        item_id,
                        space_free,
                        j,
                        space_state,
                        restrict_lw=False,
                        rotation_action=rot,
                    )
                    if can_place and rotation_candidate is not None:
                        feasibility_mask[i, j, rot] = True
                        rotation_dims[i, j, rot, 0] = int(rotation_candidate[0])
                        rotation_dims[i, j, rot, 1] = int(rotation_candidate[1])
                        rotation_dims[i, j, rot, 2] = int(rotation_candidate[2])

        feasible_slice = feasibility_mask[:valid_item, :valid_space, :rotation_count]
        if valid_item > 0 and valid_space > 0 and rotation_count > 0:
            item_has_feasible[:valid_item] = feasible_slice.any(dim=2).any(dim=1)
            space_has_feasible[:valid_space] = feasible_slice.any(dim=2).any(dim=0)
            rotation_has_feasible[:rotation_count] = feasible_slice.any(dim=1).any(dim=0)

        if valid_item > 0 and item_has_feasible[:valid_item].any():
            infeasible_items = ~item_has_feasible[:valid_item]
            if infeasible_items.any():
                item_logits_all = item_logits_all.clone()
                masked_slice = item_logits_all[0, :valid_item]
                masked_slice = masked_slice.masked_fill(infeasible_items, -1e9)
                item_logits_all[0, :valid_item] = masked_slice
        if valid_space > 0 and space_has_feasible[:valid_space].any():
            infeasible_spaces = ~space_has_feasible[:valid_space]
            if infeasible_spaces.any():
                space_logits_all = space_logits_all.clone()
                masked_space_slice = space_logits_all[0, :valid_space]
                masked_space_slice = masked_space_slice.masked_fill(infeasible_spaces, -1e9)
                space_logits_all[0, :valid_space] = masked_space_slice
        if rotation_count > 0 and rotation_has_feasible[:rotation_count].any():
            infeasible_rots = ~rotation_has_feasible[:rotation_count]
            if infeasible_rots.any():
                rotation_logits = rotation_logits.clone()
                masked_rot_slice = rotation_logits[0, :rotation_count]
                masked_rot_slice = masked_rot_slice.masked_fill(infeasible_rots, -1e9)
                rotation_logits[0, :rotation_count] = masked_rot_slice

        def masked_softmax(logits_tensor, valid_len, active_mask=None):
            if valid_len <= 0:
                return torch.zeros_like(logits_tensor)
            mask = torch.zeros_like(logits_tensor)
            mask[0, :valid_len] = 1
            if active_mask is not None:
                active_mask = active_mask.to(torch.bool)
                mask_slice = mask[0, :valid_len]
                mask_slice[~active_mask] = 0
                mask[0, :valid_len] = mask_slice
            if mask[0, :valid_len].sum() == 0:
                result = torch.zeros_like(logits_tensor)
                if valid_len > 0:
                    fallback = F.softmax(logits_tensor[0, :valid_len], dim=0)
                    result[0, :valid_len] = fallback
                return result
            masked_logits = logits_tensor.masked_fill(mask == 0, -1e9)
            return F.softmax(masked_logits, dim=1)

        item_probs_full = torch.zeros((1, self.item_num_max), device=device)
        space_probs_full = torch.zeros((1, self.space_free_num_max), device=device)
        rotation_probs_full = torch.zeros((1, rotation_count), device=device)
        if valid_item > 0 and valid_space > 0 and rotation_count > 0 and feasible_slice.any():
            base_item = item_logits_all[0, :valid_item].view(valid_item, 1, 1)
            base_space = space_logits_all[0, :valid_space].view(1, valid_space, 1)
            base_rot = rotation_logits[0, :rotation_count].view(1, 1, rotation_count)

            joint_logits = base_item + base_space + base_rot
            if pair_bias is not None:
                joint_logits = joint_logits + pair_bias.unsqueeze(-1)
            if pair_logits_extra is not None:
                joint_logits = joint_logits + pair_logits_extra.unsqueeze(-1)
            joint_logits = joint_logits.masked_fill(~feasible_slice, -1e9)

            joint_probs = F.softmax(joint_logits.view(-1), dim=0).view(
                valid_item, valid_space, rotation_count
            )

            item_probs_vec = joint_probs.sum(dim=(1, 2))
            space_probs_vec = joint_probs.sum(dim=(0, 2))
            rotation_probs_vec = joint_probs.sum(dim=(0, 1))

            item_probs_full[0, :valid_item] = item_probs_vec
            space_probs_full[0, :valid_space] = space_probs_vec
            rotation_probs_full[0, :rotation_count] = rotation_probs_vec
        else:
            if pair_bias is not None or pair_logits_extra is not None:
                item_logits_all = item_logits_all.clone()
                space_logits_all = space_logits_all.clone()
            if pair_bias is not None:
                item_logits_all[0, :valid_item] += pair_bias.mean(dim=1)
                space_logits_all[0, :valid_space] += pair_bias.mean(dim=0)
            if pair_logits_extra is not None:
                item_logits_all[0, :valid_item] += pair_logits_extra.mean(dim=1)
                space_logits_all[0, :valid_space] += pair_logits_extra.mean(dim=0)
            item_probs_full = masked_softmax(
                item_logits_all, valid_item, item_has_feasible[:valid_item] if valid_item > 0 else None
            )
            space_probs_full = masked_softmax(
                space_logits_all, valid_space, space_has_feasible[:valid_space] if valid_space > 0 else None
            )
            rotation_probs_full = masked_softmax(
                rotation_logits, rotation_count, rotation_has_feasible[:rotation_count] if rotation_count > 0 else None
            )

        # 归一化以避免数值问题
        if item_probs_full.sum() > 0:
            item_probs_full = item_probs_full / item_probs_full.sum()
        if space_probs_full.sum() > 0:
            space_probs_full = space_probs_full / space_probs_full.sum()
        if rotation_probs_full.sum() > 0:
            rotation_probs_full = rotation_probs_full / rotation_probs_full.sum()

        self.check_nan_inf(item_probs_full, "item_probs")
        self.check_nan_inf(space_probs_full, "space_probs")
        self.check_nan_inf(rotation_probs_full, "rotation_probs")

        return item_probs_full, space_probs_full, rotation_probs_full, feasibility_mask, rotation_dims


class DqnPolicy6(nn.Module):
    """改进后的策略网络：移除高度图依赖，引入Transformer聚合，增强对大窗口的鲁棒性。"""

    def __init__(self, item_num_max, space_free_num_max):
        super(DqnPolicy6, self).__init__()
        self.item_num_max = item_num_max
        self.space_free_num_max = space_free_num_max
        self.rotation_dim = 2

        self.item_embed = nn.Sequential(
            nn.Linear(4, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
        )

        self.space_embed = nn.Sequential(
            nn.Linear(5, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
        )

        item_layer = nn.TransformerEncoderLayer(
            d_model=128,
            nhead=4,
            dim_feedforward=256,
            dropout=0.1,
            activation="gelu",
        )
        space_layer = nn.TransformerEncoderLayer(
            d_model=128,
            nhead=4,
            dim_feedforward=256,
            dropout=0.1,
            activation="gelu",
        )
        self.item_encoder = nn.TransformerEncoder(item_layer, num_layers=2)
        self.space_encoder = nn.TransformerEncoder(space_layer, num_layers=2)

        self.item_summary_proj = nn.Sequential(
            nn.Linear(128 * 3, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
        )
        self.space_summary_proj = nn.Sequential(
            nn.Linear(128 * 3, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
        )

        self.context_proj = nn.Sequential(
            nn.Linear(128 * 2, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

        self.item_head = nn.Sequential(
            nn.LayerNorm(128 + 256),
            nn.Linear(128 + 256, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1),
        )
        self.space_head = nn.Sequential(
            nn.LayerNorm(128 + 256),
            nn.Linear(128 + 256, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1),
        )

        compat_input_dim = 128 * 3 + 256
        self.compat_mlp = nn.Sequential(
            nn.Linear(compat_input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh(),
        )

        self.rotation_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, self.rotation_dim),
        )

        self.item_bonus_scale = nn.Parameter(torch.tensor(0.6))
        self.space_bonus_scale = nn.Parameter(torch.tensor(0.6))

        # Critic网络：输出状态价值V(s)
        # 使用context_vec作为输入，因为它包含了状态的整体信息
        self.critic_head = nn.Sequential(
            nn.Linear(256, 128),  # context_vec的维度是256
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),  # 输出标量V(s)
        )

    def check_nan_inf(self, tensor, name):
        if tensor is None:
            return False
        if torch.isnan(tensor).any():
            print(f"{name}存在nan")
            return True
        if torch.isinf(tensor).any():
            print(f"{name}存在inf")
            return True
        return False

    def _masked_softmax(self, logits, valid_mask):
        if logits.numel() == 0 or valid_mask is None:
            return torch.zeros_like(logits)
        if valid_mask.sum() == 0:
            return torch.zeros_like(logits)
        masked_logits = logits.masked_fill(~valid_mask, -1e9)
        probs = F.softmax(masked_logits, dim=-1)
        probs = probs.masked_fill(~valid_mask, 0.0)
        norm = probs.sum()
        if norm > 0:
            probs = probs / norm
        return probs

    def _encode_items(self, item, item_unplaced_order, device, container_stats):
        denom_x, denom_y, denom_z, container_volume, container_plane_area, denom_diag = container_stats
        features = torch.full((self.item_num_max, 4), -1.0, device=device)
        valid_item = min(self.item_num_max, len(item_unplaced_order))
        for i in range(valid_item):
            cur_item = item[item_unplaced_order[i]]
            length = float(cur_item.length)
            width = float(cur_item.width)
            height = float(cur_item.height)
            volume = float(cur_item.volume)
            features[i] = torch.tensor(
                [
                    min(length / denom_x, 1.0),
                    min(width / denom_y, 1.0),
                    min(height / denom_z, 1.0),
                    min(volume / container_volume, 1.0),
                ],
                dtype=torch.float32,
                device=device,
            )
        item_padding_mask = torch.ones((1, self.item_num_max), dtype=torch.bool, device=device)
        if valid_item > 0:
            item_padding_mask[0, :valid_item] = False
        embedded = self.item_embed(features)
        encoded = self.item_encoder(embedded.unsqueeze(1), src_key_padding_mask=item_padding_mask).squeeze(1)
        encoded_valid = encoded[:valid_item]
        return encoded, encoded_valid, valid_item, item_padding_mask.squeeze(0)

    def _encode_spaces(self, space_free, device, stats, space_state):
        denom_x, denom_y, denom_z, container_volume, container_plane_area, denom_diag = stats
        norm_length = max(denom_x, 1.0)
        norm_width = max(denom_y, 1.0)
        norm_height = max(denom_z, 1.0)

        space_features = torch.full((self.space_free_num_max, 5), -1.0, device=device)
        valid_space = min(self.space_free_num_max, len(space_free))

        occupancy_arr = None
        if space_state is not None:
            occupancy_arr = np.asarray(space_state).astype(np.float32)

        support_values = []

        for i in range(valid_space):
            sf = space_free[i]
            support_ratio = 1.0 if sf.z == 0 else 0.0
            if occupancy_arr is not None and sf.z > 0:
                x0 = int(sf.x)
                y0 = int(sf.y)
                z0 = int(sf.z)
                L = max(int(round(sf.length)), 0)
                W = max(int(round(sf.width)), 0)
                if (
                    z0 - 1 >= 0
                    and x0 < occupancy_arr.shape[0]
                    and y0 < occupancy_arr.shape[1]
                    and z0 - 1 < occupancy_arr.shape[2]
                ):
                    x1 = min(x0 + L, occupancy_arr.shape[0])
                    y1 = min(y0 + W, occupancy_arr.shape[1])
                    if x1 > x0 and y1 > y0:
                        support_slice = occupancy_arr[x0:x1, y0:y1, z0 - 1]
                        area = max((x1 - x0) * (y1 - y0), 1)
                        support_ratio = float(np.clip(support_slice.sum() / area, 0.0, 1.0))
            volume_ratio = min(float(sf.volume) / container_volume, 1.0)

            support_values.append(support_ratio)

            space_features[i] = torch.tensor(
                [
                    float(np.clip(float(sf.length) / norm_length, 0.0, 1.0)),
                    float(np.clip(float(sf.width) / norm_width, 0.0, 1.0)),
                    float(np.clip(float(sf.height) / norm_height, 0.0, 1.0)),
                    support_ratio,
                    volume_ratio,
                ],
                dtype=torch.float32,
                device=device,
            )

        space_padding_mask = torch.ones((1, self.space_free_num_max), dtype=torch.bool, device=device)
        if valid_space > 0:
            space_padding_mask[0, :valid_space] = False

        embedded = self.space_embed(space_features)
        encoded = self.space_encoder(embedded.unsqueeze(1), src_key_padding_mask=space_padding_mask).squeeze(1)
        encoded_valid = encoded[:valid_space]

        aux = {"support_values": support_values}
        return encoded, encoded_valid, valid_space, space_padding_mask.squeeze(0), aux

    def forward(self, item, item_unplaced_order, space_free, device, space_size=None, space_state=None):
        from can_place_item import can_place_item

        if space_size is not None:
            container_dims = (float(space_size[0]), float(space_size[1]), float(space_size[2]))
        elif space_state is not None:
            state_shape = np.asarray(space_state).shape
            if len(state_shape) == 3:
                container_dims = (float(state_shape[0]), float(state_shape[1]), float(state_shape[2]))
            else:
                container_dims = (1.0, 1.0, 1.0)
        else:
            container_dims = (1.0, 1.0, 1.0)

        denom_x = max(container_dims[0], 1.0)
        denom_y = max(container_dims[1], 1.0)
        denom_z = max(container_dims[2], 1.0)
        container_volume = max(container_dims[0] * container_dims[1] * container_dims[2], 1.0)
        container_plane_area = max(container_dims[0] * container_dims[1], 1.0)
        container_diagonal = math.sqrt(container_dims[0] ** 2 + container_dims[1] ** 2 + container_dims[2] ** 2)
        denom_diag = max(container_diagonal, 1.0)
        container_stats = (denom_x, denom_y, denom_z, container_volume, container_plane_area, denom_diag)

        item_encoded, item_valid_embed, valid_item, item_padding_mask = self._encode_items(
            item, item_unplaced_order, device, container_stats
        )
        space_encoded, space_valid_embed, valid_space, space_padding_mask, space_aux = self._encode_spaces(
            space_free, device, container_stats, space_state
        )

        def summarize_embeddings(embeddings, valid_len):
            if valid_len > 0:
                mean_feat = embeddings[:valid_len].mean(dim=0)
                max_feat = embeddings[:valid_len].max(dim=0).values
                if valid_len > 1:
                    std_feat = embeddings[:valid_len].std(dim=0, unbiased=False)
                else:
                    std_feat = embeddings[:valid_len].new_zeros(embeddings.size(-1))
            else:
                zero_feat = embeddings.new_zeros(embeddings.size(-1))
                mean_feat = zero_feat
                max_feat = zero_feat
                std_feat = zero_feat
            return torch.cat([mean_feat, max_feat, std_feat], dim=0).unsqueeze(0)

        item_summary = self.item_summary_proj(summarize_embeddings(item_encoded, valid_item))
        space_summary = self.space_summary_proj(summarize_embeddings(space_encoded, valid_space))

        context_vec = self.context_proj(torch.cat([item_summary, space_summary], dim=-1)).squeeze(0)

        rotation_logits = self.rotation_head(context_vec.unsqueeze(0))
        rotation_count = rotation_logits.shape[1]

        item_logits_all = torch.full((1, self.item_num_max), -1e9, device=device)
        space_logits_all = torch.full((1, self.space_free_num_max), -1e9, device=device)

        if valid_item > 0:
            item_context = context_vec.unsqueeze(0).expand(valid_item, -1)
            item_logits = self.item_head(torch.cat([item_valid_embed, item_context], dim=1)).view(-1)
            item_logits_all[0, :valid_item] = item_logits
        if valid_space > 0:
            space_context = context_vec.unsqueeze(0).expand(valid_space, -1)
            space_logits = self.space_head(torch.cat([space_valid_embed, space_context], dim=1)).view(-1)
            space_logits_all[0, :valid_space] = space_logits

        feasibility_mask = torch.zeros(
            (self.item_num_max, self.space_free_num_max, rotation_count),
            dtype=torch.bool,
            device=device,
        )
        rotation_dims = torch.zeros(
            (self.item_num_max, self.space_free_num_max, rotation_count, 3),
            dtype=torch.int32,
            device=device,
        )
        item_has_feasible = torch.zeros(self.item_num_max, dtype=torch.bool, device=device)
        space_has_feasible = torch.zeros(self.space_free_num_max, dtype=torch.bool, device=device)
        rotation_has_feasible = torch.zeros(rotation_count, dtype=torch.bool, device=device)
        pair_valid_map = torch.zeros((self.item_num_max, self.space_free_num_max), dtype=torch.bool, device=device)

        for i in range(valid_item):
            item_id = item_unplaced_order[i]
            for j in range(valid_space):
                feasible_any = False
                for rot in range(rotation_count):
                    can_place, rotation_candidate = can_place_item(
                        space_size,
                        item,
                        item_id,
                        space_free,
                        j,
                        space_state,
                        restrict_lw=False,
                        rotation_action=rot,
                    )
                    if can_place and rotation_candidate is not None:
                        feasibility_mask[i, j, rot] = True
                        rotation_dims[i, j, rot, 0] = int(rotation_candidate[0])
                        rotation_dims[i, j, rot, 1] = int(rotation_candidate[1])
                        rotation_dims[i, j, rot, 2] = int(rotation_candidate[2])
                        feasible_any = True
                        rotation_has_feasible[rot] = True
                if feasible_any:
                    item_has_feasible[i] = True
                    space_has_feasible[j] = True
                    pair_valid_map[i, j] = True

        if valid_item > 0:
            infeasible_items = item_has_feasible[:valid_item] == False
            if infeasible_items.any():
                logits = item_logits_all[0, :valid_item]
                logits = logits.masked_fill(infeasible_items, -1e9)
                item_logits_all[0, :valid_item] = logits
        if valid_space > 0:
            infeasible_spaces = space_has_feasible[:valid_space] == False
            if infeasible_spaces.any():
                logits = space_logits_all[0, :valid_space]
                logits = logits.masked_fill(infeasible_spaces, -1e9)
                space_logits_all[0, :valid_space] = logits
        if rotation_has_feasible.any():
            infeasible_rot = rotation_has_feasible == False
            masked_rot = rotation_logits.clone()
            masked_rot[0, infeasible_rot] = -1e9
            rotation_logits = masked_rot

        if valid_item > 0 and valid_space > 0:
            item_expand = item_valid_embed.unsqueeze(1).expand(-1, valid_space, -1)
            space_expand = space_valid_embed.unsqueeze(0).expand(valid_item, -1, -1)
            diff = (item_expand - space_expand).abs()
            context_expand = context_vec.unsqueeze(0).unsqueeze(0).expand(valid_item, valid_space, -1)
            pair_features = torch.cat([item_expand, space_expand, diff, context_expand], dim=-1)
            pair_scores = self.compat_mlp(pair_features).squeeze(-1)
            valid_mask = pair_valid_map[:valid_item, :valid_space]
            if valid_mask.any():
                pair_scores = pair_scores.masked_fill(~valid_mask, 0.0)
                item_counts = valid_mask.sum(dim=1).clamp(min=1).float()
                space_counts = valid_mask.sum(dim=0).clamp(min=1).float()
                item_bonus = (pair_scores.sum(dim=1) / item_counts) * self.item_bonus_scale
                space_bonus = (pair_scores.sum(dim=0) / space_counts) * self.space_bonus_scale
                item_logits_all[0, :valid_item] += item_bonus
                space_logits_all[0, :valid_space] += space_bonus

        item_probs = torch.zeros((1, self.item_num_max), device=device)
        space_probs = torch.zeros((1, self.space_free_num_max), device=device)
        rotation_probs = torch.zeros((1, rotation_count), device=device)

        if valid_item > 0:
            valid_mask = torch.zeros(self.item_num_max, dtype=torch.bool, device=device)
            valid_mask[:valid_item] = item_has_feasible[:valid_item]
            if valid_mask.sum() == 0:
                valid_mask[:valid_item] = True
            probs = self._masked_softmax(item_logits_all[0], valid_mask)
            item_probs[0, :valid_item] = probs[:valid_item]
        if valid_space > 0:
            valid_mask = torch.zeros(self.space_free_num_max, dtype=torch.bool, device=device)
            valid_mask[:valid_space] = space_has_feasible[:valid_space]
            if valid_mask.sum() == 0:
                valid_mask[:valid_space] = True
            probs = self._masked_softmax(space_logits_all[0], valid_mask)
            space_probs[0, :valid_space] = probs[:valid_space]
        if rotation_count > 0:
            rot_mask = rotation_has_feasible.clone()
            if rot_mask.sum() == 0:
                rot_mask = torch.ones_like(rot_mask)
            rot_probs = self._masked_softmax(rotation_logits[0], rot_mask)
            rotation_probs[0, :rotation_count] = rot_probs

        self.check_nan_inf(item_probs, "item_probs")
        self.check_nan_inf(space_probs, "space_probs")
        self.check_nan_inf(rotation_probs, "rotation_probs")

        # Critic网络：计算状态价值V(s)
        # 使用context_vec作为状态表示
        state_value = self.critic_head(context_vec.unsqueeze(0)).squeeze(-1)  # shape: (1,)
        self.check_nan_inf(state_value, "state_value")

        return item_probs, space_probs, rotation_probs, feasibility_mask, rotation_dims, state_value


class DqnQNetwork(nn.Module):
    """DQN网络：输出Q值而不是概率分布"""
    
    def __init__(self, item_num_max, space_free_num_max):
        super(DqnQNetwork, self).__init__()
        self.item_num_max = item_num_max
        self.space_free_num_max = space_free_num_max
        self.rotation_dim = 2
        
        # 使用与DqnPolicy6相同的编码器结构
        self.item_embed = nn.Sequential(
            nn.Linear(4, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
        )
        
        self.space_embed = nn.Sequential(
            nn.Linear(5, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
        )
        
        item_layer = nn.TransformerEncoderLayer(
            d_model=128,
            nhead=4,
            dim_feedforward=256,
            dropout=0.1,
            activation="gelu",
        )
        space_layer = nn.TransformerEncoderLayer(
            d_model=128,
            nhead=4,
            dim_feedforward=256,
            dropout=0.1,
            activation="gelu",
        )
        self.item_encoder = nn.TransformerEncoder(item_layer, num_layers=2)
        self.space_encoder = nn.TransformerEncoder(space_layer, num_layers=2)
        
        self.item_summary_proj = nn.Sequential(
            nn.Linear(128 * 3, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
        )
        self.space_summary_proj = nn.Sequential(
            nn.Linear(128 * 3, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
        )
        
        self.context_proj = nn.Sequential(
            nn.Linear(128 * 2, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        
        # Q值头：输出Q值而不是logits
        self.item_q_head = nn.Sequential(
            nn.LayerNorm(128 + 256),
            nn.Linear(128 + 256, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1),  # 输出单个Q值
        )
        self.space_q_head = nn.Sequential(
            nn.LayerNorm(128 + 256),
            nn.Linear(128 + 256, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1),  # 输出单个Q值
        )
        
        self.rotation_q_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, self.rotation_dim),  # 输出每个旋转方向的Q值
        )
        
        # 兼容性网络：用于计算item-space对的联合Q值
        compat_input_dim = 128 * 3 + 256
        self.compat_mlp = nn.Sequential(
            nn.Linear(compat_input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh(),
        )
        
        self.item_bonus_scale = nn.Parameter(torch.tensor(0.6))
        self.space_bonus_scale = nn.Parameter(torch.tensor(0.6))
    
    def _encode_items(self, item, item_unplaced_order, device, container_stats):
        """编码货物特征（与DqnPolicy6相同）"""
        denom_x, denom_y, denom_z, container_volume, container_plane_area, denom_diag = container_stats
        features = torch.full((self.item_num_max, 4), -1.0, device=device)
        valid_item = min(self.item_num_max, len(item_unplaced_order))
        for i in range(valid_item):
            cur_item = item[item_unplaced_order[i]]
            length = float(cur_item.length)
            width = float(cur_item.width)
            height = float(cur_item.height)
            volume = float(cur_item.volume)
            features[i] = torch.tensor(
                [
                    min(length / denom_x, 1.0),
                    min(width / denom_y, 1.0),
                    min(height / denom_z, 1.0),
                    min(volume / container_volume, 1.0),
                ],
                dtype=torch.float32,
                device=device,
            )
        item_padding_mask = torch.ones((1, self.item_num_max), dtype=torch.bool, device=device)
        if valid_item > 0:
            item_padding_mask[0, :valid_item] = False
        embedded = self.item_embed(features)
        encoded = self.item_encoder(embedded.unsqueeze(1), src_key_padding_mask=item_padding_mask).squeeze(1)
        encoded_valid = encoded[:valid_item]
        return encoded, encoded_valid, valid_item, item_padding_mask.squeeze(0)
    
    def _encode_spaces(self, space_free, device, stats, space_state):
        """编码空间特征（与DqnPolicy6相同）"""
        denom_x, denom_y, denom_z, container_volume, container_plane_area, denom_diag = stats
        norm_length = max(denom_x, 1.0)
        norm_width = max(denom_y, 1.0)
        norm_height = max(denom_z, 1.0)
        
        space_features = torch.full((self.space_free_num_max, 5), -1.0, device=device)
        valid_space = min(self.space_free_num_max, len(space_free))
        
        occupancy_arr = None
        if space_state is not None:
            occupancy_arr = np.asarray(space_state).astype(np.float32)
        
        support_values = []
        
        for i in range(valid_space):
            sf = space_free[i]
            support_ratio = 1.0 if sf.z == 0 else 0.0
            if occupancy_arr is not None and sf.z > 0:
                x0 = int(sf.x)
                y0 = int(sf.y)
                z0 = int(sf.z)
                L = max(int(round(sf.length)), 0)
                W = max(int(round(sf.width)), 0)
                if (
                    z0 - 1 >= 0
                    and x0 < occupancy_arr.shape[0]
                    and y0 < occupancy_arr.shape[1]
                    and z0 - 1 < occupancy_arr.shape[2]
                ):
                    x1 = min(x0 + L, occupancy_arr.shape[0])
                    y1 = min(y0 + W, occupancy_arr.shape[1])
                    if x1 > x0 and y1 > y0:
                        support_slice = occupancy_arr[x0:x1, y0:y1, z0 - 1]
                        area = max((x1 - x0) * (y1 - y0), 1)
                        support_ratio = float(np.clip(support_slice.sum() / area, 0.0, 1.0))
            volume_ratio = min(float(sf.volume) / container_volume, 1.0)
            
            support_values.append(support_ratio)
            
            space_features[i] = torch.tensor(
                [
                    float(np.clip(float(sf.length) / norm_length, 0.0, 1.0)),
                    float(np.clip(float(sf.width) / norm_width, 0.0, 1.0)),
                    float(np.clip(float(sf.height) / norm_height, 0.0, 1.0)),
                    support_ratio,
                    volume_ratio,
                ],
                dtype=torch.float32,
                device=device,
            )
        
        space_padding_mask = torch.ones((1, self.space_free_num_max), dtype=torch.bool, device=device)
        if valid_space > 0:
            space_padding_mask[0, :valid_space] = False
        
        embedded = self.space_embed(space_features)
        encoded = self.space_encoder(embedded.unsqueeze(1), src_key_padding_mask=space_padding_mask).squeeze(1)
        encoded_valid = encoded[:valid_space]
        
        aux = {"support_values": support_values}
        return encoded, encoded_valid, valid_space, space_padding_mask.squeeze(0), aux
    
    def forward(self, item, item_unplaced_order, space_free, device, space_size=None, space_state=None):
        """
        前向传播，输出Q值
        
        Returns:
            item_q_values: (1, item_num_max) 每个货物的Q值
            space_q_values: (1, space_free_num_max) 每个空间的Q值
            rotation_q_values: (1, rotation_dim) 每个旋转方向的Q值
            joint_q_values: (item_num_max, space_free_num_max, rotation_dim) 联合Q值矩阵
            feasibility_mask: (item_num_max, space_free_num_max, rotation_dim) 可行性掩码
        """
        from can_place_item import can_place_item
        
        if space_size is not None:
            container_dims = (float(space_size[0]), float(space_size[1]), float(space_size[2]))
        elif space_state is not None:
            state_shape = np.asarray(space_state).shape
            if len(state_shape) == 3:
                container_dims = (float(state_shape[0]), float(state_shape[1]), float(state_shape[2]))
            else:
                container_dims = (1.0, 1.0, 1.0)
        else:
            container_dims = (1.0, 1.0, 1.0)
        
        denom_x = max(container_dims[0], 1.0)
        denom_y = max(container_dims[1], 1.0)
        denom_z = max(container_dims[2], 1.0)
        container_volume = max(container_dims[0] * container_dims[1] * container_dims[2], 1.0)
        container_plane_area = max(container_dims[0] * container_dims[1], 1.0)
        container_diagonal = math.sqrt(container_dims[0] ** 2 + container_dims[1] ** 2 + container_dims[2] ** 2)
        denom_diag = max(container_diagonal, 1.0)
        container_stats = (denom_x, denom_y, denom_z, container_volume, container_plane_area, denom_diag)
        
        item_encoded, item_valid_embed, valid_item, item_padding_mask = self._encode_items(
            item, item_unplaced_order, device, container_stats
        )
        space_encoded, space_valid_embed, valid_space, space_padding_mask, space_aux = self._encode_spaces(
            space_free, device, container_stats, space_state
        )
        
        def summarize_embeddings(embeddings, valid_len):
            if valid_len > 0:
                mean_feat = embeddings[:valid_len].mean(dim=0)
                max_feat = embeddings[:valid_len].max(dim=0).values
                if valid_len > 1:
                    std_feat = embeddings[:valid_len].std(dim=0, unbiased=False)
                else:
                    std_feat = embeddings[:valid_len].new_zeros(embeddings.size(-1))
            else:
                zero_feat = embeddings.new_zeros(embeddings.size(-1))
                mean_feat = zero_feat
                max_feat = zero_feat
                std_feat = zero_feat
            return torch.cat([mean_feat, max_feat, std_feat], dim=0).unsqueeze(0)
        
        item_summary = self.item_summary_proj(summarize_embeddings(item_encoded, valid_item))
        space_summary = self.space_summary_proj(summarize_embeddings(space_encoded, valid_space))
        
        context_vec = self.context_proj(torch.cat([item_summary, space_summary], dim=-1)).squeeze(0)
        
        # 计算旋转Q值
        rotation_q_values = self.rotation_q_head(context_vec.unsqueeze(0))  # (1, rotation_dim)
        
        # 计算item和space的Q值
        item_q_values_all = torch.full((1, self.item_num_max), -1e9, device=device)
        space_q_values_all = torch.full((1, self.space_free_num_max), -1e9, device=device)
        
        if valid_item > 0:
            item_context = context_vec.unsqueeze(0).expand(valid_item, -1)
            item_q_values = self.item_q_head(torch.cat([item_valid_embed, item_context], dim=1)).view(-1)
            item_q_values_all[0, :valid_item] = item_q_values
        
        if valid_space > 0:
            space_context = context_vec.unsqueeze(0).expand(valid_space, -1)
            space_q_values = self.space_q_head(torch.cat([space_valid_embed, space_context], dim=1)).view(-1)
            space_q_values_all[0, :valid_space] = space_q_values
        
        # 计算联合Q值矩阵和可行性掩码
        rotation_count = rotation_q_values.shape[1]
        joint_q_values = torch.zeros(
            (self.item_num_max, self.space_free_num_max, rotation_count),
            device=device
        )
        feasibility_mask = torch.zeros(
            (self.item_num_max, self.space_free_num_max, rotation_count),
            dtype=torch.bool,
            device=device,
        )
        
        if valid_item > 0 and valid_space > 0:
            # 计算兼容性分数
            item_expand = item_valid_embed.unsqueeze(1).expand(-1, valid_space, -1)
            space_expand = space_valid_embed.unsqueeze(0).expand(valid_item, -1, -1)
            diff = (item_expand - space_expand).abs()
            context_expand = context_vec.unsqueeze(0).unsqueeze(0).expand(valid_item, valid_space, -1)
            pair_features = torch.cat([item_expand, space_expand, diff, context_expand], dim=-1)
            compat_scores = self.compat_mlp(pair_features).squeeze(-1)  # (valid_item, valid_space)
            
            # 检查可行性并计算联合Q值
            for i in range(valid_item):
                item_id = item_unplaced_order[i]
                for j in range(valid_space):
                    for rot in range(rotation_count):
                        can_place, _ = can_place_item(
                            space_size,
                            item,
                            item_id,
                            space_free,
                            j,
                            space_state,
                            restrict_lw=True,
                            rotation_action=rot,
                        )
                        if can_place:
                            feasibility_mask[i, j, rot] = True
                            # 联合Q值 = item Q + space Q + rotation Q + 兼容性分数
                            item_q = item_q_values_all[0, i]
                            space_q = space_q_values_all[0, j]
                            rotation_q = rotation_q_values[0, rot]
                            compat = compat_scores[i, j]
                            joint_q_values[i, j, rot] = item_q + space_q + rotation_q + compat
        
        return item_q_values_all, space_q_values_all, rotation_q_values, joint_q_values, feasibility_mask
