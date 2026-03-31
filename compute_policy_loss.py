from my_imports import torch, np, F


def compute_policy_loss(
    device,
    probs_log1,
    probs_log2,
    probs_log3,
    rewards_log,
    data_method,
    episode,
    supervise_quantity,
    action_target=None,
    joint_log_corrections=None,
    values_log=None,  # 状态值V(s)列表，用于AC算法
):
    # 损失函数
    if len(rewards_log) == 0:
        print("奖励列表有误。")
        return torch.tensor(0.0, requires_grad=True)
    if data_method == 3 or (data_method == 5 and episode <= supervise_quantity):
        # 100
        '''
        # 监督学习：使用交叉熵损失
        loss = torch.tensor(0.0, requires_grad=True)
        for i, (target_action1, target_action2) in enumerate(action_target):
            if i < len(probs_log1) and i < len(probs_log2):
                # 使用负对数似然损失（等价于交叉熵）
                loss = loss - probs_log1[target_action1] - probs_log2[target_action2]
        return loss
        '''
        lambda_reg = 0.1
        # 将概率分布和目标动作转换为张量
        item_probs_tensor = torch.stack(probs_log1).squeeze(1).to(device)
        space_probs_tensor = torch.stack(probs_log2).squeeze(1).to(device)
        rotation_probs_tensor = torch.stack(probs_log3).squeeze(1).to(device)
        epsilon = 1e-10
        item_probs_tensor = item_probs_tensor.clamp(min=epsilon)
        space_probs_tensor = space_probs_tensor.clamp(min=epsilon)
        rotation_probs_tensor = rotation_probs_tensor.clamp(min=epsilon)

        target1_tensor = torch.tensor([t[0] for t in action_target], dtype=torch.long, device=device)
        target2_tensor = torch.tensor([t[1] for t in action_target], dtype=torch.long, device=device)
        target3_tensor = torch.tensor([t[2] for t in action_target], dtype=torch.long, device=device)

        # 交叉熵损失
        # loss1 = F.cross_entropy(item_probs_tensor, target1_tensor)
        # loss2 = F.cross_entropy(space_probs_tensor, target2_tensor)
        loss1 = F.nll_loss(torch.log(item_probs_tensor), target1_tensor)
        loss2 = F.nll_loss(torch.log(space_probs_tensor), target2_tensor)
        loss3 = F.nll_loss(torch.log(rotation_probs_tensor), target3_tensor)
        supervised_loss = loss1 + loss2 + loss3

        # 奖励正则化
        # reward_reg = -torch.tensor(rewards_log).mean()  # 负号因为要最小化损失

        # 将奖励与采样动作的概率结合，类似策略梯度。
        log_probs_item = torch.log(item_probs_tensor[range(len(target1_tensor)), target1_tensor])
        log_probs_space = torch.log(space_probs_tensor[range(len(target2_tensor)), target2_tensor])
        log_probs_rotation = torch.log(rotation_probs_tensor[range(len(target3_tensor)), target3_tensor])
        rewards_tensor = torch.tensor(rewards_log, device=device)
        # 改进的标准化：更安全，避免标准差接近0时的不稳定
        if len(rewards_tensor) > 1:
            std = rewards_tensor.std()
            mean = rewards_tensor.mean()
            if std > 1e-6:  # 只有当标准差足够大时才标准化
                rewards_tensor = (rewards_tensor - mean) / std
            else:
                # 标准差太小，只做中心化（减去均值），不做缩放
                rewards_tensor = rewards_tensor - mean
        else:
            # 只有1个样本，只做中心化
            mean = rewards_tensor.mean()
            rewards_tensor = rewards_tensor - mean
        reward_reg = -(rewards_tensor * (log_probs_item + log_probs_space + log_probs_rotation)).mean()
        loss = supervised_loss + lambda_reg * reward_reg
        return loss

    elif data_method == 1 or data_method == 2 or data_method == 4 or (data_method == 5 and episode > supervise_quantity):
        # AC算法：使用优势函数A = R - V(s)
        gamma = 0.99
        
        # 计算折扣累积奖励（蒙特卡洛回报）
        discounted_rewards = []
        discounted_reward = 0
        for reward in reversed(rewards_log):
            discounted_reward = reward + gamma * discounted_reward
            discounted_rewards.append(discounted_reward)  # 尾部追加
        discounted_rewards = list(reversed(discounted_rewards))  # 整体反转
        discounted_rewards = torch.tensor(discounted_rewards, device=device, dtype=torch.float32)
        
        # 获取状态值V(s)
        if values_log is not None and len(values_log) == len(rewards_log):
            # 将values_log转换为张量
            values_tensor = torch.stack(values_log).squeeze().to(device)  # shape: (T,)
            if values_tensor.dim() == 0:  # 如果只有一个元素
                values_tensor = values_tensor.unsqueeze(0)
        else:
            # 如果没有提供values_log，使用零作为baseline（退化为REINFORCE）
            print("警告：未提供values_log，使用零baseline（退化为REINFORCE）")
            values_tensor = torch.zeros_like(discounted_rewards)
        
        # 计算优势函数：A(s,a) = R - V(s)
        # 注意：在Actor损失中使用detach，但在Critic损失中不使用
        advantages = discounted_rewards - values_tensor.detach()  # detach V(s)以避免在Actor损失中影响Critic梯度
        
        # 计算策略损失（Actor损失）
        probs_log1_tensor = torch.stack(probs_log1).to(device)
        probs_log2_tensor = torch.stack(probs_log2).to(device)
        probs_log3_tensor = torch.stack(probs_log3).to(device)
        if joint_log_corrections is not None and len(joint_log_corrections) == len(probs_log1):
            joint_corrections_tensor = torch.stack(joint_log_corrections).to(device)
        else:
            joint_corrections_tensor = torch.zeros_like(probs_log1_tensor)
        total_log_probs = probs_log1_tensor + probs_log2_tensor + probs_log3_tensor - joint_corrections_tensor
        
        # Actor损失：-log π(a|s) * A(s,a)
        actor_loss = -(total_log_probs * advantages).sum()
        actor_loss = actor_loss / len(advantages)  # 归一化
        
        # Critic损失：MSE(V(s), R)
        critic_loss = F.mse_loss(values_tensor, discounted_rewards)
        
        # 总损失：Actor损失 + Critic损失
        # 可以调整权重，这里使用1:1的比例
        loss = actor_loss + critic_loss
        
        # 检查
        loss_temp = loss
        loss_temp = loss_temp.cpu().detach().numpy()
        if np.isnan(loss_temp).any() or np.isinf(loss_temp).any():
            print("损失函数有误！")
        
        return loss
