import numpy as np

def reward_calculate(space_size, volume_utilization_step, space_state_volume, can_place, 
                     space_free=None, space_free_id=None,
                     item_volume=None,
                     alpha_box_ratio=None, alpha_utilization=None, alpha_bottom=None,
                     alpha_fail=None,
                     item_num_max=None, item_unplaced_order=None, item=None, selected_item_id=None):
    """
    奖励计算函数（参考 Online-3D-BPP-DRL 的简洁设计）
    
    参数:
        space_size: 容器尺寸 [L, W, H]
        volume_utilization_step: 当前货物占用当前空闲空间的利用率
        space_state_volume: 已占用的总体积
        can_place: 是否成功放置
        space_free: 空闲空间列表（可选，用于计算底部奖励）
        space_free_id: 选择的空闲空间索引（可选，用于计算底部奖励）
        item_volume: 当前货物的体积（可选，用于计算box_ratio奖励）
        alpha_box_ratio: 货物体积/容器体积奖励权重（默认10.0，参考参考代码）
        alpha_utilization: 当前空间利用率奖励权重（默认2.0）
        alpha_bottom: 底部优先奖励权重（默认1.0）
        alpha_fail: 失败惩罚（默认0.0，参考参考代码）
    
    返回:
        计算后的奖励值
    
    说明:
        - 主要奖励：参考参考代码，使用 box_ratio = 货物体积/容器体积，乘以缩放因子
        - 辅助奖励：当前空间利用率（局部优化信号）
        - 可选奖励：底部优先奖励（简化版）
    """
    # 默认权重配置（调整后：使 box_ratio 占主导，其他作为辅助）
    # 注意：由于奖励会被标准化，重要的是奖励之间的相对差异，而不是绝对值
    if alpha_box_ratio is None:
        alpha_box_ratio = 50.0  # 从10.0提高到50.0，使其成为主要奖励信号
        # 这样即使 box_ratio 很小（0.01），reward_box_ratio = 0.01 * 50 = 0.5，仍然有显著影响
    if alpha_utilization is None:
        alpha_utilization = 1.0  # 从2.0降低到1.0，减少影响，作为辅助信号
    if alpha_bottom is None:
        alpha_bottom = 0.5  # 从1.0降低到0.5，作为辅助信号
    if alpha_fail is None:
        alpha_fail = 0.0  # 参考代码失败时返回0.0
    
    capacity = space_size[0] * space_size[1] * space_size[2]
    
    # ========== 主要奖励：参考参考代码的 box_ratio 设计 ==========
    # 参考代码：reward = box_ratio * 10，其中 box_ratio = 货物体积 / 容器体积
    # 改进：使用非线性函数放大差异，使标准化后仍能体现差异
    if can_place and item_volume is not None and capacity > 0:
        box_ratio = item_volume / capacity  # 货物体积与容器体积的比值 [0, 1]
        
        # ========== 非线性奖励方案：改变奖励的相对关系 ==========
        # 方案1：平方放大（推荐）- 大货物的奖励会被显著放大，差异更大
        # 使用平方函数放大差异，使标准化后仍能体现明显差异
        # 关键：通过平方函数，大货物的奖励增长更快，相对差异更大
        if box_ratio > 0.1:  # 大货物（体积>10%容器）
            # 平方放大：box_ratio=0.2 → 0.04，但通过权重补偿，使奖励仍然很大
            reward_box_ratio = (box_ratio ** 2) * alpha_box_ratio * 10.0  # 平方放大，权重10.0
        else:  # 小货物
            reward_box_ratio = box_ratio * alpha_box_ratio * 0.2  # 线性，权重更小
        
        # 方案2：指数放大 - 更激进的差异放大（备选）
        # if box_ratio > 0.1:
        #     reward_box_ratio = (box_ratio ** 1.5) * alpha_box_ratio * 2.5
        # else:
        #     reward_box_ratio = box_ratio * alpha_box_ratio * 0.3
        
        # 方案3：分段线性 - 大货物获得更高权重（之前使用，差异不够大）
        # if box_ratio > 0.1:
        #     reward_box_ratio = box_ratio * alpha_box_ratio * 1.5
        # else:
        #     reward_box_ratio = box_ratio * alpha_box_ratio * 0.5
        
        # 方案4：原始线性（回退方案）
        # reward_box_ratio = box_ratio * alpha_box_ratio
        # ===================================================================
    else:
        reward_box_ratio = 0.0
    # ===================================================================
    
    # ========== 辅助奖励：当前空间利用率（局部优化信号） ==========
    # 鼓励选择能更好填充当前空间的货物-空间组合
    reward_utilization = volume_utilization_step if can_place else 0.0  # [0, 1]
    reward_utilization = reward_utilization * alpha_utilization
    # ===================================================================
    
    # ========== 可选奖励：简化的底部优先奖励 ==========
    reward_bottom = 0.0
    if can_place and space_free is not None and space_free_id is not None:
        if 0 <= space_free_id < len(space_free):
            space = space_free[space_free_id]
            
            # 归一化 z 坐标到 [0, 1]
            if space_size[2] > 0:
                norm_z = space.z / space_size[2]  # z 坐标归一化：0（底部）到 1（顶部）
            else:
                norm_z = 0.0
            
            # 简化的底部优先奖励：使用线性衰减
            # norm_z = 0（底部）时，reward_bottom = 1.0
            # norm_z = 1（顶部）时，reward_bottom = 0.0
            reward_bottom = 1.0 - norm_z  # [0, 1]
    reward_bottom = reward_bottom * alpha_bottom
    # ===================================================================
    
    # ========== 可选奖励：货物选择奖励（简化版） ==========
    # 当 item_num_max > 1 时，给选择"更好"货物的决策额外奖励
    reward_item_selection = 0.0
    if item_num_max and item_num_max > 1 and can_place and item_unplaced_order is not None and item is not None and selected_item_id is not None:
        selected_item = item[selected_item_id]
        item_volume_selected = selected_item.volume
        
        # 计算该货物在所有未放置货物中的排名（体积）
        item_volumes = [item[idx].volume for idx in item_unplaced_order if idx != selected_item_id]
        if len(item_volumes) > 0:
            # 计算该货物的体积排名（0=最大，1=最小）
            larger_count = sum(1 for vol in item_volumes if vol > item_volume_selected)
            rank_ratio = larger_count / len(item_volumes) if len(item_volumes) > 0 else 0.5
            
            # 如果选择了体积较大的货物（排名前30%），给予小奖励
            if rank_ratio <= 0.3:
                reward_item_selection = 0.2 * (1.0 - rank_ratio)  # 最大奖励0.2
        
        # 考虑空间利用率：如果选择的货物在当前空间中利用率很高，额外奖励
        if volume_utilization_step > 0.85:  # 利用率超过85%
            reward_item_selection += 0.1
    # ===================================================================
    
    # 计算总奖励
    # ========== 测试模式：只保留 box_ratio，用于验证奖励函数是否生效 ==========
    # 如果训练结果改变了，说明 reward_box_ratio 确实有效
    # 如果训练结果没改变，说明问题不在奖励函数本身
    TEST_MODE_BOX_RATIO_ONLY = False  # 设置为 True 来测试只使用 box_ratio
    if TEST_MODE_BOX_RATIO_ONLY:
        total_reward = reward_box_ratio  # 只保留 box_ratio
    else:
        total_reward = reward_box_ratio + reward_utilization + reward_bottom + reward_item_selection
    # ===================================================================

    if not can_place:
        return alpha_fail

    # ========== 调试输出：验证奖励函数是否生效 ==========
    # 取消注释下面的行来查看奖励值（用于验证奖励函数是否真的改变了奖励）
    '''
    if can_place and item_volume is not None:
        print(f"[奖励调试] item_volume={item_volume:.2f}, box_ratio={box_ratio:.6f}, "
              f"reward_box_ratio={reward_box_ratio:.4f}, utilization={reward_utilization:.4f}, "
              f"bottom={reward_bottom:.4f}, item_selection={reward_item_selection:.4f}, "
              f"total={total_reward:.4f}")
    # ===================================================================
    '''
    return total_reward
