# ========== 改进：兼容性检查缓存机制 ==========
from functools import lru_cache
from collections import OrderedDict
import numpy as np

# 缓存大小：根据 item_num_max 和空间数量动态调整
# 假设最多有 item_num_max * space_free_num_max 个组合需要缓存
# 设置一个合理的缓存大小（例如 1000 个条目）
_COMPATIBILITY_CACHE_SIZE = 1000

# 全局 LRU 缓存（使用 OrderedDict 实现）
_compatibility_cache = OrderedDict()
_cache_hits = 0
_cache_misses = 0

def _get_cache_key(item_dims, space_info, support_hash=None):
    """
    生成缓存键
    
    参数:
        item_dims: (length, width, height) 元组
        space_info: (x, y, z, length, width, height) 元组
        support_hash: space_state 在支撑层的哈希值（可选）
    """
    # 将 item_dims 和 space_info 转换为可哈希的元组
    key = (tuple(item_dims), tuple(space_info))
    if support_hash is not None:
        key = key + (support_hash,)
    return key

def _compute_support_hash(space_state, x0, y0, z0, L, W):
    """
    计算支撑层的哈希值（用于稳定性检查的缓存）
    
    参数:
        space_state: 空间状态矩阵
        x0, y0, z0: 空间起始坐标
        L, W: 货物的长度和宽度
    """
    if space_state is None or z0 <= 0:
        return None
    
    try:
        # 取底面的支撑切片（z0-1层）
        support = space_state[x0:x0 + L, y0:y0 + W, z0 - 1]
        if support.size == 0:
            return None
        # 使用支撑区域的哈希值作为缓存键的一部分
        # 使用 sum 和 shape 作为简化的哈希（避免对整个数组进行哈希计算）
        return (int(support.sum()), support.shape[0], support.shape[1])
    except:
        return None

def can_place_item(space_size, item, item_id, space_free, space_free_id, space_state=None, restrict_lw=True, rotation_action=None):
    """
    检查货物是否可以放入空间（带缓存优化）
    
    缓存机制：
    - 缓存键基于：货物尺寸、空间位置和尺寸、支撑层信息
    - 使用 LRU 缓存策略，避免重复计算
    """
    global _compatibility_cache, _cache_hits, _cache_misses
    
    # 旋转货物，看是否能放入空间内
    item_dims = (item[item_id].length, item[item_id].width, item[item_id].height)
    space_dims = [space_free[space_free_id].length, space_free[space_free_id].width, space_free[space_free_id].height]
    from itertools import permutations

    x0 = space_free[space_free_id].x
    y0 = space_free[space_free_id].y
    z0 = space_free[space_free_id].z
    
    space_info = (x0, y0, z0, space_dims[0], space_dims[1], space_dims[2])

    # 尝试从缓存中获取结果
    # 注意：由于稳定性检查依赖于 space_state，我们需要计算支撑层的哈希值
    # 但为了简化，我们先检查尺寸兼容性（这部分不需要 space_state）
    
    # 对于尺寸检查，我们可以直接使用缓存
    # 但对于稳定性检查，需要 space_state 信息，所以我们需要在实际检查时计算
    # 这里我们采用混合策略：先检查尺寸，再检查稳定性

    best_rotation = None  # 旋转方向
    best_secondary_metric = -1  # 用于打破分数并列的次级指标（更大的最小剩余尺寸优先）
    # 初始化最差分数为-1（因为我们要最大化旋转评分）
    worst_waste_score = -1  # 旋转评分（越高越好，这里复用变量名）

    # ========== 新增：根据未放置货物动态确定“可用最小尺寸”阈值，避免产生过细条空间 ==========
    try:
        unplaced_min_dims = []
        for it in item:
            if not it.placed:
                unplaced_min_dims.append(min(it.length, it.width, it.height))
        # 若无未放置货物，则退化为1
        min_required_dim = max(1, min(unplaced_min_dims)) if len(unplaced_min_dims) > 0 else 1

        # 额外统计量：使用分位数/中位数作为更“稳健”的阈值参考
        if len(unplaced_min_dims) > 0:
            unplaced_min_dims_sorted = sorted(unplaced_min_dims)
            n_dim = len(unplaced_min_dims_sorted)
            median_required_dim = unplaced_min_dims_sorted[n_dim // 2]
            p25_required_dim = unplaced_min_dims_sorted[max(0, (n_dim * 25) // 100)]  # 近似25分位
            # 保证至少为1
            median_required_dim = max(1, median_required_dim)
            p25_required_dim = max(1, p25_required_dim)
        else:
            median_required_dim = 1
            p25_required_dim = 1
    except Exception:
        min_required_dim = 1
        median_required_dim = 1
        p25_required_dim = 1
    # ===================================================================

    # 定义候选旋转集合
    # 只支持2种旋转：长宽互换，高度保持不变（与网络输出一致）
    rotations = [
        (item_dims[0], item_dims[1], item_dims[2]),  # idx 0: (L,W,H) 原始方向
        (item_dims[1], item_dims[0], item_dims[2])   # idx 1: (W,L,H) 长宽互换
    ]
    # 如指定了具体旋转动作，只检查该动作；索引越界则钳制
    if rotation_action is not None:
        idx = 0 if rotation_action < 0 else rotation_action
        idx = min(idx, len(rotations) - 1)
        rotation_iter = [rotations[idx]]
    else:
        rotation_iter = rotations

    for perm in rotation_iter:
        # 尺寸可行（这部分可以缓存）
        if not all(perm[i] <= space_dims[i] for i in range(3)):
            continue

        # 稳定性约束（可选）：要求底面在z0-1层有足够支撑
        # 这部分需要 space_state，所以无法完全缓存，但我们可以缓存部分结果
        stable_ok = True
        support_hash = None
        if space_state is not None:
            L, W, H = perm[0], perm[1], perm[2]
            # 计算支撑层的哈希值
            support_hash = _compute_support_hash(space_state, x0, y0, z0, L, W)
            cache_key = _get_cache_key(perm, space_info, support_hash)
            
            # 检查缓存
            if cache_key in _compatibility_cache:
                cached_result = _compatibility_cache[cache_key]
                _cache_hits += 1
                # 将命中的条目移到末尾（标记为最新使用）
                _compatibility_cache.move_to_end(cache_key)
                if cached_result is not None:
                    # 缓存命中，使用缓存的结果
                    rotation_score = cached_result['score']
                    if rotation_score > worst_waste_score:
                        worst_waste_score = rotation_score
                        best_rotation = perm
                    continue
            else:
                _cache_misses += 1
            
            # 放置在地面z=0则天然稳定
            if z0 > 0:
                # 取底面的支撑切片（z0-1层）
                support = space_state[x0:x0 + L, y0:y0 + W, z0 - 1]
                # 若范围越界或支持张量为空，视为不稳定
                if support.size == 0:
                    stable_ok = False
                else:
                    area = L * W
                    support_area = int((support == 1).sum())
                    # 角点是否被支撑
                    LU = int(support[0, 0] == 1)
                    LD = int(support[L - 1, 0] == 1)
                    RU = int(support[0, W - 1] == 1)
                    RD = int(support[L - 1, W - 1] == 1)
                    # 规则参考 Online-3D-BPP-DRL-main 的 check_box 稳定性判据
                    if support_area / area > 0.95:
                        stable_ok = True
                    elif (LU + LD + RU + RD) == 3 and support_area / area > 0.85:
                        stable_ok = True
                    elif (LU + LD + RU + RD) == 4 and support_area / area > 0.50:
                        stable_ok = True
                    else:
                        stable_ok = False
        else:
            # 如果没有 space_state，使用简单的缓存键
            cache_key = _get_cache_key(perm, space_info, None)
            if cache_key in _compatibility_cache:
                cached_result = _compatibility_cache[cache_key]
                _cache_hits += 1
                # 将命中的条目移到末尾（标记为最新使用）
                _compatibility_cache.move_to_end(cache_key)
                if cached_result is not None:
                    rotation_score = cached_result['score']
                    if rotation_score > worst_waste_score:
                        worst_waste_score = rotation_score
                        best_rotation = perm
                    continue
            else:
                _cache_misses += 1
        
        if not stable_ok:
            # 稳定性检查失败，缓存结果
            if support_hash is not None or space_state is None:
                cache_key = _get_cache_key(perm, space_info, support_hash)
                if len(_compatibility_cache) >= _COMPATIBILITY_CACHE_SIZE:
                    _compatibility_cache.popitem(last=False)
                _compatibility_cache[cache_key] = None  # None 表示不可行
                _compatibility_cache.move_to_end(cache_key)
            continue

        # ========== 改进的评估方式 ==========
        # 评估旋转方向的好坏：优先选择完全匹配维度最多的，其次选择剩余空间更集中的
        
        # 1. 计算完全匹配的维度数量（最重要：完全匹配的维度越多，产生的剩余空间越少）
        perfect_fit_count = sum(1 for i in range(3) if space_dims[i] - perm[i] == 0)
        
        # 2. 计算剩余维度的差值（剩余空间的大小）
        remaining_dims = [space_dims[i] - perm[i] for i in range(3)]
        # 过滤掉完全匹配的维度（差值为0的维度不会产生剩余空间）
        active_remaining = [r for r in remaining_dims if r > 0]
        
        # 3. 评估剩余空间的分布
        # 剩余空间的数量（非零剩余维度的数量，越少越好）
        remaining_space_count = len(active_remaining)
        
        # 4. 评估剩余空间的最小尺寸（剩余空间的最小尺寸越大，越容易放置后续货物）
        min_remaining_size = min(active_remaining) if active_remaining else 0
        
        # 5. 综合评分
        # 优先考虑完全匹配的维度数量（权重最高）
        # 其次考虑剩余空间的分散程度（剩余维度越少越好）
        # 最后考虑剩余空间的最小尺寸（最小尺寸越大越好）
        
        # 分数计算：完全匹配数优先，剩余空间分散度其次，最小尺寸最后
        # 使用多级评分：完全匹配数 * 10000 + (3-剩余维度数) * 1000 + 最小尺寸
        # 这样确保完全匹配数多的总是优先，完全匹配数相同时剩余维度少的优先
        rotation_score = (
            perfect_fit_count * 10000 +           # 完全匹配数（权重最高）
            (3 - remaining_space_count) * 1000 +   # 剩余空间分散度（剩余维度越少分数越高）
            min_remaining_size                     # 剩余空间最小尺寸
        )
        
        # 6. 强化惩罚：避免产生“小于未放置货物可用最小尺寸”的细条剩余空间
        # 对于每个非零剩余维度，如果它 < min_required_dim，施加强惩罚；特别地，若为1，再追加惩罚
        if active_remaining:
            sliver_count = sum(1 for r in active_remaining if r < min_required_dim)
            one_width_count = sum(1 for r in active_remaining if r == 1)
            if sliver_count > 0:
                rotation_score -= sliver_count * 5000  # 强惩罚：小于最小可用尺寸
            if one_width_count > 0:
                rotation_score -= one_width_count * 2000  # 额外惩罚：1宽条

        # 7. 进一步偏好：若某一维完全贴合（0剩余），同时其它非零剩余维度均>=min_required_dim，给予加分
        if perfect_fit_count >= 1 and active_remaining and all(r >= min_required_dim for r in active_remaining):
            rotation_score += 1500

        # 7.1 使用更稳健的阈值（中位数/25分位）作为奖励引导：剩余的最小尺寸如果 >= 中位数，额外加分；若 < 25分位，减分
        if active_remaining:
            if min_remaining_size >= median_required_dim:
                rotation_score += 800
            elif min_remaining_size < p25_required_dim:
                rotation_score -= 1200

        # 7.2 剩余维度的“形状”惩罚：若留下两个以上维度，且长宽比过大，惩罚（避免极端细长空间）
        if len(active_remaining) >= 2:
            max_r = max(active_remaining)
            min_r = max(1, min(active_remaining))
            aspect_ratio = max_r / min_r
            if aspect_ratio > 3.0:
                rotation_score -= int((aspect_ratio - 3.0) * 500)  # 超过阈值的部分线性惩罚

        # 7.3 接触面积偏好：更大的底面接触面积更稳定，有利于后续分割（轻微加分）
        contact_area = perm[0] * perm[1]
        rotation_score += int(contact_area * 0.01)  # 轻量级加分，避免压过主导项

        # 7.4 “可容纳度”偏好：剩余的最小尺寸能够容纳的未放置货物比例越高，加分
        if active_remaining:
            if len(unplaced_min_dims) > 0:
                fit_popularity = sum(1 for d in unplaced_min_dims if d <= min_remaining_size)
                rotation_score += fit_popularity * 20

            # 对极小剩余进一步惩罚，避免无意义的狭缝
            if min_remaining_size < 2:
                rotation_score -= 800

        # 计算用于打破并列的次级指标：尽量让最小剩余尺寸更大
        secondary_metric = min(active_remaining) if active_remaining else 10**9
        
        # 8. 选择分数最高的旋转方向（若并列，选择最小剩余尺寸更大的）
        if (rotation_score > worst_waste_score or
            (rotation_score == worst_waste_score and secondary_metric > best_secondary_metric)):  # 这里复用变量名，实际是分数（越高越好）
            worst_waste_score = rotation_score
            best_rotation = perm
            best_secondary_metric = secondary_metric
        
        # 缓存结果（LRU 策略）：只缓存稳定且可行的旋转
        cache_key = _get_cache_key(perm, space_info, support_hash)
        if len(_compatibility_cache) >= _COMPATIBILITY_CACHE_SIZE:
            # 删除最旧的条目（FIFO）
            _compatibility_cache.popitem(last=False)
        _compatibility_cache[cache_key] = {'score': rotation_score, 'rotation': perm}
        # 将新条目移到末尾（标记为最新使用）
        _compatibility_cache.move_to_end(cache_key)

    if best_rotation:
        return True, best_rotation
    return False, None

def clear_compatibility_cache():
    """清空兼容性检查缓存"""
    global _compatibility_cache, _cache_hits, _cache_misses
    _compatibility_cache.clear()
    _cache_hits = 0
    _cache_misses = 0

def get_cache_stats():
    """获取缓存统计信息"""
    global _cache_hits, _cache_misses
    total = _cache_hits + _cache_misses
    hit_rate = _cache_hits / total if total > 0 else 0.0
    return {
        'hits': _cache_hits,
        'misses': _cache_misses,
        'hit_rate': hit_rate,
        'cache_size': len(_compatibility_cache)
    }
# ===================================================================