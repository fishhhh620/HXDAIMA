# 随机产生货物数据
from my_imports import Item, SpaceFree, np, pd, deepcopy, time, os
from can_place_item import can_place_item
from place_item import place_item
from parameter import parameter
from save_result import save_result
from get_item_order import get_item_order

print('开始运行！')
time_begin = time.time()  # 记录开始的时间
# 数据选择 全局变量
data_method = 5
space_manage_method = 'simple3' # 修改这里：'corner' 或 'simple3'
space_size, item_lwh_min, item_lwh_max, volume_utilization_mean_max, space_free_num_max, item_place_times_max, item_num_max, space_manage_method = parameter(
    data_method, space_manage_method)
data_scale = 10000  # 数据集规模

# ========== 新增：控制平均每个箱子货物数量的参数 ==========
target_avg_item_count = 32.94  # 目标平均每个箱子货物数量
p = 0.5  # 初始 p 值
p_adjust_interval = 50  # 每生成多少个箱子调整一次 p
p_step = 0.05  # 每次调整 p 的步长
# 根据模式设置 p 的上限，Simple3 模式分割效率高，需要更大的 p 上限
if space_manage_method == 'simple3':
    p_step = 0.05  # 更大的步长
    p_max = 0.95  # 更高的上限 (严格限制 < 1.0)
else:
    p_step = 0.05
    p_max = 0.95
print(f'目标平均货物数量: {target_avg_item_count}, 初始 p 值: {p}, p 上限: {p_max}')
# =========================================================

example_id = 0  # 数据编号
data_all = []
item_count_history = []  # 用于记录每个箱子的货物数量
# 随机分割
while example_id < data_scale:
    space_free_num_split_max = 0  # 分割过程中空闲空间的最大数量
    # p = np.random.rand()  # 一定的概率即使可以分割也不进行分割保留一些体积大的货物
    # 空闲空间
    tag = 1  # 是否可以继续分割的标志
    space_free = []
    space_free.append(SpaceFree(0, 0, 0, space_size[0], space_size[1], space_size[2]))
    space_state = np.zeros((space_size[0], space_size[1], space_size[2]))  # 空间的状态
    item = []
    t = 0
    t_max = 500  # 尝试最大次数
    while tag:
        t = t + 1
        # space_free.sort(key=lambda space: space.z)  # 空间按照z轴排序
        # temp1 = [i for i, space_free in enumerate(space_free) if space_free.segment1 is True]  # 找到可以继续分割的箱子
        if len(space_free) == 0:
            tag = 0
            break
        # min_z_temp = min(obj.z for obj in space_free)  # z轴最小值
        # temp2 = [index for index, obj in enumerate(space_free) if obj.z == min_z_temp]  # 所有符合最小z的空闲空间
        # temp3 = np.random.randint(0, min(len(space_free), space_free_num, len(temp2)))
        # space_free_id = temp2[temp3]  # 待分割空间的编号
        space_free_id = np.random.randint(0, min(len(space_free), space_free_num_max))
        # print('空闲空间，长：%d，宽：%d，高：%d' % (space_free[space_free_id].length, space_free[space_free_id].width, space_free[space_free_id].height))
        for i in range(0, 3):
            # 分割长
            if i == 0 and space_free[space_free_id].length >= item_lwh_min[i] * 2:
                if p < np.random.rand():
                    # 分割
                    length = np.random.randint(item_lwh_min[i],
                                               min(item_lwh_max[i], space_free[space_free_id].length - item_lwh_min[i],
                                                   space_free[space_free_id].length) + 1)
                else:
                    # 不分割：在约束范围内尽量填满空间
                    length = min(space_free[space_free_id].length, item_lwh_max[i])
                    if length < item_lwh_min[i]:
                        length = space_free[space_free_id].length  # 如果空间太小就直接使用整个空间
            elif i == 0:
                # 无法继续分割成2个货物，则不分割
                length = space_free[space_free_id].length
            # 分割宽
            if i == 1 and space_free[space_free_id].width >= item_lwh_min[i] * 2:
                if p < np.random.rand():
                    width = np.random.randint(item_lwh_min[i],
                                              min(item_lwh_max[i], space_free[space_free_id].width - item_lwh_min[i],
                                                  space_free[space_free_id].width) + 1)
                else:
                    # 不分割：在约束范围内尽量填满空间
                    width = min(space_free[space_free_id].width, item_lwh_max[i])
                    if width < item_lwh_min[i]:
                        width = space_free[space_free_id].width
            elif i == 1:
                width = space_free[space_free_id].width
            # 分割高
            if i == 2 and space_free[space_free_id].height >= item_lwh_min[i] * 2:
                if p < np.random.rand():
                    height = np.random.randint(item_lwh_min[i],
                                               min(item_lwh_max[i], space_free[space_free_id].height - item_lwh_min[i],
                                                   space_free[space_free_id].height) + 1)
                else:
                    # 不分割：在约束范围内尽量填满空间
                    height = min(space_free[space_free_id].height, item_lwh_max[i])
                    if height < item_lwh_min[i]:
                        height = space_free[space_free_id].height
            elif i == 2:
                height = space_free[space_free_id].height
        # 更新货物的长宽高 以及位置
        item_temp = deepcopy(item)
        item.append(Item(length, width, height))
        # print('货物，长：%d，宽：%d，高：%d' % (item[len(item) - 1].length, item[len(item) - 1].width, item[len(item) - 1].height))
        item[len(item) - 1].space_free_id = space_free_id
        # 稳定性与可行性检查（包含旋转选择）
        can_place, best_rotation = can_place_item(space_size, item, len(item) - 1, space_free, space_free_id, space_state)
        if not can_place:
            # 不可放置，回滚并重试
            item = deepcopy(item_temp)
            if t < t_max:
                continue
            else:
                break
        else:
            # 采用返回的最优可行朝向
            item[len(item) - 1].length = best_rotation[0]
            item[len(item) - 1].width = best_rotation[1]
            item[len(item) - 1].height = best_rotation[2]
        # 更新 空闲空间
        '''
        temp1 = [i for i, obj in enumerate(space_free) if (obj.length < item_lwh_min[0])]
        temp2 = [i for i, obj in enumerate(space_free) if (obj.width < item_lwh_min[1])]
        temp3 = [i for i, obj in enumerate(space_free) if (obj.height < item_lwh_min[2])]
        if len(temp1) != 0 or len(temp2) != 0 or len(temp3) != 0:
            # print("空闲空间长宽高有误！")
            pass
        '''
        space_free_new, space_occupied_volume, space_state_volume, space_state_new = place_item(item, len(item) - 1,
                                                                                                space_free,
                                                                                                space_free_id,
                                                                                                space_state,
                                                                                                method=space_manage_method)  # 放置货物
        # 更新空闲空间的最大值
        if len(space_free_new) > space_free_num_split_max:
            space_free_num_split_max = len(space_free_new)
        temp1 = [i for i, obj in enumerate(space_free_new) if (obj.length < item_lwh_min[0])]
        temp2 = [i for i, obj in enumerate(space_free_new) if (obj.width < item_lwh_min[1])]
        temp3 = [i for i, obj in enumerate(space_free_new) if (obj.height < item_lwh_min[2])]
        if len(temp1) != 0 or len(temp2) != 0 or len(temp3) != 0:
            # visualize_packing(data_scale, example_id, space_size, item_temp, 100)  # 货物是一个整体的矩形
            # visualize_packing(data_scale, example_id + 1, space_size, item, 100)  # 货物是一个整体的矩形
            # print("空闲空间长宽高有误！")
            space_occupied_volume = sum(i.volume for i in item if i.placed)
            space_state_volume = np.count_nonzero(space_state_new)
            item = deepcopy(item_temp)
            if t < t_max:
                continue
            else:
                break
        # 还原，放弃这一次的分割
        space_free = space_free_new
        space_state = space_state_new
        # space_free.sort(key=lambda space: space.z)  # 空间按照z轴排序
    # 验证货物长宽高 约束
    temp1 = [i for i, obj in enumerate(item) if (item_lwh_max[0] < obj.length or obj.length < item_lwh_min[0])]
    temp2 = [i for i, obj in enumerate(item) if (item_lwh_max[1] < obj.width or obj.width < item_lwh_min[1])]
    temp3 = [i for i, obj in enumerate(item) if (item_lwh_max[2] < obj.height or obj.height < item_lwh_min[2])]
    temp4 = sum(i.volume for i in item)
    if temp4 != space_size[0] * space_size[1] * space_size[2] or len(temp1) != 0 or len(temp2) != 0 or len(temp3) != 0:
        # print('体积不符合约束！长宽高不符合约束！')
        pass
    else:
        example_id = example_id + 1
        # ========== 记录货物数量并调整 p (最终修正版) ==========
        item_count_history.append(len(item))
        if example_id % p_adjust_interval == 0:
            # 使用所有已产生箱子的货物数量的平均值
            avg_count = np.mean(item_count_history)
            print(f'[调试] 已生成 {example_id} 个样本，近 {p_adjust_interval} 个平均货物数: {avg_count:.2f}, 当前 p: {p:.2f}')
            
            if avg_count > target_avg_item_count:
                # 货物太多，需要增加 p (让不分割概率变大，货物变少)
                if p < p_max:
                    p = min(p_max, p + p_step)
                    print(f'       货物太多 > {target_avg_item_count}，增加 p 为 {p:.2f}')
                else:
                    print(f'       货物太多，但 p 已达上限 {p:.2f}')
            elif avg_count < target_avg_item_count:
                # 货物太少，需要减小 p (让分割概率变高，货物变多)
                if p > 0.05:
                    p = max(0.05, p - p_step)
                    print(f'       货物太少 < {target_avg_item_count}，减小 p 为 {p:.2f}')
                else:
                    print(f'       货物太少，但 p 已达下限 {p:.2f}')
        # =======================================================
        
        print('%d符合约束！' % example_id)
        # 画图
        # visualize_packing(data_scale, example_id, space_size, item, 100)  # 货物是一个整体的矩形
        # 模拟货物顺序 item_num_max
        if len(item) <= item_num_max:
            # 货物少
            order = []
            item_unplaced_order_temp = get_item_order(item, "random")
            item_unplaced_order = deepcopy(item_unplaced_order_temp)
            for i in range(0, len(item)):
                idx = item_unplaced_order.index(i)
                order.append(idx)
                del item_unplaced_order[idx]
        elif len(item) > item_num_max:
            # 货物多
            order = []
            item_unplaced_order_temp = get_item_order(item, "random")
            item_unplaced_order = deepcopy(item_unplaced_order_temp)
            for i in range(0, len(item)):
                idx = item_unplaced_order.index(i)
                if idx >= item_num_max:
                    temp1 = [index for index, value in enumerate(item_unplaced_order[0:item_num_max]) if
                             value > i]  # 窗口内 超过i的下标
                    if len(temp1) > 0:  # 确保temp1不为空
                        temp2 = np.random.randint(0, len(temp1))  # 随机一个下标
                        idx2 = temp1[temp2]  # 在窗口内 找到一个下标j
                        temp3 = item_unplaced_order_temp.index(item_unplaced_order[idx2])  # j对应的值的下标
                        item_unplaced_order[idx], item_unplaced_order[idx2] = item_unplaced_order[idx2], \
                        item_unplaced_order[idx]  # j和i交换
                        idx = idx2
                        temp4 = item_unplaced_order_temp.index(i)  # i对应的值的下标
                        item_unplaced_order_temp[temp4], item_unplaced_order_temp[temp3] = item_unplaced_order_temp[temp3], \
                        item_unplaced_order_temp[temp4]  # 交换i和j
                    else:
                        # 如果temp1为空，说明窗口内没有大于i的值，随机选择一个窗口内的位置交换
                        if item_num_max > 0:
                            idx2 = np.random.randint(0, item_num_max)
                            temp3 = item_unplaced_order_temp.index(item_unplaced_order[idx2])  # j对应的值的下标
                            item_unplaced_order[idx], item_unplaced_order[idx2] = item_unplaced_order[idx2], \
                            item_unplaced_order[idx]  # j和i交换
                            idx = idx2
                            temp4 = item_unplaced_order_temp.index(i)  # i对应的值的下标
                            item_unplaced_order_temp[temp4], item_unplaced_order_temp[temp3] = item_unplaced_order_temp[temp3], \
                            item_unplaced_order_temp[temp4]  # 交换i和j
                order.append(idx)
                del item_unplaced_order[idx]
            # item_unplaced_order = np.arange(0, len(item))
            # item_unplaced_order =
            # while item_unplaced_order:
            # temp = np.random.randint(0, min(item_num_max, len(item_unplaced_order)))
            # order.append(temp)
            # temp1 = np.random.permutation(np.arange(0, item_num_max))

        # 导出结果
        item_new = [[example_id, item_unplaced_order_temp[idx],
                     item[item_unplaced_order_temp[idx]].length, item[item_unplaced_order_temp[idx]].width,
                     item[item_unplaced_order_temp[idx]].height,
                     item[item_unplaced_order_temp[idx]].x, item[item_unplaced_order_temp[idx]].y,
                     item[item_unplaced_order_temp[idx]].z,
                     i, item[item_unplaced_order_temp[idx]].space_free_id, space_free_num_split_max] for idx, i in
                    enumerate(order)]
        data_all = data_all + item_new
# 写入数据
head = ["0数据集编号",
        "1货物序号",
        "2长，最小值" + str(item_lwh_min[0]) + "，最大值" + str(item_lwh_max[0]),
        "3宽，最小值" + str(item_lwh_min[1]) + "，最大值" + str(item_lwh_max[1]),
        "4高，最小值" + str(item_lwh_min[2]) + "，最大值" + str(item_lwh_max[2]),
        "5起始X，箱子长" + str(space_size[0]),
        "6起始Y，箱子宽" + str(space_size[1]),
        "7起始Z，箱子高" + str(space_size[2]),
        "8选择货物的编号",
        "9货物选择空闲空间的编号",
        "10空闲空间的最大值"]
current_script_dir = os.path.dirname(os.path.abspath(__file__))  # 当前脚本路径
# 根据space_manage_method生成不同的文件名
if space_manage_method == 'simple3':
    file_name_suffix = '_our_simple3'
else:
    file_name_suffix = '_our'
    
filename = '数据_' + str(space_size[0]) + str(space_size[1]) + str(space_size[2]) + file_name_suffix + '.xlsx'
    
save_result(filename, head, data_all, '规模' + str(data_scale), current_script_dir)  # 保存主数据

# ========== 新增：写入统计参数 ==========
if len(item_count_history) > 0:
    avg_all = np.mean(item_count_history)
    
    # 计算前80%的平均值
    num_80_percent = int(len(item_count_history) * 0.8)
    avg_80_percent = np.mean(item_count_history[:num_80_percent]) if num_80_percent > 0 else 0
    
    # 计算后20%的平均值
    avg_20_percent = np.mean(item_count_history[num_80_percent:]) if num_80_percent < len(item_count_history) else 0
    
    stats_data = [['统计项', '数值'],
                  ['总样本数', len(item_count_history)],
                  ['所有箱子平均货物数量', avg_all],
                  ['前80%箱子平均货物数量', avg_80_percent],
                  ['后20%箱子平均货物数量', avg_20_percent],
                  ['目标平均货物数量', target_avg_item_count],
                  ['最终p值', p]]
    
    print(f'\n=== 数据统计 ===')
    print(f'总样本数: {len(item_count_history)}')
    print(f'所有箱子平均货物数量: {avg_all:.2f}')
    print(f'前80%箱子平均货物数量: {avg_80_percent:.2f}')
    print(f'后20%箱子平均货物数量: {avg_20_percent:.2f}')
    print(f'目标平均货物数量: {target_avg_item_count}')
    print(f'最终p值: {p:.2f}')
    print(f'================\n')
    
    # 保存统计结果到 "数据参数" sheet
    save_result(filename, stats_data[0], stats_data[1:], '数据参数', current_script_dir)
else:
    print("警告：没有生成任何有效样本！")
# =======================================
time_end = time.time()  # 记录结束的时间
time_dif = time_end - time_begin  # 相差的时间
print(f'时间为:{time_dif / 60:.2f}分钟')
print(f'时间为:{time_dif / 60 / 60:.2f}小时')
print("结束运行！")
