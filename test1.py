from my_imports import SpaceFree, DqnPolicy, DqnPolicy2, DqnPolicy3, DqnPolicy4, DqnPolicy5, DqnPolicy6, np, pd, torch, deepcopy, time, F, Categorical, Item
from generate_items import generate_items
from get_item_order import get_item_order
from can_place_item import can_place_item
from place_item import place_item
from visualize_packing import visualize_packing
from parameter import parameter
from save_result import save_result


def test1(_data_scale, _file_name, _result_file_name, _result_detail_file_name, _policy_name, _folder_name, _space_manage_method='corner'):
    # if __name__ == "__main__":
    print(f'\n开始运行')
    time_begin = time.time()  # 记录开始的时间
    # 数据选择 全局变量
    data_method = 3
    space_manage_method = _space_manage_method  # 空间管理方法：'corner' 或 'simple3'
    space_size, item_lwh_min, item_lwh_max, volume_utilization_mean_max, space_free_num_max, item_place_times_max, item_num_max, space_manage_method = parameter(data_method, space_manage_method)
    # 判断是否能启用GPU
    if torch.cuda.is_available():
        device = torch.device('cpu')  # 不是bug，无需修改，使用GPU cuda:0
        print(f'当前的设备是GPU. \n\n')
    else:
        device = torch.device('cpu')
        print(f'当前的设备是CPU. \n\n')

    # 生成数据
    if data_method == 1:
        # 随机产生货物数据
        data_scale = 1  # 数据规模
        example_id = 1  # 算例的编号
        item_order_strategy = 'random'  # 货物摆放的顺序 'random', 'volume_desc', 'volume_asc', 'mixed', 'normal'
        item_num = 50  # 货物的数量
        max_episode = 10  # 最大回合数
        item = generate_items(item_num, item_lwh_min, item_lwh_max, data_method)
        item_new = deepcopy(item)
    elif data_method == 2:
        # 导入数据 训练一组数据
        item_order_strategy = 'normal'  # 货物摆放的顺序 'random', 'volume_desc', 'volume_asc', 'mixed', 'normal'
        data_scale = 10  # 数据规模
        file_name = '数据101010_our.xlsx'
        data = pd.read_excel(file_name, "规模" + str(data_scale))  # 导入数据
        data = np.array(data)  # 将Frame数据转换成Array数据
        max_episode = data_scale  # 最大回合数
    elif data_method == 3 or data_method == 4 or data_method == 5:
        item_order_strategy = 'normal'  # 货物摆放的顺序 'random', 'volume_desc', 'volume_asc', 'mixed', 'normal'
        data_scale = _data_scale  # 数据规模
        file_name = _file_name
        data = pd.read_excel(file_name, "规模" + str(data_scale))  # 导入数据
        data = np.array(data)  # 将Frame数据转换成Array数据
        max_episode = len(np.unique(data[:, 0]))  # 最大回合数

    policy_name = _policy_name
    # 可视化控制：每一步装载后是否绘图（1==1 开启，1==2 关闭）
    enable_step_visualization = (1 == 2)
    if file_name == 'data\数据_101010_3k.xlsx':
        train_or_test = '测试_101010_3k_'
    elif file_name == 'data\数据_101010_rs.xlsx':
        train_or_test = '测试_101010_rs_'
    elif file_name == 'data\数据_101010_cut1.xlsx':
        train_or_test = '测试_101010_cut1_'
    elif file_name == 'data\数据_101010_cut2.xlsx':
        train_or_test = '测试_101010_cut2_'
    elif file_name == 'data\数据_202010_cut2.xlsx':
        train_or_test = '测试_202010_cut2_'
        space_size = [20, 20, 10]
    elif file_name == 'data\数据_303010_cut2.xlsx':
        train_or_test = '测试_303010_cut2_'
        space_size = [30, 30, 10]
    elif file_name == 'data\数据_101010_our.xlsx':
        train_or_test = '测试_101010_our_'
    elif file_name == 'data\数据_101010_our_simple3.xlsx':
        train_or_test = '测试_101010_our_simple3_'
    elif file_name == 'data\数据_101010_cut2_1w.xlsx':
        train_or_test = '测试_101010_cut2_1w_'
    # DQN
    data_all = []
    summary_all = []  # 新增，用于summary统计
    episode_items_all = []  # 新增，每箱item对象副本集合
    volume_utilization_sum = 0  # 体积利用率 和
    volume_utilization_mean = 0  # 体积利用率 平均
    for episode in range(1, max_episode + 1):  # 循环每一个回合
        print(f'回合{episode}开始')
        # ========== 改进：每个episode开始时清空兼容性检查缓存 ==========
        from can_place_item import clear_compatibility_cache
        clear_compatibility_cache()  # 清空缓存，确保每个episode使用全新的状态
        # ===================================================================
        episode_start_time = time.time()  # 记录本箱子处理开始时间
        single_item_times = []
        if data_method == 1:
            item = deepcopy(item_new)
        elif data_method == 2:
            data_temp = data[data[:, 0] == episode]
            item_num = data_temp.shape[0]  # 货物的数量
            item = generate_items(item_num, data_temp[:, 2:5], None, data_method)
            example_id = episode
        elif data_method == 3 or data_method == 4 or data_method == 5:
            data_temp = data[data[:, 0] == episode]
            item_num = data_temp.shape[0]  # 货物的数量
            item = generate_items(item_num, data_temp[:, 2:5], None, data_method)
            example_id = episode
        # 记录本箱初始待装入货物总数
        initial_item_total = len(item)
        # 模型
        # policy = DqnPolicy(input_dim=input_dim, output_dim=output_dim).to(device)  # p
        # policy = DqnPolicy2(item_num_max=item_num_max, space_free_num_max=space_free_num_max).to(device)  # p2
        # policy = DqnPolicy4().to(device)  # p4
        policy = DqnPolicy6(item_num_max=item_num_max, space_free_num_max=space_free_num_max).to(device)  # p6
        policy.load_state_dict(torch.load(policy_name))  # 加载模型 dqn_policy  final_dqn_policy
        policy.eval()  # 将模型设置为评估模式
        # 开始摆放箱子
        # 初始化箱子的空闲空间 连续的空间
        space_free = []
        space_free.append(SpaceFree(0, 0, 0, space_size[0], space_size[1], space_size[2]))
        space_state = np.zeros((space_size[0], space_size[1], space_size[2]))  # 空间的状态
        # 对于每一轮循环，都要重新初始化环境
        item_order = get_item_order(item, strategy=item_order_strategy)
        space_occupied_volume = 0  # 初始化体积
        space_state_volume = 0  # 初始化体积
        item_placed_num = 0  # 记录被摆放的货物数量
        item_unplaced_order = item_order.copy()  # 新的回合 更新所有货物为未摆放
        while item_unplaced_order and len(space_free) > 0 and len(item_unplaced_order) > 0:
            item_start_time = time.time()
            # ========== 参考 Online-3D-BPP-DRL-main：测试时当 item_num_max > 1 时，按体积排序后只传入最大货物 ==========
            # 如果 item_num_max > 1，先对窗口内的货物按体积排序，然后将最大的货物传入网络
            if item_num_max > 1:
                window_size = min(item_num_max, len(item_unplaced_order))
                if window_size > 1:
                    # 对窗口内的货物按体积降序排序
                    window_items = item_unplaced_order[:window_size]
                    window_items_sorted = sorted(window_items, key=lambda idx: item[idx].volume, reverse=True)
                    # 只保留最大的货物，其他货物保持原顺序
                    # 将最大货物放到第一个位置，其他货物保持原顺序
                    max_item = window_items_sorted[0]
                    # 从原列表中移除最大货物
                    item_unplaced_order.remove(max_item)
                    # 将最大货物插入到第一个位置
                    item_unplaced_order.insert(0, max_item)
                    # 实际传入网络的货物数量为1（只传入最大货物）
                    test_item_num_max = 1
                else:
                    test_item_num_max = 1
            else:
                test_item_num_max = item_num_max
            # ===================================================================

            volume_utilization_step = 0  # 记录每一步的空间利用率，应该越高越好
            # 循环一个回合里的每一步，循环放入货物
            # 通过神经网络的前向传播，获取动作概率分布probs
            # item_probs, space_probs = policy(item, item_unplaced_order, space_free, device)  # DqnPolicy2
            # item_probs, space_probs = policy(item, item_unplaced_order, space_free, item_num_max, space_free_num_max, device)  # DqnPolicy4
            with torch.no_grad():
                (
                    item_probs,
                    space_probs,
                    rotation_probs,
                    feasibility_mask,
                    rotation_dims,
                    state_value,  # AC算法返回的状态值V(s)，测试时不需要使用
                ) = policy(
                    item,
                    item_unplaced_order,
                    space_free,
                    device,
                    space_size,
                    space_state,
                )  # p6
            # 测试时使用"联合可行性贪婪"：在可行集合内选择 item-空间-旋转 的联合最高分
            # 注意：如果 item_num_max > 1，test_item_num_max 已经被设置为1（只传入最大货物）
            valid_item = min(test_item_num_max, len(item_unplaced_order))
            valid_space = min(space_free_num_max, len(space_free))
            
            # 修复：边界情况检查，如果没有有效的货物或空间，直接跳出循环
            if valid_item == 0 or valid_space == 0:
                print(f'没有有效的货物或空间：valid_item={valid_item}, valid_space={valid_space}')
                break
            
            best_score = -1.0
            best_i = None
            best_j = None
            best_rot = None
            rotation_count = rotation_probs.shape[1] if rotation_probs is not None else 1
            fm_slice = None
            rd_slice = None
            if feasibility_mask is not None and feasibility_mask.numel() > 0:
                fm_slice = feasibility_mask[:valid_item, :valid_space, :rotation_count]
            if rotation_dims is not None and rotation_dims.numel() > 0:
                rd_slice = rotation_dims[:valid_item, :valid_space, :rotation_count, :]
            # 遍历可行索引，选择联合分数最高且可放置的组合
            for i in range(valid_item):
                cur_item_id = item_unplaced_order[i]
                pi = float(item_probs[0, i])
                if pi <= 0:
                    continue
                for j in range(valid_space):
                    pj = float(space_probs[0, j])
                    if pj <= 0:
                        continue
                    best_rot_local = None
                    best_rot_local_score = -1.0
                    for rot in range(rotation_count):
                        mask_flag = False
                        if fm_slice is not None and fm_slice.numel() > 0:
                            mask_flag = bool(fm_slice[i, j, rot].item())
                        else:
                            fallback_flag, _ = can_place_item(
                                space_size,
                                item,
                                cur_item_id,
                                space_free,
                                j,
                                space_state,
                                restrict_lw=True,  # 只使用2个旋转方向
                                rotation_action=rot,
                            )
                            mask_flag = fallback_flag
                        if not mask_flag:
                            continue
                        pr = float(rotation_probs[0, rot]) if rotation_probs is not None else 1.0
                        joint_score = pi * pj * pr
                        if joint_score > best_rot_local_score:
                            best_rot_local_score = joint_score
                            best_rot_local = rot
                    if best_rot_local is not None and best_rot_local_score > best_score:
                        best_score = best_rot_local_score
                        best_i = i
                        best_j = j
                        best_rot = best_rot_local

            # 如果找到可行的最高分组合则采用，否则退回到边际贪婪
            if best_i is not None:
                action1 = torch.tensor([best_i])
                action2 = torch.tensor([best_j])
                action3 = torch.tensor([best_rot])
            else:
                # 修复：限制argmax的范围，只在实际有效的范围内选择
                # 避免选择超出实际货物/空间数量的索引
                if valid_item > 0:
                    action1 = torch.argmax(item_probs[:, :valid_item], dim=1)
                else:
                    print(f'错误：valid_item={valid_item}，无法选择货物')
                    break
                if valid_space > 0:
                    action2 = torch.argmax(space_probs[:, :valid_space], dim=1)
                else:
                    print(f'错误：valid_space={valid_space}，无法选择空间')
                    break
                action3 = torch.argmax(rotation_probs, dim=1) if rotation_probs is not None else torch.tensor([0])
            
            # 修复：在访问索引之前，先检查action1和action2是否有效
            if action1 is None or action2 is None:
                print(f'错误：action1或action2为None，无法继续')
                break
            
            # 修复：检查索引是否在有效范围内
            if action1.item() < 0 or action1.item() >= len(item_unplaced_order):
                error_msg = (f'错误：action1.item()={action1.item()}超出item_unplaced_order范围（长度={len(item_unplaced_order)}）。')
                raise IndexError(error_msg)
            if action2.item() < 0 or action2.item() >= len(space_free):
                error_msg = (f'错误：action2.item()={action2.item()}超出space_free范围（长度={len(space_free)}）。')
                raise IndexError(error_msg)
            
            item_id = item_unplaced_order[action1.item()]  # 货物序号
            space_free_id = action2.item()  # 选择的空闲空间的编号
            
            # 修复：确保索引在有效范围内后再删除
            if action1.item() >= 0 and action1.item() < len(item_unplaced_order) and \
               action2.item() >= 0 and action2.item() < len(space_free):
                del item_unplaced_order[action1.item()]
                cache_mask_flag = False
                if (
                    fm_slice is not None
                    and rd_slice is not None
                    and action1.item() < valid_item
                    and action2.item() < valid_space
                    and action3.item() < rotation_count
                ):
                    cache_mask_flag = bool(fm_slice[action1.item(), action2.item(), action3.item()].item())
                if cache_mask_flag:
                    can_place = True
                    rotation_tensor = rd_slice[action1.item(), action2.item(), action3.item()]
                    rotation = [int(v) for v in rotation_tensor.tolist()]
                else:
                    can_place, rotation = can_place_item(
                        space_size,
                        item,
                        item_id,
                        space_free,
                        space_free_id,
                        space_state,
                        restrict_lw=True,  # 只使用2个旋转方向
                        rotation_action=action3.item(),
                    )
                    if can_place and rotation is not None:
                        rotation = [int(v) for v in rotation]
                if can_place:
                    # 判断箱子能否放入
                    # print(f'成功++++++++++++++++++++')
                    # print(f'货物{item_id}长宽高{item[item_id].length} {item[item_id].width} {item[item_id].height} 体积{item[item_id].volume}')
                    # print(f'空间{space_free_id}长宽高{space_free[space_free_id].length} {space_free[space_free_id].width} {space_free[space_free_id].height}')
                    # 货物放入箱子，拆分空闲空间
                    item[item_id].length, item[item_id].width, item[item_id].height = rotation  # 货物旋转，更新 货物长宽高
                    item[item_id].volume = (
                        item[item_id].length * item[item_id].width * item[item_id].height
                    )  # 同步更新体积
                    volume_utilization_step = item[item_id].volume / space_free[space_free_id].volume  # 当前货物占用当前空闲空间的利用率
                    space_free_new, space_occupied_volume, space_state_volume, space_state_new = place_item(item, item_id, space_free, space_free_id, space_state, method=space_manage_method)  # 放置货物
                    space_free = space_free_new
                    space_state = space_state_new
                    item_placed_num = item_placed_num + 1
                    # 只在货物成功放入后，统计装载该货物用时
                    single_item_times.append(time.time() - item_start_time)

                    # 每一步装载后绘制该箱子的当前装载状态（可开关）
                    if enable_step_visualization:
                        current_util1 = space_occupied_volume / space_size[0] / space_size[1] / space_size[2] * 100
                        # 临时平均利用率（包含当前箱子进行中的状态）
                        current_mean_util = (volume_utilization_sum + current_util1) / episode
                        visualize_packing(data_scale, episode, space_size, item, current_util1, current_mean_util, train_or_test, _folder_name)

                    # 删除不满足货物最小尺寸的空间（优化，与训练代码保持一致）
                    item_placed_flags = np.array([it.placed for it in item])
                    item_length = np.array([it.length for it in item])
                    item_width = np.array([it.width for it in item])
                    item_height = np.array([it.height for it in item])
                    item_unplaced_mask = ~item_placed_flags
                    if np.any(item_unplaced_mask):
                        min_length = item_length[item_unplaced_mask].min()
                        min_width  = item_width[item_unplaced_mask].min()
                        min_height = item_height[item_unplaced_mask].min()
                        min_lwh = min(min_length, min_width, min_height)
                        space_free = [sp for sp in space_free if sp.length >= min_lwh and sp.width >= min_lwh and sp.height >= min_lwh]
                else:
                    # 放入失败
                    # print(f'失败--------------------')
                    # print(f'货物{item_id}长宽高{item[item_id].length} {item[item_id].width} {item[item_id].height} 体积{item[item_id].volume}')
                    # 修复：失败时不再使用space_free_id，因为可能已经无效
                    # print(f'空间{space_free_id}长宽高{space_free[space_free_id].length} {space_free[space_free_id].width} {space_free[space_free_id].height}')

                    item[item_id].place_times += 1  # 增加该货物的放置次数
                    if item[item_id].place_times < item_place_times_max:
                        # ========== 改进：失败后插入队列末尾，避免窗口内顺序混乱 ==========
                        # 插入队列末尾，让其他货物有机会被选择，避免陷入失败循环
                        item_unplaced_order.append(item_id)  # 插入队列末尾
                        # ===================================================================
                        # 不append时间，失败不统计
                    else:
                        # 无法再放置
                        item[item_id].place_times_max = True
                        # 不append时间，失败不统计
                    # break
                    continue
            else:
                print(f'空间数量有误！')
                # 不append时间，异常不统计
                break
        # 体积利用率
        volume_utilization1 = space_occupied_volume / space_size[0] / space_size[1] / space_size[2] * 100
        volume_utilization2 = space_state_volume / space_size[0] / space_size[1] / space_size[2] * 100
        volume_utilization_sum = volume_utilization_sum + volume_utilization1
        volume_utilization_mean = volume_utilization_sum / episode
        mean_single_item_time = np.mean(single_item_times) if single_item_times else 0
        # 保存summary
        summary_all.append([episode, volume_utilization1, item_placed_num, mean_single_item_time, initial_item_total])
        episode_items_all.append([(i.length, i.width, i.height, i.x, i.y, i.z, i.placed) for i in item])  # 新增，只存关键参数快照（与train1.py保持一致）
        print(f'回合{episode}结束，space_occupied_volume体积利用率为{volume_utilization1:.2f}%，体积平均利用率为{volume_utilization_mean:.2f}%')
        print(f'space_state_volume体积利用率为{volume_utilization2:.2f}%')

        '''
        # 画图
        if episode == max_episode or 1 == 2:
            visualize_packing(data_scale, episode, space_size, item, volume_utilization1, volume_utilization_mean, train_or_test, _folder_name)  # 货物是一个整体的矩形
        '''
        
        # 导出结果
        item = [[i.length, i.width, i.height, i.x, i.y, i.z, i.placed, volume_utilization1, episode] for i in item]
        data_all = data_all + item
    # 结束循环
    summary_head = ["箱子编号", "体积利用率", "货物数量", "平均单货物装载时间", "初始货物数量"]
    save_result(_result_file_name, summary_head, summary_all, train_or_test + str(data_scale), _folder_name)

    
    # 汇总行写summary作为一新行append
    # ------ 修正：全局加权平均所有成功装载货物耗时 ------
    total_time = 0
    total_count = 0
    for row in summary_all:
        total_time += row[3] * row[2]
        total_count += row[2]
    avg_util = np.mean([x[1] for x in summary_all])
    # 平均装载货物数（保持原有指标不变）
    total_placed_items = np.sum([x[2] for x in summary_all])
    box_count = len(summary_all)
    avg_num = (total_placed_items / box_count) if box_count else 0
    # 新增：平均箱子初始货物数量
    total_initial_items = np.sum([x[4] for x in summary_all])
    avg_initial_num = (total_initial_items / box_count) if box_count else 0
    avg_item_time = total_time / total_count if total_count else 0
    summary_avg_row = [train_or_test + str(data_scale), avg_util, avg_num, avg_item_time, avg_initial_num]
    save_result(_result_file_name, ["类型", "平均体积利用率", "平均装载货物数", "平均装载货物时间", "平均箱子初始货物数量"], [summary_avg_row], "summary", _folder_name)
    

    # 找到最接近平均装箱率的箱子id（只计算一次）
    best_idx = np.argmin(np.abs([x[1] - avg_util for x in summary_all]))
    best_episode = summary_all[best_idx][0]
    # 将tuple快照恢复为Item对象列表
    item_best_obj_tuples = episode_items_all[best_idx]
    item_best_obj = []
    for i in item_best_obj_tuples:
        obj = Item(i[0], i[1], i[2])
        obj.x = i[3]
        obj.y = i[4]
        obj.z = i[5]
        obj.placed = i[6]
        item_best_obj.append(obj)
    visualize_packing(data_scale, best_episode, space_size, item_best_obj, summary_all[best_idx][1], np.mean([x[1] for x in summary_all]), train_or_test, _folder_name)


    # 保存最接近平均装箱率那箱的明细数据（复用已计算的变量）
    detail_sheet_name = train_or_test + str(data_scale) + "_detail"
    head = ["长", "宽", "高", "起始x", "起始y", "起始z", "是否放置", "体积利用率", "数据集编号"]
    detail_data = [[i.length, i.width, i.height, i.x, i.y, i.z, i.placed, summary_all[best_idx][1], best_episode] for i in item_best_obj]
    save_result(_result_detail_file_name, head, detail_data, detail_sheet_name, _folder_name)

    time_end = time.time()  # 记录结束的时间
    time_dif = time_end - time_begin  # 相差的时间
    print(f'测试时间为:{time_dif / 60:.2f}分钟')
    print(f'测试时间为:{time_dif / 60 / 60:.2f}小时')
    print(f'结束运行\n')
