# 设置固定的空间数量
from typing import Dict, List, Optional, Tuple

from my_imports import SpaceFree, DqnPolicy, DqnPolicy2, DqnPolicy3, DqnPolicy4, DqnPolicy5, DqnPolicy6, np, pd, torch, \
    optim, Categorical, time, deepcopy, os, Item
from generate_items import generate_items
from get_item_order import get_item_order
from can_place_item import can_place_item
from place_item import place_item
from reward_calculate import reward_calculate
from visualize_packing import visualize_packing
from compute_policy_loss import compute_policy_loss
from parameter import parameter
from save_result import save_result
from test1 import test1
from create_directory import create_directory

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

if __name__ == "__main__":
    print(f'\n开始运行')
    time_begin = time.time()  # 记录开始的时间
    # 数据选择 全局变量
    # 1：随机产生数据
    # 2：分割的数据，一个数据训练多次
    # 3：多个数据，每个数据训练一次，100
    # 4：多个数据，每个数据训练一次，非100
    # 5：多个数据，每个数据训练1次，一半监督学习，一半非监督学习
    data_method = 5
    # 空间管理方法选择：'corner' 或 'simple3'
    # 'corner' - 基于角点的最大空间延伸方法（默认，创新点）
    # 'simple3' - 普通3块空间切分方法（消融实验用）
    space_manage_method = 'simple3'  # 可改为 'simple3' 进行消融实验

    space_size, item_lwh_min, item_lwh_max, volume_utilization_mean_max, space_free_num_max, item_place_times_max, item_num_max, space_manage_method = parameter(
        data_method, space_manage_method)
    total_space_volume = space_size[0] * space_size[1] * space_size[2]
    # 判断是否能启用GPU
    if torch.cuda.is_available():
        # 获取GPU数量
        gpu_count = torch.cuda.device_count()
        print(f"GPU数量: {gpu_count}")
        device = torch.device('cuda:0')
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
        file_name = ''  # data_method == 1 时不需要文件
        supervise_ratio = 0  # 监督学习比例0-100
        supervise_quantity = 0  # 监督学习回合数量
        item = generate_items(item_num, item_lwh_min, item_lwh_max, data_method)
        item_new = deepcopy(item)
    elif data_method == 2:
        # 导入数据 训练一组数据
        item_order_strategy = 'random'  # 货物摆放的顺序 'random', 'volume_desc', 'volume_asc', 'mixed', 'normal'
        data_scale = 10  # 数据规模
        example_id = 1  # 算例的编号
        file_name = 'data\数据_101010_our.xlsx'
        supervise_ratio = 0  # 监督学习比例0-100
        supervise_quantity = 0  # 监督学习回合数量
        data = pd.read_excel(file_name, "规模" + str(data_scale))  # 导入数据
        data = np.array(data)  # 将Frame数据转换成Array数据
        max_episode = 1000  # 最大回合数
    elif data_method == 3:
        # 3：100 监督学习
        item_order_strategy = 'normal'  # 货物摆放的顺序 'random', 'volume_desc', 'volume_asc', 'mixed', 'normal'
        data_scale = 10  # 数据规模
        file_name = 'data\数据_101010_our.xlsx'
        supervise_ratio = 100  # 监督学习比例0-100（100表示全部监督学习）
        data = pd.read_excel(file_name, "规模" + str(data_scale))  # 导入数据
        data = np.array(data)  # 将Frame数据转换成Array数据
        max_episode = len(np.unique(data[:, 0]))  # 最大回合数
        supervise_quantity = round(max_episode * supervise_ratio / 100)  # 监督学习回合数量
    elif data_method == 4:
        # 4：非100 非监督学习
        item_order_strategy = 'normal'  # 货物摆放的顺序 'random', 'volume_desc', 'volume_asc', 'mixed', 'normal'
        data_scale = 10  # 数据规模
        file_name = 'data\数据_101010_our.xlsx'
        supervise_ratio = 0  # 监督学习比例0-100
        data = pd.read_excel(file_name, "规模" + str(data_scale))  # 导入数据
        data = np.array(data)  # 将Frame数据转换成Array数据
        max_episode = len(np.unique(data[:, 0]))  # 最大回合数
        supervise_quantity = round(max_episode * supervise_ratio / 100)  # 监督学习回合数量
    elif data_method == 5:
        # 5：一半监督学习，一半非监督学习
        item_order_strategy = 'normal'  # 货物摆放的顺序 'random', 'volume_desc', 'volume_asc', 'mixed', 'normal'
        data_scale = 20  # 数据规模
        # file_name = 'data\数据_101010_our.xlsx'
        file_name = 'data\数据_101010_our_simple3.xlsx'
        data = pd.read_excel(file_name, "规模" + str(data_scale))  # 导入数据
        data = np.array(data)  # 将Frame数据转换成Array数据
        max_episode = len(np.unique(data[:, 0]))  # 最大回合数
        supervise_ratio = 90  # 监督学习比例0-100
        supervise_quantity = round(max_episode * supervise_ratio / 100)  # 监督学习回合数量

    # 设置文件夹名称（可根据需要修改，与 data_scale 等参数一起调整）
    current_script_dir = os.path.dirname(os.path.abspath(__file__))  # 当前脚本路径
    folder_name = '05' + '_训练' + str(max_episode) + '_货物' + str(item_num_max) + '_空间' + str(
        space_free_num_max) + '_监督' + str(supervise_ratio) + '_P6' + '_空间' + str(
        space_manage_method) + '_训练n' + '_测试n'
    folder_name = os.path.join(current_script_dir, folder_name)  # 拼接 路径

    # ========== 修复：在 item_num_max == 1 时，如果模型存在，先备份模型再清空文件夹 ==========
    import shutil
    import tempfile

    model_policy_name_check = os.path.join(folder_name, 'final_dqn_policy.pth')
    model_backup_path = None
    if item_num_max == 1 and os.path.exists(model_policy_name_check):
        # 模型存在，先备份到临时位置
        temp_dir = tempfile.gettempdir()
        model_backup_path = os.path.join(temp_dir, 'final_dqn_policy_backup.pth')
        shutil.copy2(model_policy_name_check, model_backup_path)
        print(f'[保护模型] 检测到模型文件，已备份到临时位置: {model_backup_path}')
    # ===================================================================

    create_directory(folder_name)

    # ========== 修复：如果备份了模型，恢复模型文件 ==========
    if model_backup_path is not None and os.path.exists(model_backup_path):
        shutil.copy2(model_backup_path, model_policy_name_check)
        os.remove(model_backup_path)  # 删除临时备份
        print(f'[保护模型] 已恢复模型文件到: {model_policy_name_check}')
    # ===================================================================

    result_file_name = os.path.join(folder_name, 'result.xlsx')
    policy_name = 'final_dqn_policy.pth'
    if file_name and file_name == 'data\数据_101010_our.xlsx':
        train_or_test = '训练_101010_our_'
    elif file_name and file_name == 'data\数据_101010_our_simple3.xlsx':
        train_or_test = '训练_101010_our_simple3_'
    else:
        train_or_test = '训练_'
    torch.manual_seed(543)  # 使得后续所有随机操作（如 torch.rand()、模型初始化等）生成的结果可重复
    # ========== 参考 Online-3D-BPP-DRL-main：训练时强制 item_num_max=1 ==========
    train_item_num_max = 1  # 训练时不管参数设置如何，都强制使用1
    print(f'[训练模式] 强制设置 item_num_max={train_item_num_max}（参数设置值为 {item_num_max}）')
    # ===================================================================
    # policy = DqnPolicy2(item_num_max=train_item_num_max, space_free_num_max=space_free_num_max).to(device)  # p2
    # policy = DqnPolicy4().to(device)  # p4
    policy = DqnPolicy6(item_num_max=train_item_num_max, space_free_num_max=space_free_num_max).to(device)  # p6
    # 优化器，学习率调度器，动态调整优化器的学习率，间隔step_size调整一次学习率
    # ========== 改进4：根据 item_num_max 调整学习率调度策略 ==========
    # 注意：训练时使用 train_item_num_max（固定为1），所以这里总是使用 else 分支
    if train_item_num_max > 1:
        # item_num_max > 1 时，使用更小的学习率和更频繁的调度
        initial_lr = 3e-5  # 更小的初始学习率（从 5e-5 降到 3e-5）
        step_size = max(1, int(max_episode / 10))  # 每10%训练回合降低一次（更频繁）
        print(
            f'[改进] item_num_max={train_item_num_max}，使用更小的学习率 {initial_lr} 和更频繁的调度（每 {step_size} 回合）')
    else:
        initial_lr = 5e-5  # 保持原值
        step_size = int(max_episode + 1)

    optimizer = optim.Adam(policy.parameters(), lr=initial_lr, weight_decay=1e-5)  # 优化器 学习率（从1e-4降到5e-5，添加权重衰减）
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.9)
    # ===================================================================
    volume_utilization_sum = 0  # 体积利用率 和
    volume_utilization_mean = 0  # 体积利用率 平均
    effective_episode_count = 0  # 实际完成训练的箱子数量（跳过的箱子不计入）
    data_all = []
    summary_all = []  # 新增，用于存summary的数据: [episode, volume_utilization1, item_placed_num, mean_single_item_time]
    episode_items_all = []  # 新增，每箱item对象副本集合
    skipped_supervise_boxes = 0  # 监督学习阶段跳过的箱子数量
    skipped_supervise_records = []  # 记录跳过箱子的具体信息 [episode, reason]

    # 批次更新相关变量
    # 批次大小建议：
    # - 小规模训练（<100个箱子）: 8-16
    # - 中等规模（100-5000个箱子）: 16-32（推荐）
    # - 大规模训练（>5000个箱子）: 32-64（如果内存充足）
    # 当前训练规模：根据max_episode自动调整
    if max_episode < 100:
        batch_size = 8  # 小规模
    elif max_episode < 5000:
        batch_size = 20  # 中等规模（推荐值）
    else:
        batch_size = 32  # 大规模
    print(f'批次大小设置为: {batch_size}（训练规模: {max_episode}个箱子）')
    batch_probs_log1 = []  # 批次缓冲区：存储每个箱子的 probs_log1
    batch_probs_log2 = []  # 批次缓冲区：存储每个箱子的 probs_log2
    batch_rewards_log = []  # 批次缓冲区：存储每个箱子的 rewards_log
    batch_values_log = []  # 批次缓冲区：存储每个箱子的 values_log（状态值V(s)，用于AC算法）
    batch_action_target = []  # 批次缓冲区：存储每个箱子的 action_target
    batch_episodes = []  # 批次缓冲区：存储每个箱子的 episode 编号
    batch_probs_log3 = []  # 批次缓冲区：旋转动作概率日志
    batch_joint_log_corrections = []  # 批次缓冲区：联合采样归一化修正项

    # ========== 新增：根据 item_num_max 决定是否训练 ==========
    if item_num_max == 1:
        # item_num_max == 1 时，先检查是否有模型，有则加载测试，无则训练
        model_policy_name = os.path.join(folder_name, 'final_dqn_policy.pth')

        if os.path.exists(model_policy_name):
            # 模型存在，直接加载进行测试
            print(f'[测试模式] item_num_max={item_num_max}，发现已存在模型，将加载模型进行测试...')
            print(f'[测试模式] 结果将保存到文件夹: {folder_name}')
            do_training = False

            # 设置结果文件路径
            result_file_name = os.path.join(folder_name, 'result.xlsx')
            result_detail_file_name = os.path.join(folder_name, 'result_detail.xlsx')

            # 加载模型（使用 train_item_num_max=1，因为这是 item_num_max=1 的模型）
            print(f'[测试模式] 正在从当前文件夹加载模型: {folder_name}')
            print(f'[测试模式] 模型文件路径: {model_policy_name}')
            policy = DqnPolicy6(item_num_max=train_item_num_max, space_free_num_max=space_free_num_max).to(device)

            # 加载检查点
            checkpoint = torch.load(model_policy_name, map_location=device)

            # 加载模型权重（使用 strict=False 以允许部分不匹配）
            missing_keys, unexpected_keys = policy.load_state_dict(checkpoint, strict=False)
            if missing_keys:
                print(f'[警告] 以下键未加载（使用随机初始化）: {missing_keys}')
            if unexpected_keys:
                print(f'[警告] 以下键在检查点中但不在模型中: {unexpected_keys}')

            policy.eval()  # 设置为评估模式
            print(f'[测试模式] 模型加载完成，使用 item_num_max={train_item_num_max} 进行测试')
        else:
            # 模型不存在，进行训练
            print(f'[训练模式] item_num_max={item_num_max}，未发现模型文件，开始训练...')
            print(f'[训练模式] 结果将保存到文件夹: {folder_name}')
            do_training = True
            model_policy_name = None  # 训练模式下不需要加载模型
    else:
        # item_num_max > 1 时，跳过训练，直接加载 item_num_max=1 的模型
        print(f'[测试模式] item_num_max={item_num_max}，跳过训练，将加载 item_num_max=1 的模型...')
        print(f'[测试模式] 结果将保存到文件夹: {folder_name}')
        do_training = False
        # 构建 item_num_max=1 的文件夹路径（用于加载模型）
        # 基于 folder_name 的逻辑，将 item_num_max 替换为 1
        folder_name_model = folder_name.replace('货物' + str(item_num_max), '货物1')
        model_policy_name = os.path.join(folder_name_model, 'final_dqn_policy.pth')

        # 检查模型文件是否存在
        if not os.path.exists(model_policy_name):
            raise FileNotFoundError(
                f'找不到训练模型文件: {model_policy_name}\n请先运行 item_num_max=1 的训练以生成模型。')

        # 加载模型（注意：测试时使用实际的 item_num_max，而不是 train_item_num_max=1）
        print(f'[测试模式] 正在从 item_num_max=1 的文件夹加载模型: {os.path.dirname(model_policy_name)}')
        print(f'[测试模式] 模型文件路径: {model_policy_name}')
        policy = DqnPolicy6(item_num_max=item_num_max, space_free_num_max=space_free_num_max).to(
            device)  # 使用实际的 item_num_max

        # 加载检查点
        checkpoint = torch.load(model_policy_name, map_location=device)

        # 加载模型权重（使用 strict=False 以允许部分不匹配）
        missing_keys, unexpected_keys = policy.load_state_dict(checkpoint, strict=False)
        if missing_keys:
            print(f'[警告] 以下键未加载（使用随机初始化）: {missing_keys}')
        if unexpected_keys:
            print(f'[警告] 以下键在检查点中但不在模型中: {unexpected_keys}')

        policy.eval()  # 设置为评估模式
        print(f'[测试模式] 模型加载完成，使用 item_num_max={item_num_max} 进行测试')
    # ===================================================================

    if do_training:
        for episode in range(1, max_episode + 1):  # 循环每一个回合
            print(f'回合{episode}开始')
            # ========== 改进：每个episode开始时清空兼容性检查缓存 ==========
            from can_place_item import clear_compatibility_cache

            clear_compatibility_cache()  # 清空缓存，确保每个episode使用全新的状态
            # ===================================================================
            episode_start_time = time.time()  # 记录每个箱子处理开始时间
            single_item_times = []  # 记录每件货物计算时间
            # ========== 改进：在监督学习模式下，按第2列（索引1，货物序号）排序，恢复原始生成顺序 ==========
            # 判断是否为监督学习模式（需要在所有分支之前定义，确保作用域正确）
            is_supervise_mode = (data_method == 3 or (data_method == 5 and episode <= supervise_quantity))
            # ===================================================================
            if data_method == 1:
                item = deepcopy(item_new)
            elif data_method == 2:
                data_temp = data[data[:, 0] == example_id]
                item_num = data_temp.shape[0]  # 货物的数量
                item = generate_items(item_num, data_temp[:, 2:5], None, data_method)
            elif data_method == 3 or data_method == 4 or data_method == 5:
                data_temp = data[data[:, 0] == episode]
                if is_supervise_mode:
                    # 按第2列（索引1，货物序号）排序，恢复原始生成顺序
                    sort_indices = np.argsort(data_temp[:, 1])  # 按货物序号排序
                    data_temp = data_temp[sort_indices]  # 重新排序data_temp
                item_num = data_temp.shape[0]  # 货物的数量
                item = generate_items(item_num, data_temp[:, 2:5], None, data_method)
                example_id = episode
            # 记录本箱初始待装入货物总数
            initial_item_total = len(item)
            # 初始化箱子的空闲空间 连续的空间
            space_free = []
            space_free.append(SpaceFree(0, 0, 0, space_size[0], space_size[1], space_size[2]))
            space_state = np.zeros((space_size[0], space_size[1], space_size[2]))  # 空间的状态
            # 对于每一轮循环，都要重新初始化环境
            # ========== 改进：在监督学习模式下，使用原始顺序[0, 1, 2, ...]，确保与排序后的data_temp一致 ==========
            if is_supervise_mode:
                # 监督学习模式下，使用原始顺序，确保与按货物序号排序后的data_temp一致
                item_order = list(range(len(item)))  # [0, 1, 2, 3, ...]
            else:
                item_order = get_item_order(item, strategy=item_order_strategy)
            # ===================================================================
            probs_log1 = list()  # 将概率列表化
            probs_log2 = list()
            probs_log3 = list()
            rewards_log = list()  # 将奖励列表化
            values_log = list()  # 将状态值V(s)列表化（用于AC算法）
            action_target = list()  # 用于存储监督学习的目标动作s
            joint_log_corrections = list()  # 记录联合采样的归一化对数项
            space_occupied_volume = 0  # 初始化体积
            space_state_volume = 0  # 初始化体积
            item_placed_num = 0  # 记录被摆放的货物数量
            item_unplaced_order = item_order.copy()  # 新的回合 更新所有货物为未摆放
            skip_episode_due_to_supervise = False
            skip_episode_reason = ''
            while item_unplaced_order and len(space_free) > 0 and len(item_unplaced_order) > 0:
                item_start_time = time.time()
                # ========== 参考 Online-3D-BPP-DRL-main：训练时 item_num_max=1，不需要窗口选择 ==========
                # 训练时强制使用 item_num_max=1，所以不需要窗口选择逻辑
                # 窗口选择逻辑已完全注释掉，因为训练时 train_item_num_max 总是为1
                # ===================================================================

                volume_utilization_step = 0  # 记录每一步的空间利用率，应该越高越好
                # 通过神经网络的前向传播，获取动作概率分布probs
                # item_probs, space_probs = policy(item, item_unplaced_order, space_free, device)  # p2
                # item_probs, space_probs = policy(item, item_unplaced_order, space_free, item_num_max, space_free_num_max, device)  # p4
                item_probs, space_probs, rotation_probs, feasibility_mask, rotation_dims, state_value = policy(
                    item,
                    item_unplaced_order,
                    space_free,
                    device,
                    space_size,
                    space_state,
                )  # p6 with rotation and critic

                # 收集状态值V(s)用于AC算法
                values_log.append(state_value)

                # 基于神经网络给出的概率分布采样/选择动作
                m1 = Categorical(item_probs)
                m2 = Categorical(space_probs)
                m3 = Categorical(rotation_probs)

                is_rl_mode = (
                        data_method == 1
                        or data_method == 2
                        or data_method == 4
                        or (data_method == 5 and episode > supervise_quantity)
                )
                joint_selection = False
                joint_score_sum = None
                cached_rotation_dims = None
                cached_can_place_flag = None
                feasibility_cache: Dict[Tuple[int, int, int], Tuple[bool, Optional[List[float]]]] = {}

                valid_item = min(train_item_num_max, len(item_unplaced_order))
                valid_space = min(space_free_num_max, len(space_free))
                rotation_count = rotation_probs.shape[1] if rotation_probs is not None else 1

                fm_slice = None
                rd_slice = None
                if feasibility_mask is not None and feasibility_mask.numel() > 0:
                    fm_slice = feasibility_mask[:valid_item, :valid_space, :rotation_count]
                if rotation_dims is not None and rotation_dims.numel() > 0:
                    rd_slice = rotation_dims[:valid_item, :valid_space, :rotation_count, :]

                if fm_slice is not None and rd_slice is not None and fm_slice.numel() > 0:
                    for i in range(valid_item):
                        for j in range(valid_space):
                            for rot in range(rotation_count):
                                mask_value = bool(fm_slice[i, j, rot].item())
                                if mask_value:
                                    dims_tensor = rd_slice[i, j, rot]
                                    dims_list = [int(v) for v in dims_tensor.tolist()]
                                    feasibility_cache[(i, j, rot)] = (True, dims_list)
                                else:
                                    feasibility_cache[(i, j, rot)] = (False, None)

                if is_rl_mode:
                    joint_candidates = []
                    joint_scores = []

                    for i in range(valid_item):
                        pi = item_probs[0, i]
                        if pi.item() <= 0:
                            continue
                        for j in range(valid_space):
                            pj = space_probs[0, j]
                            if pj.item() <= 0:
                                continue
                            for rot in range(rotation_count):
                                pr = rotation_probs[0, rot] if rotation_probs is not None else pi.new_tensor(1.0)
                                if pr.item() <= 0:
                                    continue
                                if (i, j, rot) not in feasibility_cache:
                                    can_place_flag, rotation_candidate = can_place_item(
                                        space_size,
                                        item,
                                        item_unplaced_order[i],
                                        space_free,
                                        j,
                                        space_state,
                                        restrict_lw=True,  # 只使用2个旋转方向
                                        rotation_action=rot,
                                    )
                                    if can_place_flag and rotation_candidate is not None:
                                        rotation_candidate = [int(v) for v in rotation_candidate]
                                    feasibility_cache[(i, j, rot)] = (can_place_flag, rotation_candidate)
                                else:
                                    can_place_flag, rotation_candidate = feasibility_cache[(i, j, rot)]
                                if not can_place_flag:
                                    continue
                                score = pi * pj * pr
                                joint_candidates.append((i, j, rot, rotation_candidate))
                                joint_scores.append(score)

                    if joint_candidates:
                        joint_scores_tensor = torch.stack(joint_scores)
                        score_sum = joint_scores_tensor.sum()
                        if score_sum.item() > 0:
                            joint_probs = joint_scores_tensor / score_sum
                            joint_dist = Categorical(joint_probs)
                            joint_idx = joint_dist.sample()
                            selected_i, selected_j, selected_rot, rotation_candidate = joint_candidates[
                                joint_idx.item()]
                            action1 = torch.tensor(selected_i, device=device)
                            action2 = torch.tensor(selected_j, device=device)
                            action3 = torch.tensor(selected_rot, device=device)
                            cached_rotation_dims = rotation_candidate
                            cached_can_place_flag = True
                            joint_score_sum = score_sum
                            joint_selection = True
                            # 联合选择模式下，也需要设置item_id和space_free_id
                            # 添加边界检查，确保索引有效
                            if selected_i < 0 or selected_i >= len(item_unplaced_order):
                                error_msg = (
                                    f'错误：联合选择模式下selected_i={selected_i}超出item_unplaced_order范围（长度={len(item_unplaced_order)}）。')
                                raise IndexError(error_msg)
                            if selected_j < 0 or selected_j >= len(space_free):
                                error_msg = (
                                    f'错误：联合选择模式下selected_j={selected_j}超出space_free范围（长度={len(space_free)}）。')
                                raise IndexError(error_msg)
                            item_id = item_unplaced_order[selected_i]
                            space_free_id = selected_j

                if not joint_selection:
                    if data_method == 1 or data_method == 2:
                        action1 = m1.sample()
                        action2 = m2.sample()
                        action3 = m3.sample()
                    elif data_method == 3 or (data_method == 5 and episode <= supervise_quantity):
                        # 100 监督学习
                        # 注意：data_temp已按第2列（货物序号）排序，item_order也是原始顺序[0, 1, 2, ...]
                        # 因此第9列（索引8）和第10列（索引9）的读取顺序与排序后的data_temp一致
                        # 检查item_placed_num是否在有效范围内
                        if item_placed_num >= len(data_temp):
                            error_msg = (
                                f'错误：item_placed_num={item_placed_num}超出data_temp范围（长度={len(data_temp)}）。'
                                f'监督学习模式下数据索引超出范围，请检查数据一致性。episode={episode}')
                            if is_supervise_mode:
                                skip_episode_due_to_supervise = True
                                if not skip_episode_reason:
                                    skip_episode_reason = '监督标签数量与货物数量不一致'
                                print(error_msg)
                                break
                            else:
                                raise IndexError(error_msg)

                        # 第9列（索引8）：货物选择索引（在item_unplaced_order中的位置）
                        # 注意：虽然训练时item_num_max=1，但item_unplaced_order会随着放置而减少，所以仍需要此索引
                        action1_target = data_temp[item_placed_num, 8]
                        action1 = torch.tensor(action1_target, device=device)

                        # 检查action1_target是否在item_unplaced_order的有效范围内
                        if action1.item() < 0 or action1.item() >= len(item_unplaced_order):
                            error_msg = (
                                f'错误：action1_target={action1.item()}超出item_unplaced_order范围（长度={len(item_unplaced_order)}）。'
                                f'监督学习标签索引无效。episode={episode}, item_placed_num={item_placed_num}')
                            if is_supervise_mode:
                                skip_episode_due_to_supervise = True
                                if not skip_episode_reason:
                                    skip_episode_reason = '监督标签item索引无效'
                                print(error_msg)
                                break
                            else:
                                raise IndexError(error_msg)

                        item_id = item_unplaced_order[action1.item()]

                        # 第10列（索引9）：空间选择索引（在space_free中的位置）
                        # 使用item_placed_num作为索引，确保与按货物序号排序后的data_temp一致
                        action2_target = data_temp[item_placed_num, 9]
                        action2 = torch.tensor(action2_target, device=device)

                        # 检查action2_target是否在space_free的有效范围内
                        if action2.item() < 0 or action2.item() >= len(space_free):
                            error_msg = (
                                f'错误：action2_target={action2.item()}超出space_free范围（长度={len(space_free)}）。'
                                f'监督学习标签索引无效。episode={episode}, item_placed_num={item_placed_num}')
                            if is_supervise_mode:
                                skip_episode_due_to_supervise = True
                                if not skip_episode_reason:
                                    skip_episode_reason = '监督标签空间索引无效'
                                print(error_msg)
                                break
                            else:
                                raise IndexError(error_msg)

                        space_free_id = action2.item()
                        # 旋转监督标签：数据集中无标签，默认使用0（第一种排列）
                        action3_target = 0
                        action3 = torch.tensor(action3_target, device=device)
                        action_target.append((action1_target, action2_target, action3_target))
                    elif data_method == 4 or (data_method == 5 and episode > supervise_quantity):
                        action1 = m1.sample()
                        action2 = m2.sample()
                        action3 = m3.sample()

                if not (data_method == 3 or (data_method == 5 and episode <= supervise_quantity)):
                    # 非监督学习模式下，需要检查action1和action2的边界
                    if action1.item() < 0 or action1.item() >= len(item_unplaced_order):
                        error_msg = (
                            f'错误：action1.item()={action1.item()}超出item_unplaced_order范围（长度={len(item_unplaced_order)}）。'
                            f'episode={episode}')
                        raise IndexError(error_msg)
                    if action2.item() < 0 or action2.item() >= len(space_free):
                        error_msg = (f'错误：action2.item()={action2.item()}超出space_free范围（长度={len(space_free)}）。'
                                     f'episode={episode}')
                        raise IndexError(error_msg)
                    item_id = item_unplaced_order[action1.item()]
                    space_free_id = action2.item()
                log_prob_item = None
                log_prob_space = None
                log_prob_rotation = None
                joint_log_correction_value = None
                if is_rl_mode:
                    log_prob_item = m1.log_prob(action1)
                    log_prob_space = m2.log_prob(action2)
                    log_prob_rotation = m3.log_prob(action3)
                    if joint_score_sum is not None:
                        joint_log_correction_value = torch.log(joint_score_sum + 1e-12)
                        joint_log_correction_value = joint_log_correction_value.view_as(log_prob_item)
                    else:
                        joint_log_correction_value = torch.zeros_like(log_prob_item)

                # 装入货物
                # 修复：使用更清晰的边界检查逻辑
                # action1.item()应该在[0, min(train_item_num_max, len(item_unplaced_order))-1]范围内
                # action2.item()应该在[0, min(space_free_num_max, len(space_free))-1]范围内
                valid_item_max = min(train_item_num_max, len(item_unplaced_order))
                valid_space_max = min(space_free_num_max, len(space_free))
                if action1.item() >= 0 and action1.item() < valid_item_max and \
                        action2.item() >= 0 and action2.item() < valid_space_max:
                    # 重要：在删除 item_unplaced_order 之前判断能否放入
                    # 因为 can_place_item 需要与模型推理时使用相同的状态
                    if cached_can_place_flag is not None and cached_rotation_dims is not None:
                        can_place = cached_can_place_flag
                        rotation = cached_rotation_dims
                    else:
                        cache_key = (action1.item(), action2.item(), action3.item())
                        cache_entry = feasibility_cache.get(cache_key)
                        if cache_entry is not None:
                            can_place, rotation = cache_entry
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
                            feasibility_cache[cache_key] = (can_place, rotation)
                    # 监督学习模式下，在判断 can_place 之前处理旋转
                    if data_method == 3 or (data_method == 5 and episode <= supervise_quantity):
                        # 100 监督学习
                        # 按监督标签映射到固定2种旋转（索引恒定）：只允许长宽互换
                        dims = (item[item_id].length, item[item_id].width, item[item_id].height)
                        index_perms = [
                            (0, 1, 2),  # 0: (L,W,H) 原始方向
                            (1, 0, 2),  # 1: (W,L,H) 长宽互换
                        ]
                        idx = max(0, min(action3.item(), len(index_perms) - 1))
                        i, j, k = index_perms[idx]
                        rotation = [dims[i], dims[j], dims[k]]
                        # 监督学习模式下，总是使用监督标签指定的旋转重新检查
                        # 因为监督学习的旋转可能与之前RL模式检查的旋转不同
                        can_place, _ = can_place_item(
                            space_size,
                            item,
                            item_id,
                            space_free,
                            space_free_id,
                            space_state,
                            restrict_lw=False,
                            rotation_action=action3.item(),
                        )
                        # 更新缓存（使用监督学习的旋转）
                        cache_key = (action1.item(), action2.item(), action3.item())
                        feasibility_cache[cache_key] = (can_place, rotation)

                    # 无论成功失败，都要先从列表中删除，避免重复
                    # 如果失败且需要重试，会在失败处理中重新插入
                    # 再次检查索引有效性（虽然前面已经检查过，但为了安全起见）
                    if action1.item() >= 0 and action1.item() < len(item_unplaced_order):
                        del item_unplaced_order[action1.item()]
                    else:
                        error_msg = (f'错误：尝试删除item_unplaced_order时索引无效。'
                                     f'action1.item()={action1.item()}, len(item_unplaced_order)={len(item_unplaced_order)}')
                        raise IndexError(error_msg)
                    if can_place:
                        # 判断箱子能否放入
                        # print(f'成功++++++++++++++++++++')
                        # print(f'货物{item_id}长宽高{item[item_id].length} {item[item_id].width} {item[item_id].height} 体积{item[item_id].volume}')
                        # print(f'空间{space_free_id}长宽高{space_free[space_free_id].length} {space_free[space_free_id].width} {space_free[space_free_id].height}')
                        # 货物放入箱子，拆分空闲空间
                        item[item_id].length, item[item_id].width, item[item_id].height = rotation  # 货物旋转，更新 货物长宽高
                        item[item_id].volume = item[item_id].length * item[item_id].width * item[item_id].height  # 更新体积
                        volume_utilization_step = item[item_id].volume / space_free[
                            space_free_id].volume  # 当前货物占用当前空闲空间的利用率
                        # ========== 改进：在放置前保存空间信息用于计算边缘奖励 ==========
                        # 注意：place_item 会删除原空间并创建新空间，所以要在放置前保存
                        selected_space = space_free[space_free_id]  # 保存选中的空间信息（用于边缘奖励计算）
                        # ===================================================================
                        space_free_new, space_occupied_volume, space_state_volume, space_state_new = place_item(item,
                                                                                                                item_id,
                                                                                                                space_free,
                                                                                                                space_free_id,
                                                                                                                space_state,
                                                                                                                method=space_manage_method)  # 放置货物
                        space_free = space_free_new
                        space_state = space_state_new
                        # ========== 改进：传递空间坐标信息以计算底部装载奖励 ==========
                        # 使用放置前的空间信息（selected_space）计算底部奖励
                        # 创建一个临时空间列表用于奖励计算
                        temp_space_free_for_reward = [selected_space]
                        reward_total = reward_calculate(space_size, volume_utilization_step, space_state_volume,
                                                        can_place, space_free=temp_space_free_for_reward,
                                                        space_free_id=0,
                                                        item_volume=item[item_id].volume,
                                                        # 添加货物体积参数（参考参考代码的box_ratio设计）
                                                        item_num_max=train_item_num_max,
                                                        item_unplaced_order=item_unplaced_order,
                                                        item=item, selected_item_id=item_id)  # 计算奖励（参考参考代码的简洁设计）
                        # ===================================================================
                        if is_rl_mode:
                            # 非监督学习
                            probs_log1.append(log_prob_item)  # 保存每次行动对应的概率分布
                            probs_log2.append(log_prob_space)
                            probs_log3.append(log_prob_rotation)
                            correction = (
                                joint_log_correction_value
                                if joint_log_correction_value is not None
                                else torch.zeros_like(log_prob_item)
                            )
                            joint_log_corrections.append(correction)
                        elif data_method == 3 or (data_method == 5 and episode <= supervise_quantity):
                            # 监督学习
                            # probs_log1.append(torch.log(item_probs))  # 完整的对数概率分布
                            # probs_log2.append(torch.log(space_probs))  # 完整的对数概率分布
                            probs_log1.append(item_probs)  # 完整的概率分布
                            probs_log2.append(space_probs)  # 完整的概率分布
                            probs_log3.append(rotation_probs)  # 完整的概率分布

                        rewards_log.append(reward_total)  # 保存每次行动对应的奖励
                        item_placed_num = item_placed_num + 1
                        # 只在货物成功放入后，统计装载该货物用时
                        single_item_times.append(time.time() - item_start_time)
                        # 删除不满足货物最小尺寸的空间（优化，item_unplaced批量和空间筛选均为numpy/列表推导）
                        item_placed_flags = np.array([it.placed for it in item])
                        item_length = np.array([it.length for it in item])
                        item_width = np.array([it.width for it in item])
                        item_height = np.array([it.height for it in item])
                        item_unplaced_mask = ~item_placed_flags
                        if np.any(item_unplaced_mask):
                            min_length = item_length[item_unplaced_mask].min()
                            min_width = item_width[item_unplaced_mask].min()
                            min_height = item_height[item_unplaced_mask].min()
                            min_lwh = min(min_length, min_width, min_height)
                            space_free = [sp for sp in space_free if
                                          sp.length >= min_lwh and sp.width >= min_lwh and sp.height >= min_lwh]
                    else:
                        # 放入失败
                        # print(f'失败--------------------')
                        # print(f'货物{item_id}长宽高{item[item_id].length} {item[item_id].width} {item[item_id].height} 体积{item[item_id].volume}')
                        # print(f'空间{space_free_id}长宽高{space_free[space_free_id].length} {space_free[space_free_id].width} {space_free[space_free_id].height}')

                        item[item_id].place_times += 1  # 增加该货物的放置次数
                        if item[item_id].place_times < item_place_times_max:
                            # ========== 改进：失败后插入队列末尾，避免窗口内顺序混乱 ==========
                            # 插入队列末尾，让其他货物有机会被选择，避免陷入失败循环
                            item_unplaced_order.append(item_id)  # 插入队列末尾
                            # ===================================================================
                            # 不append时间，失败不统计
                        else:
                            item[item_id].place_times_max = True
                            # 不append时间，失败不统计
                        # 更新probs
                        if is_rl_mode:
                            # 非监督学习
                            probs_log1.append(log_prob_item)  # 保存每次行动对应的概率分布
                            probs_log2.append(log_prob_space)
                            probs_log3.append(log_prob_rotation)
                            correction = (
                                joint_log_correction_value
                                if joint_log_correction_value is not None
                                else torch.zeros_like(log_prob_item)
                            )
                            joint_log_corrections.append(correction)
                        elif data_method == 3 or (data_method == 5 and episode <= supervise_quantity):
                            # 监督学习
                            # 检查item_placed_num是否在有效范围内（失败时item_placed_num还未递增）
                            if item_placed_num >= len(data_temp):
                                error_msg = (
                                    f'错误：item_placed_num={item_placed_num}超出data_temp范围（长度={len(data_temp)}）。'
                                    f'监督学习模式下数据索引超出范围，请检查数据一致性。episode={episode}')
                                raise IndexError(error_msg)

                            # probs_log1.append(torch.log(item_probs))  # 完整的对数概率分布
                            # probs_log2.append(torch.log(space_probs))  # 完整的对数概率分布
                            probs_log1.append(item_probs)  # 完整的概率分布
                            probs_log2.append(space_probs)  # 完整的概率分布
                            probs_log3.append(rotation_probs)  # 完整的概率分布
                            # 失败时也添加action_target，确保长度与probs_log一致
                            action_target.append((action1_target, action2_target, action3_target))  # 添加目标动作（即使是失败的）
                        # ========== 改进：传递空间坐标信息以计算底部装载奖励（失败情况） ==========
                        # 失败时也传递item_volume，虽然不会使用，但保持接口一致性
                        item_volume_fail = item[item_id].volume if item_id is not None and item_id < len(item) else None
                        reward_total = reward_calculate(space_size, volume_utilization_step, space_state_volume,
                                                        can_place, space_free=space_free, space_free_id=space_free_id,
                                                        item_volume=item_volume_fail,  # 添加货物体积参数（参考参考代码的box_ratio设计）
                                                        item_num_max=train_item_num_max,
                                                        item_unplaced_order=item_unplaced_order,
                                                        item=item, selected_item_id=item_id)  # 计算奖励（参考参考代码的简洁设计）
                        # ===================================================================
                        rewards_log.append(reward_total)  # 保存每次行动对应的奖励
                        # break
                        continue
                else:
                    print(f'货物编号，空间编号有误！')
                    if is_supervise_mode:
                        skip_episode_due_to_supervise = True
                        if not skip_episode_reason:
                            skip_episode_reason = '空间编号无效'
                    # 不append时间，异常不统计
                    break
            supervise_failed = False
            if is_supervise_mode:
                fully_loaded = (space_occupied_volume == total_space_volume)
                all_items_placed = (item_placed_num == initial_item_total)
                if not fully_loaded or not all_items_placed:
                    supervise_failed = True
                    if not skip_episode_reason:
                        skip_episode_reason = '未达到100%装载率'
            if is_supervise_mode and (skip_episode_due_to_supervise or supervise_failed):
                skipped_supervise_boxes += 1
                skipped_supervise_records.append(
                    [episode, skip_episode_reason if skip_episode_reason else '原因未记录'])
                print(
                    f'[监督学习] 回合{episode}被跳过，原因：{skip_episode_reason if skip_episode_reason else "未知"}；'
                    f'累计跳过{skipped_supervise_boxes}个箱子')
                continue

            # 体积利用率
            volume_utilization1 = space_occupied_volume / space_size[0] / space_size[1] / space_size[2] * 100
            volume_utilization2 = space_state_volume / space_size[0] / space_size[1] / space_size[2] * 100
            volume_utilization_sum = volume_utilization_sum + volume_utilization1
            effective_episode_count += 1
            if effective_episode_count > 0:
                volume_utilization_mean = volume_utilization_sum / effective_episode_count
            else:
                volume_utilization_mean = 0
            mean_single_item_time = np.mean(single_item_times) if single_item_times else 0

            # 统计本箱summary
            summary_all.append(
                [episode, volume_utilization1, item_placed_num, mean_single_item_time, initial_item_total])
            # episode_items_all.append(deepcopy(item))  # 新增，存本回合item对象副本
            episode_items_all.append(
                [(i.length, i.width, i.height, i.x, i.y, i.z, i.placed) for i in item])  # 新增，只存关键参数快照

            if volume_utilization_mean > volume_utilization_mean_max:
                # 当达到目标利用率时
                print(f'训练成功!')
                # 处理剩余批次数据后再退出
                if len(batch_episodes) > 0:
                    print(f'\n训练成功提前结束，处理剩余批次数据（{len(batch_episodes)}个箱子）')
                    # 计算批次损失：对每个箱子的损失求平均
                    batch_loss_sum = torch.tensor(0.0, device=device, requires_grad=True)
                    valid_batch_count = 0

                    for i in range(len(batch_episodes)):
                        episode_i = batch_episodes[i]
                        loss_i = compute_policy_loss(device, batch_probs_log1[i], batch_probs_log2[i],
                                                     batch_probs_log3[i],
                                                     batch_rewards_log[i], data_method, episode_i,
                                                     supervise_quantity, batch_action_target[i],
                                                     batch_joint_log_corrections[i],
                                                     batch_values_log[i] if i < len(batch_values_log) else None)

                        # 检查损失是否有效
                        loss_temp = loss_i.cpu().detach().numpy()
                        if not (np.isnan(loss_temp).any() or np.isinf(loss_temp).any()):
                            batch_loss_sum = batch_loss_sum + loss_i
                            valid_batch_count += 1

                    if valid_batch_count > 0:
                        # 计算平均损失
                        batch_loss = batch_loss_sum / valid_batch_count
                        print(
                            f'剩余批次更新：箱子 {batch_episodes[0]}-{batch_episodes[-1]}，批次损失={batch_loss.item():.20f}')

                        # 梯度下降算法
                        optimizer.zero_grad()  # 先将梯度归零
                        batch_loss.backward()  # 求出所有参数的梯度，计算损失函数对模型参数的梯度

                        # 检查梯度
                        grad_valid = True
                        for param in policy.parameters():
                            if param.grad is not None and (
                                    torch.isnan(param.grad).any() or torch.isinf(param.grad).any()):
                                print(f'梯度包含 NaN 或 Inf！')
                                grad_valid = False
                                break

                        if grad_valid:
                            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)  # 启用梯度裁剪防止梯度爆炸
                            optimizer.step()  # 反向传播，根据新的梯度，更新模型的参数
                            scheduler.step()  # 调整学习率
                        else:
                            # 梯度无效时的处理
                            print(f'跳过剩余批次参数更新')
                            optimizer.zero_grad()  # 清除无效梯度
                # 跳出回合循环，结束训练。因为已经存在某个回合训练超过5000次还没有失败，训练成功
                break

            if len(probs_log1) <= 0 or len(probs_log2) <= 0 or len(probs_log3) <= 0:
                print(f'probs_log长度有误！')
                # 如果数据无效，跳过这个箱子，不添加到批次中
                print(
                    f'回合{episode}结束，space_occupied_volume体积利用率为{volume_utilization1:.2f}%，体积平均利用率为{volume_utilization_mean:.2f}%')
                print(f'space_state_volume体积利用率为{volume_utilization2:.2f}%，跳过批次更新（数据无效）')
                # 导出结果
                item = [[i.length, i.width, i.height, i.x, i.y, i.z, i.placed, volume_utilization1, episode] for i in
                        item]
                data_all = data_all + item
                continue

            # 将当前箱子的数据添加到批次缓冲区
            batch_probs_log1.append(probs_log1)
            batch_probs_log2.append(probs_log2)
            batch_probs_log3.append(probs_log3)
            batch_rewards_log.append(rewards_log)
            batch_values_log.append(values_log)  # 添加状态值V(s)用于AC算法
            batch_action_target.append(action_target)
            batch_episodes.append(episode)
            batch_joint_log_corrections.append(joint_log_corrections)

            print(
                f'回合{episode}结束，space_occupied_volume体积利用率为{volume_utilization1:.2f}%，体积平均利用率为{volume_utilization_mean:.2f}%')
            print(
                f'space_state_volume体积利用率为{volume_utilization2:.2f}%，批次进度: {len(batch_episodes)}/{batch_size}')

            # 当达到批次大小时，执行批次更新
            if len(batch_episodes) >= batch_size:
                # 计算批次损失：对每个箱子的损失求平均
                batch_loss_sum = torch.tensor(0.0, device=device, requires_grad=True)
                valid_batch_count = 0

                for i in range(len(batch_episodes)):
                    episode_i = batch_episodes[i]
                    loss_i = compute_policy_loss(device, batch_probs_log1[i], batch_probs_log2[i], batch_probs_log3[i],
                                                 batch_rewards_log[i], data_method, episode_i,
                                                 supervise_quantity, batch_action_target[i],
                                                 batch_joint_log_corrections[i],
                                                 batch_values_log[i] if i < len(batch_values_log) else None)

                    # 检查损失是否有效
                    loss_temp = loss_i.cpu().detach().numpy()
                    if not (np.isnan(loss_temp).any() or np.isinf(loss_temp).any()):
                        batch_loss_sum = batch_loss_sum + loss_i
                        valid_batch_count += 1

                if valid_batch_count > 0:
                    # 计算平均损失
                    batch_loss = batch_loss_sum / valid_batch_count
                    print(f'批次更新：箱子 {batch_episodes[0]}-{batch_episodes[-1]}，批次损失={batch_loss.item():.20f}')

                    # 梯度下降算法
                    optimizer.zero_grad()  # 先将梯度归零
                    batch_loss.backward()  # 求出所有参数的梯度，计算损失函数对模型参数的梯度
                    # 检查梯度
                    grad_valid = True
                    for param in policy.parameters():
                        if param.grad is not None and (torch.isnan(param.grad).any() or torch.isinf(param.grad).any()):
                            print(f'梯度包含 NaN 或 Inf！')
                            grad_valid = False
                            break

                    if grad_valid:
                        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)  # 启用梯度裁剪防止梯度爆炸
                        optimizer.step()  # 反向传播，根据新的梯度，更新模型的参数
                        scheduler.step()  # 调整学习率
                    else:
                        # 梯度无效时的处理
                        print(f'跳过本轮批次参数更新')
                        optimizer.zero_grad()  # 清除无效梯度

                    # 清空批次缓冲区（无论是否更新，都清空本批次）
                    batch_probs_log1 = []
                    batch_probs_log2 = []
                    batch_probs_log3 = []
                    batch_rewards_log = []
                    batch_values_log = []
                    batch_action_target = []
                    batch_episodes = []
                    batch_joint_log_corrections = []
                else:
                    # 梯度无效时的处理
                    print(f'批次更新跳过：所有箱子的损失都无效')

            # 画图
            '''
            if episode == max_episode or 1 == 2:
                visualize_packing(data_scale, example_id, space_size, item, volume_utilization1, volume_utilization_mean,
                                  train_or_test, folder_name)  # 货物是一个整体的矩形
            '''
            # 导出结果
            item = [[i.length, i.width, i.height, i.x, i.y, i.z, i.placed, volume_utilization1, episode] for i in item]
            data_all = data_all + item

    if do_training:
        # 结束循环前，处理剩余的批次数据（不足batch_size个箱子的情况）
        if len(batch_episodes) > 0:
            print(f'\n训练结束，处理剩余批次数据（{len(batch_episodes)}个箱子）')
            # 计算批次损失：对每个箱子的损失求平均
            batch_loss_sum = torch.tensor(0.0, device=device, requires_grad=True)
            valid_batch_count = 0

            for i in range(len(batch_episodes)):
                episode_i = batch_episodes[i]
                loss_i = compute_policy_loss(device, batch_probs_log1[i], batch_probs_log2[i], batch_probs_log3[i],
                                             batch_rewards_log[i], data_method, episode_i,
                                             supervise_quantity, batch_action_target[i], batch_joint_log_corrections[i],
                                             batch_values_log[i] if i < len(batch_values_log) else None)

                # 检查损失是否有效
                loss_temp = loss_i.cpu().detach().numpy()
                if not (np.isnan(loss_temp).any() or np.isinf(loss_temp).any()):
                    batch_loss_sum = batch_loss_sum + loss_i
                    valid_batch_count += 1

            if valid_batch_count > 0:
                # 计算平均损失
                batch_loss = batch_loss_sum / valid_batch_count
                print(f'剩余批次更新：箱子 {batch_episodes[0]}-{batch_episodes[-1]}，批次损失={batch_loss.item():.20f}')

                # 梯度下降算法
                optimizer.zero_grad()  # 先将梯度归零
                batch_loss.backward()  # 求出所有参数的梯度，计算损失函数对模型参数的梯度

                # 检查梯度
                grad_valid = True
                for param in policy.parameters():
                    if param.grad is not None and (torch.isnan(param.grad).any() or torch.isinf(param.grad).any()):
                        print(f'梯度包含 NaN 或 Inf！')
                        grad_valid = False
                        break

                if grad_valid:
                    torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)  # 启用梯度裁剪防止梯度爆炸
                    optimizer.step()  # 反向传播，根据新的梯度，更新模型的参数
                    scheduler.step()  # 调整学习率
                else:
                    # 梯度无效时的处理
                    print(f'跳过剩余批次参数更新')
                    optimizer.zero_grad()  # 清除无效梯度

        # 结束循环
        # 保存训练过程每箱summary
        result_file_name = os.path.join(folder_name, 'result.xlsx')
        result_detail_file_name = os.path.join(folder_name, 'result_detail.xlsx')
        # head = ["长", "宽", "高", "起始x", "起始y", "起始z", "是否放置", "体积利用率", "数据集编号"]
        # save_result(result_file_name, head, data_all, train_or_test + str(data_scale), folder_name)  # 保存详细结果
        summary_head = ["箱子编号", "体积利用率", "货物数量", "平均单货物装载时间", "初始货物数量"]
        save_result(result_file_name, summary_head, summary_all, train_or_test + str(data_scale), folder_name)
        if skipped_supervise_boxes > 0:
            skip_head = ["箱子编号", "跳过原因"]
            save_result(result_file_name, skip_head, skipped_supervise_records, "supervise_skip", folder_name)
        # 汇总统计均值写 summary 表
        # ------ 修正：全局加权平均所有成功装载货物耗时 ------
        total_time = 0
        total_count = 0
        for row in summary_all:
            # row[3]: mean_single_item_time，row[2]: item_placed_num
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
        save_result(result_file_name,
                    ["类型", "平均体积利用率", "平均装载货物数", "平均装载货物时间", "平均箱子初始货物数量"],
                    [summary_avg_row], "summary", folder_name)
        if skipped_supervise_boxes > 0:
            print(f'[监督学习] 共跳过{skipped_supervise_boxes}个箱子，详情见 result.xlsx 的 supervise_skip 表')

        # 保存模型，模型名称为：final_dqn_policy
        # 注意：无论是否有剩余批次，只要进行了训练，都要保存模型
        policy_name_save = os.path.join(folder_name, policy_name)  # 拼接 模型路径
        torch.save(policy.state_dict(), policy_name_save)
        print(f'[训练完成] 模型已保存到: {policy_name_save}')
        time_end = time.time()  # 记录结束的时间
        time_dif = time_end - time_begin  # 相差的时间

        # 找到最接近平均装箱率的箱子id
        if len(summary_all) > 0:
            best_idx = np.argmin(np.abs([x[1] - avg_util for x in summary_all]))
            best_episode = summary_all[best_idx][0]
            # 直接用对象数组
            item_best_obj = episode_items_all[best_idx]
            # 保存明细到sheet: folder_name + "detail"
            sheet_name = train_or_test + str(data_scale) + "_detail"
            head = ["长", "宽", "高", "起始x", "起始y", "起始z", "是否放置", "体积利用率", "数据集编号"]
            # 先将tuple快照恢复为Item对象列表，并补充赋值（x, y, z, placed）
            item_best_obj_objects = []
            for i in item_best_obj:
                obj = Item(i[0], i[1], i[2])
                obj.x = i[3]
                obj.y = i[4]
                obj.z = i[5]
                obj.placed = i[6]
                item_best_obj_objects.append(obj)
            # 明细生成也直接用tuple下标
            detail_data = [[i[0], i[1], i[2], i[3], i[4], i[5], i[6], summary_all[best_idx][1], best_episode] for i in
                           item_best_obj]
            save_result(result_detail_file_name, head, detail_data, sheet_name, folder_name)
            visualize_packing(data_scale, best_episode, space_size, item_best_obj_objects, summary_all[best_idx][1],
                              np.mean([x[1] for x in summary_all]), train_or_test, folder_name)

        print(f'训练时间为:{time_dif / 60:.2f}分钟')
        print(f'训练时间为:{time_dif / 60 / 60:.2f}小时')
        print(f'结束运行\n')

        # 设置模型路径用于后续测试
        policy_name = policy_name_save
    else:
        # item_num_max > 1 时，不进行训练，直接进行测试
        print(f'[测试模式] 跳过训练，直接进行测试')
        result_file_name = os.path.join(folder_name, 'result.xlsx')
        result_detail_file_name = os.path.join(folder_name, 'result_detail.xlsx')
        policy_name = os.path.join(folder_name, 'final_dqn_policy.pth')  # 使用当前文件夹的模型路径（虽然不会保存）

    # 无论是否训练，都进行测试
    # 确定使用的模型路径：如果未训练，使用 item_num_max=1 的模型；如果训练了，使用当前训练的模型
    test_policy_name = model_policy_name if not do_training else policy_name
    test1(2000, 'data\数据_101010_our.xlsx', result_file_name, result_detail_file_name, test_policy_name, folder_name,
          space_manage_method)
    test1(2100, 'data\数据_101010_rs.xlsx', result_file_name, result_detail_file_name, test_policy_name, folder_name,
          space_manage_method)
    test1(2100, 'data\数据_101010_cut1.xlsx', result_file_name, result_detail_file_name, test_policy_name, folder_name,
          space_manage_method)
    test1(2100, 'data\数据_101010_cut2.xlsx', result_file_name, result_detail_file_name, test_policy_name, folder_name,
          space_manage_method)
    test1(3000, 'data\数据_101010_3k.xlsx', result_file_name, result_detail_file_name, test_policy_name, folder_name,
          space_manage_method)
    test1(2000, 'data\数据_101010_our_simple3.xlsx', result_file_name, result_detail_file_name, test_policy_name,
          folder_name, space_manage_method)

    # test1(2000, 'data\数据_101010_our.xlsx', result_file_name, result_detail_file_name, policy_name, folder_name)
    # test1(10000, 'data\数据_101010_cut2_1w.xlsx', result_file_name, result_detail_file_name, policy_name, folder_name)

    # test1(2100, 'data\数据_202010_cut2.xlsx', result_file_name, result_detail_file_name, policy_name, folder_name)
    # test1(2100, 'data\数据_303010_cut2.xlsx', result_file_name, result_detail_file_name, policy_name, folder_name)

    # test1(5, 'data\数据_101010_3k.xlsx', result_file_name, result_detail_file_name, policy_name, folder_name)
    # test1(5, 'data\数据_101010_cut2.xlsx', result_file_name, result_detail_file_name, policy_name, folder_name)
