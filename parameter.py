def parameter(data_method, space_manage_method='corner'):
    """
    参数说明：
    data_method: 数据生成方法 (1-5)
    space_manage_method: 空间管理方法
        'corner' - 基于角点的最大空间延伸方法（默认，现有创新点）
        'simple3' - 普通3块空间切分方法（消融实验用）
    """
    # 算法参数
    space_size = [10, 10, 10]  # 空间尺寸
    item_lwh_min = [2, 2, 2]  # 货物最小尺寸
    item_lwh_max = [5, 5, 5]  # 货物最大尺寸

    item_num_max = 1  # 货物的最大数量
    space_free_num_max = 100  # 空闲空间的数量 一个很大的数
    item_place_times_max = 5  # 每个货物最大尝试摆放的次数
    if data_method == 1 or data_method == 2 or data_method == 4:
        # 非监督学习
        volume_utilization_mean_max = 99  # 最大平均利用率
    elif data_method == 3 or data_method == 5:
        # 监督学习
        volume_utilization_mean_max = 100  # 最大平均利用率

    return space_size, item_lwh_min, item_lwh_max, volume_utilization_mean_max, space_free_num_max, item_place_times_max, item_num_max, space_manage_method
