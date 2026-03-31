from my_imports import np, Item


def generate_items(item_num, item_lwh_min, item_lwh_max, data_method):
    # 产生货物数据
    if data_method == 1:
        # 随机生成货物数据
        item = []  # 存储货物的信息
        for _ in range(item_num):
            length = np.random.randint(item_lwh_min[0], item_lwh_max[0] + 1)
            width = np.random.randint(item_lwh_min[1], item_lwh_max[1] + 1)
            height = np.random.randint(item_lwh_min[2], item_lwh_max[2] + 1)
            item.append(Item(length, width, height))
        return item
    elif data_method == 2 or data_method == 3 or data_method == 4 or data_method == 5:
        # 导入数据 训练一组数据
        item = []  # 存储货物的信息
        for i in range(0, item_num):
            length = int(item_lwh_min[i, 0])
            width = int(item_lwh_min[i, 1])
            height = int(item_lwh_min[i, 2])
            item.append(Item(length, width, height))
        return item
