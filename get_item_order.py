from my_imports import random


def get_item_order(item, strategy):
    # 生成货物摆放顺序
    item_num = len(item)
    if strategy == 'random':
        # 随机生成货物的顺序
        item_order = list(range(item_num))
        random.shuffle(item_order)
    elif strategy == 'volume_desc':
        # 按照体积 降序 对货物顺序进行排序
        item_order = sorted(range(item_num), key=lambda x: item[x].volume, reverse=True)
    elif strategy == 'volume_asc':
        # 按照体积 升序 对货物顺序进行排序
        item_order = sorted(range(item_num), key=lambda x: item[x].volume, reverse=False)
    elif strategy == 'mixed':
        # 混合策略：70%的概率按体积排序，30%随机
        if random.random() < 0.7:
            item_order = sorted(range(item_num), key=lambda x: item[x].volume, reverse=True)
        else:
            item_order = list(range(item_num))
            random.shuffle(item_order)
    elif strategy == 'normal':
        item_order = list(range(item_num))

    return item_order
