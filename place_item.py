from my_imports import np, SpaceFree, deepcopy
from calculate_max_area import calculate_max_area


def place_item_corner(item, item_id, space_free, space_free_id, space_state):
    """
    基于角点的最大空间延伸方法（现有创新点实现）
    """
    space_free_new = deepcopy(space_free)
    space_state_new = deepcopy(space_state)
    # 放入货物，更新货物数据
    item[item_id].x = space_free_new[space_free_id].x
    item[item_id].y = space_free_new[space_free_id].y
    item[item_id].z = space_free_new[space_free_id].z
    item[item_id].placed = True

    # 放入货物，更新空间占用情况
    x_start = space_free_new[space_free_id].x
    y_start = space_free_new[space_free_id].y
    z_start = space_free_new[space_free_id].z
    x_end = x_start + item[item_id].length
    y_end = y_start + item[item_id].width
    z_end = z_start + item[item_id].height
    space_state_new[x_start:x_end, y_start:y_end, z_start:z_end] = 1

    # 拆分原来的空间
    temp = space_free_new[space_free_id]  # 存储该空间的信息
    del space_free_new[space_free_id]  # 将空间信息删除
    space_generate_num = 0
    # 更新上方的空间
    if temp.height >= item[item_id].height:
        lwh = calculate_max_area(space_state_new, temp.x, temp.y, temp.z + item[item_id].height)
        if lwh[0] != 0 and lwh[1] != 0 and lwh[2] != 0:
            space_free_new.append(SpaceFree(temp.x, temp.y, temp.z + item[item_id].height, lwh[0], lwh[1], lwh[2]))
            space_generate_num = space_generate_num + 1

    # 更新右边的空间
    if temp.width >= item[item_id].width:
        lwh = calculate_max_area(space_state_new, temp.x, temp.y + item[item_id].width, temp.z)
        if lwh[0] != 0 and lwh[1] != 0 and lwh[2] != 0:
            space_free_new.append(SpaceFree(temp.x, temp.y + item[item_id].width, temp.z, lwh[0], lwh[1], lwh[2]))
            space_generate_num = space_generate_num + 1

    # 更新前面的空间
    if temp.length >= item[item_id].length:
        lwh = calculate_max_area(space_state_new, temp.x + item[item_id].length, temp.y, temp.z)
        if lwh[0] != 0 and lwh[1] != 0 and lwh[2] != 0:
            space_free_new.append(SpaceFree(temp.x + item[item_id].length, temp.y, temp.z, lwh[0], lwh[1], lwh[2]))
            space_generate_num = space_generate_num + 1

    # 检验其他空间是否因为货物放入而改变大小或被覆盖，删除被覆盖的空间，修改被改变大小的空间
    i = 0
    while i < len(space_free_new) - space_generate_num:
        lwh = calculate_max_area(space_state_new, space_free_new[i].x, space_free_new[i].y, space_free_new[i].z)
        if lwh[0] == 0 and lwh[1] == 0 and lwh[2] == 0:
            # 角点已被占用，删除该角点
            del space_free_new[i]
        else:
            # 更新空闲空间的大小
            space_free_new[i].length = lwh[0]
            space_free_new[i].width = lwh[1]
            space_free_new[i].height = lwh[2]
            space_free_new[i].volume = lwh[0] * lwh[1] * lwh[2]
            i = i + 1

    # 删除子集空间：如果空间A完全包含在空间B内，则删除空间A
    def is_space_subset(space_a, space_b):
        # 判断空间A是否是空间B的子集
        # 空间A的起始点
        a_x1, a_y1, a_z1 = space_a.x, space_a.y, space_a.z
        # 空间A的结束点
        a_x2, a_y2, a_z2 = a_x1 + space_a.length, a_y1 + space_a.width, a_z1 + space_a.height

        # 空间B的起始点
        b_x1, b_y1, b_z1 = space_b.x, space_b.y, space_b.z
        # 空间B的结束点
        b_x2, b_y2, b_z2 = b_x1 + space_b.length, b_y1 + space_b.width, b_z1 + space_b.height

        # 检查A是否完全包含在B内
        return (b_x1 <= a_x1 and a_x2 <= b_x2 and
                b_y1 <= a_y1 and a_y2 <= b_y2 and
                b_z1 <= a_z1 and a_z2 <= b_z2)

    # 标记需要删除的空间索引
    spaces_to_remove = set()
    for i in range(len(space_free_new)):
        if i in spaces_to_remove:
            continue
        for j in range(len(space_free_new)):
            if i != j and j not in spaces_to_remove:
                # 如果空间i是空间j的子集，标记删除空间i
                if is_space_subset(space_free_new[i], space_free_new[j]):
                    spaces_to_remove.add(i)
                    break

    # 从后往前删除，避免索引变化影响
    for index in sorted(spaces_to_remove, reverse=True):
        del space_free_new[index]
    # 验证体积正确性
    space_occupied_volume = sum(i.volume for i in item if i.placed)
    space_state_volume = np.count_nonzero(space_state_new)

    if space_occupied_volume == space_state_volume:
        # print(f'货物{item_id}体积检验无误，space_occupied_volume={space_occupied_volume}，space_state_volume={space_state_volume}')
        pass
    else:
        print(f'货物{item_id}体积检验有误，space_occupied_volume={space_occupied_volume}，space_state_volume={space_state_volume}')
        pass
    return space_free_new, space_occupied_volume, space_state_volume, space_state_new


def place_item_simple3(item, item_id, space_free, space_free_id, space_state):
    """
    普通空间管理方法（消融实验用）：
    货物放入后，把原空间简单几何切分成3个不重叠的长方体空间（不使用calculate_max_area）。

    切分方式：
    - 右侧空间: [x+l, x+L) × [y, y+W) × [z, z+H)
    - 前方空间: [x, x+l) × [y+w, y+W) × [z, z+H)
    - 上方空间: [x, x+l) × [y, y+w) × [z+h, z+H)
    这三个空间在x,y,z至少一维上不重叠。
    """
    space_free_new = deepcopy(space_free)
    space_state_new = deepcopy(space_state)

    # 1. 放入货物，更新货物数据
    cur_space = space_free_new[space_free_id]
    item[item_id].x = cur_space.x
    item[item_id].y = cur_space.y
    item[item_id].z = cur_space.z
    item[item_id].placed = True

    # 2. 更新体素占用
    x0 = cur_space.x
    y0 = cur_space.y
    z0 = cur_space.z
    L = cur_space.length
    W = cur_space.width
    H = cur_space.height

    l = item[item_id].length
    w = item[item_id].width
    h = item[item_id].height

    x_end = x0 + l
    y_end = y0 + w
    z_end = z0 + h
    space_state_new[x0:x_end, y0:y_end, z0:z_end] = 1

    # 3. 从空间列表中删除被拆分的那个空间
    del space_free_new[space_free_id]

    # 4. 按几何关系切3个不重叠空间
    # A: 右侧空间 [x0+l, x0+L) × [y0, y0+W) × [z0, z0+H)
    if L - l > 0:
        sf_a = SpaceFree(x0 + l, y0, z0, L - l, W, H)
        sf_a.volume = (L - l) * W * H
        space_free_new.append(sf_a)

    # B: 前方空间 [x0, x0+l) × [y0+w, y0+W) × [z0, z0+H)
    if W - w > 0:
        sf_b = SpaceFree(x0, y0 + w, z0, l, W - w, H)
        sf_b.volume = l * (W - w) * H
        space_free_new.append(sf_b)

    # C: 上方空间 [x0, x0+l) × [y0, y0+w) × [z0+h, z0+H)
    if H - h > 0:
        sf_c = SpaceFree(x0, y0, z0 + h, l, w, H - h)
        sf_c.volume = l * w * (H - h)
        space_free_new.append(sf_c)

    # 5. 体积校验
    space_occupied_volume = sum(i.volume for i in item if i.placed)
    space_state_volume = np.count_nonzero(space_state_new)

    if space_occupied_volume != space_state_volume:
        print(f'[simple3] 货物{item_id}体积检验有误，space_occupied_volume={space_occupied_volume}，space_state_volume={space_state_volume}')

    return space_free_new, space_occupied_volume, space_state_volume, space_state_new


def place_item(item, item_id, space_free, space_free_id, space_state, method='corner'):
    """
    统一入口：根据method选择使用哪种空间管理方法。

    Args:
        item: 货物列表
        item_id: 货物索引
        space_free: 空闲空间列表
        space_free_id: 选中的空闲空间索引
        space_state: 空间占用状态
        method: 空间管理方法
            'corner' - 基于角点的最大空间延伸方法（默认）
            'simple3' - 普通3块空间切分方法

    Returns:
        space_free_new, space_occupied_volume, space_state_volume, space_state_new
    """
    if method == 'simple3':
        return place_item_simple3(item, item_id, space_free, space_free_id, space_state)
    else:
        return place_item_corner(item, item_id, space_free, space_free_id, space_state)
