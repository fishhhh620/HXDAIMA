def calculate_max_area(space_state, x, y, z):
    """
    优化版本：使用更高效的算法计算最大立方体
    采用逐层扩展的方法，一旦遇到占用空间就停止该方向的扩展
    """
    space_size = space_state.shape

    # 检查起始点是否有效
    if (x >= space_size[0] or y >= space_size[1] or z >= space_size[2] or
            x < 0 or y < 0 or z < 0):
        return 0, 0, 0

    # 检查起始点是否已被占用
    if space_state[x][y][z] == 1:
        return 0, 0, 0

    max_volume = 0
    best_dims = (0, 0, 0)

    # 计算从起始点能扩展的最大范围
    max_possible_length = space_size[0] - x
    max_possible_width = space_size[1] - y
    max_possible_height = space_size[2] - z

    # 逐步扩展立方体，一旦某个方向遇到占用空间就停止
    for length in range(1, max_possible_length + 1):
        # 检查X方向扩展一层是否有占用空间
        x_blocked = False
        for j in range(y, y + 1):  # 先检查最小范围
            for k in range(z, z + 1):
                if space_state[x + length - 1][j][k] == 1:
                    x_blocked = True
                    break
            if x_blocked:
                break

        if x_blocked:
            break

        for width in range(1, max_possible_width + 1):
            # 检查Y方向扩展是否有占用空间
            y_blocked = False
            for i in range(x, x + length):
                for k in range(z, z + 1):  # 先检查最小范围
                    if space_state[i][y + width - 1][k] == 1:
                        y_blocked = True
                        break
                if y_blocked:
                    break

            if y_blocked:
                break

            for height in range(1, max_possible_height + 1):
                # 检查Z方向扩展是否有占用空间
                z_blocked = False
                for i in range(x, x + length):
                    for j in range(y, y + width):
                        if space_state[i][j][z + height - 1] == 1:
                            z_blocked = True
                            break
                    if z_blocked:
                        break

                if z_blocked:
                    break

                # 如果没有被阻挡，计算体积
                volume = length * width * height
                if volume > max_volume:
                    max_volume = volume
                    best_dims = (length, width, height)

    return best_dims
