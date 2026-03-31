from my_imports import plt, Poly3DCollection, matplotlib
from my_imports import os


def visualize_packing(data_scale, example_id, space_size, item, volume_utilization1, volume_utilization_mean, train_or_test, folder_name, pick_items=None):
    # 画图
    plt.ion()
    # matplotlib.use('Agg')  # 实现只保存图片，而不显示图片
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    # 箱子边框
    space_x = space_size[0]
    space_y = space_size[1]
    space_z = space_size[2]
    # 箱子的12条边
    edges = [
        # 底面边
        [(0., 0., 0.), (space_x, 0., 0.)],
        [(space_x, 0., 0.), (space_x, space_y, 0.)],
        [(space_x, space_y, 0.), (0., space_y, 0.)],
        [(0., space_y, 0.), (0., 0., 0.)],
        # 顶面边
        [(0., 0., space_z), (space_x, 0., space_z)],
        [(space_x, 0., space_z), (space_x, space_y, space_z)],
        [(space_x, space_y, space_z), (0., space_y, space_z)],
        [(0., space_y, space_z), (0., 0., space_z)],
        # 垂直边
        [(0., 0., 0.), (0., 0., space_z)],
        [(space_x, 0., 0.), (space_x, 0., space_z)],
        [(space_x, space_y, 0.), (space_x, space_y, space_z)],
        [(0., space_y, 0.), (0., space_y, space_z)]
    ]
    for edge in edges:
        xs, ys, zs = zip(*edge)
        ax.plot3D(xs, ys, zs, 'b-', alpha=0.6, linewidth=2)

    # 已放置的货物（红色）
    # 如指定pick_items，则绘制该数据，否则默认传入item
    draw_items = pick_items if pick_items is not None else item
    space_occupied_volume = 0
    for i in draw_items:
        if i.placed:
            space_occupied_volume += i.volume
            x, y, z = i.x, i.y, i.z
            dx, dy, dz = i.length, i.width, i.height
            # 创建货物的8个顶点
            vertices = [
                [x, y, z], [x + dx, y, z], [x + dx, y + dy, z], [x, y + dy, z],  # 底面
                [x, y, z + dz], [x + dx, y, z + dz], [x + dx, y + dy, z + dz], [x, y + dy, z + dz]  # 顶面
            ]
            # 定义立方体的12条边
            item_edges = [
                [0, 1], [1, 2], [2, 3], [3, 0],  # 底面
                [4, 5], [5, 6], [6, 7], [7, 4],  # 顶面
                [0, 4], [1, 5], [2, 6], [3, 7]  # 垂直边
            ]
            # 画出12条边
            for edge in item_edges:
                points = [vertices[edge[0]], vertices[edge[1]]]
                xs, ys, zs = zip(*points)
                ax.plot3D(xs, ys, zs, 'r-', linewidth=2)
            # 填充货物面（部分透明）
            faces = [
                [vertices[0], vertices[1], vertices[2], vertices[3]],  # 底面
                [vertices[4], vertices[5], vertices[6], vertices[7]],  # 顶面
                [vertices[0], vertices[1], vertices[5], vertices[4]],  # 前面
                [vertices[2], vertices[3], vertices[7], vertices[6]],  # 后面
                [vertices[1], vertices[2], vertices[6], vertices[5]],  # 右面
                [vertices[0], vertices[3], vertices[7], vertices[4]]  # 左面
            ]
            ax.add_collection3d(Poly3DCollection(faces, alpha=0.3, facecolor='red', edgecolor='red'))

    ax.set_xlabel('长 (X)')
    ax.set_ylabel('宽 (Y)')
    ax.set_zlabel('高 (Z)')
    ax.set_title('三维装箱，利用率为%.2f%%' % volume_utilization1)

    # 设置坐标轴范围
    ax.set_xlim(0, space_x)
    ax.set_ylim(0, space_y)
    ax.set_zlim(0, space_z)
    # plt.tight_layout()
    picture_name = f'{train_or_test}规模{data_scale}算例{example_id}，v：{volume_utilization1:.2f}%，vm：{volume_utilization_mean:.2f}%.png'
    path_name = os.path.join(folder_name, picture_name)  # 拼接 路径
    plt.savefig(path_name)
    plt.show()
    plt.close()
