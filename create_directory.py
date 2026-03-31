import os
import shutil


def create_directory(directory_path):
    """
    安全地创建或清空文件夹，保留文件夹权限和元数据

    参数:
        directory_path: 要创建或清空的文件夹路径
    """
    try:
        # 检查文件夹是否存在
        if os.path.exists(directory_path):
            # print(f"文件夹 '{directory_path}' 已存在，正在清空内容...")

            # 删除文件夹内的所有内容，但保留文件夹本身
            for filename in os.listdir(directory_path):
                file_path = os.path.join(directory_path, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)  # 删除文件或链接
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)  # 删除子文件夹
                except Exception as e:
                    print(f"删除 {file_path} 失败: {e}")

            # print(f"已清空文件夹: {directory_path}")
        else:
            # 如果文件夹不存在，直接创建
            os.makedirs(directory_path)
            # print(f"已创建新文件夹: {directory_path}")

        return True

    except Exception as e:
        print(f"操作失败: {e}")
        return False
