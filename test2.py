from my_imports import os
from parameter import parameter
from test1 import test1

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

if __name__ == "__main__":
    print(f'\n开始运行')
    data_method = 5
    supervise_ratio = 0  # 监督学习比例
    space_size, item_lwh_min, item_lwh_max, volume_utilization_mean_max, space_free_num_max, item_place_times_max, item_num_max = parameter(
        data_method)
    current_script_dir = os.path.dirname(os.path.abspath(__file__))  # 当前脚本路径
    folder_name = '13' + '训练' + str(2000) + '货物' + str(item_num_max) + '空间' + str(
        space_free_num_max) + '监督' + str(supervise_ratio) + 'P6' + '训练n' + '测试n'
    folder_name = os.path.join(current_script_dir, folder_name)  # 拼接 路径
    result_file_name = os.path.join(folder_name, 'result.xlsx')
    result_detail_file_name = os.path.join(folder_name, 'result_detail.xlsx')
    policy_name = 'final_dqn_policy.pth'
    policy_name = os.path.join(folder_name, policy_name)  # 拼接 模型路径

    test1(2100, 'data\数据_101010_cut2.xlsx', result_file_name, result_detail_file_name, policy_name, folder_name)
    # test1(2100, 'data\数据_101010_cut1.xlsx', result_file_name, result_detail_file_name, policy_name, folder_name)
    # test1(2100, 'data\数据_101010_rs.xlsx', result_file_name, result_detail_file_name, policy_name, folder_name)
