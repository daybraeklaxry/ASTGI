import os
import json
import argparse
from pathlib import Path
from collections import defaultdict
import statistics


def find_best_average_run(search_dir):
    """
    递归搜索 'metric.json', 按实验配置分组，计算每个配置下多次迭代
    的平均值和标准差，并找出最优的配置。

    输出最优MSE配置及其对应的MAE，以及最优MAE配置及其对应的MSE，
    并附上各自的标准差以衡量结果的稳定性。
    """
    root_path = Path(search_dir)

    if not root_path.is_dir():
        print(f"错误: 目录 '{search_dir}' 不存在")
        return

    print(f"🚀 开始在目录中搜索: {root_path}")
    print("🎯 目标: 查找每个配置下所有迭代的平均MSE/MAE最小值及其标准差\n")

    # 使用 defaultdict 极大简化代码
    # 数据结构: {'config_name': {'mse_list': [...], 'mae_list': [...]}}
    results_by_config = defaultdict(lambda: {'mse_list': [], 'mae_list': []})

    # --- 步骤 1: 数据收集与分组 ---
    # 使用 rglob 递归查找所有 metric.json 文件
    metric_files = list(root_path.rglob('metric.json'))
    if not metric_files:
        print("未找到任何 'metric.json' 文件。请检查目录结构和文件名。")
        return

    for metric_file in metric_files:
        try:
            # 根据您的目录结构，实验配置目录是 metric.json 文件的上3级
            # .../[config_name]/iterX/eval_.../metric.json
            config_name = metric_file.parents[2].name

            with open(metric_file, 'r') as f:
                data = json.load(f)

            # 将MSE和MAE值添加到对应配置的列表中
            results_by_config[config_name]['mse_list'].append(data['MSE'])
            results_by_config[config_name]['mae_list'].append(data['MAE'])

        except (json.JSONDecodeError, KeyError, IndexError):
            # 如果文件有问题或目录结构不符，则跳过
            print(f"警告: 跳过文件 {metric_file} (格式错误或目录结构不符)")
            continue

    # --- 步骤 2 & 3: 计算统计量并寻找最优 ---
    if not results_by_config:
        print("处理后未发现任何有效的实验配置。")
        return

    best_avg_mse = float('inf')
    best_mse_config_details = {}

    best_avg_mae = float('inf')
    best_mae_config_details = {}

    for config_name, metrics in results_by_config.items():
        mse_list = metrics['mse_list']
        mae_list = metrics['mae_list']
        iter_count = len(mse_list)

        # 确保列表不为空，避免计算错误
        if iter_count == 0:
            continue

        # 计算平均值
        avg_mse = statistics.mean(mse_list)
        avg_mae = statistics.mean(mae_list)

        # 计算标准差，衡量多次迭代结果的稳定性
        # 如果只有一个数据点，标准差为0
        if iter_count > 1:
            mse_stdev = statistics.stdev(mse_list)
            mae_stdev = statistics.stdev(mae_list)
        else:
            mse_stdev = 0.0
            mae_stdev = 0.0

        # 比较并更新最佳MSE
        if avg_mse < best_avg_mse:
            best_avg_mse = avg_mse
            best_mse_config_details = {
                'name': config_name,
                'avg_mse': avg_mse,
                'mse_stdev': mse_stdev,  # <-- 新增: MSE标准差
                'avg_mae': avg_mae,
                'mae_stdev': mae_stdev,  # <-- 新增: MAE标准差
                'iter_count': iter_count
            }

        # 比较并更新最佳MAE
        if avg_mae < best_avg_mae:
            best_avg_mae = avg_mae
            best_mae_config_details = {
                'name': config_name,
                'avg_mae': avg_mae,
                'mae_stdev': mae_stdev,  # <-- 新增: MAE标准差
                'avg_mse': avg_mse,
                'mse_stdev': mse_stdev,  # <-- 新增: MSE标准差
                'iter_count': iter_count
            }

    # --- 步骤 4: 结果展示 ---
    print("---" * 15)
    print(f"✅ 搜索完成！共分析了 {len(results_by_config)} 个独立的实验配置。")

    if not best_mse_config_details:
        print("\n未能找到任何有效的MSE结果。")
    else:
        print("\n🏆 平均MSE最优结果:")
        print(f"   - 配置名称: {best_mse_config_details['name']}")
        # <-- 更新: 同时显示平均值和标准差 -->
        print(f"   - 最小平均MSE: {best_mse_config_details['avg_mse']:.6f} (标准差: {best_mse_config_details['mse_stdev']:.6f})")
        print(f"   - 其平均MAE为: {best_mse_config_details['avg_mae']:.6f} (标准差: {best_mse_config_details['mae_stdev']:.6f})")
        print(f"   - 计算基于: {best_mse_config_details['iter_count']} 次迭代")

    if not best_mae_config_details:
        print("\n未能找到任何有效的MAE结果。")
    else:
        print("\n🏆 平均MAE最优结果:")
        print(f"   - 配置名称: {best_mae_config_details['name']}")
        # <-- 更新: 同时显示平均值和标准差 -->
        print(f"   - 最小平均MAE: {best_mae_config_details['avg_mae']:.6f} (标准差: {best_mae_config_details['mae_stdev']:.6f})")
        print(f"   - 其平均MSE为: {best_mae_config_details['avg_mse']:.6f} (标准差: {best_mae_config_details['mse_stdev']:.6f})")
        print(f"   - 计算基于: {best_mae_config_details['iter_count']} 次迭代")
    print("---" * 15)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="查找实验结果中平均MSE/MAE最小的配置，并报告其标准差。"
    )
    parser.add_argument(
        "search_dir",
        type=str,
        help="要搜索的结果根目录 (例如, /storage/results/HumanActivity/APN)。"
    )
    args = parser.parse_args()

    find_best_average_run(args.search_dir)