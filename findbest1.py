import os
import json
import argparse
from pathlib import Path
from collections import defaultdict
import statistics


# --- 新增的辅助函数 ---

def process_and_display_group_results(group_name, results_by_config, n):
    """
    处理并展示单个分组的Top-N结果。
    这是一个辅助函数，用于避免主函数中的代码重复。
    """
    if not results_by_config:
        print(f"--- 分组 '{group_name}': 未找到任何有效的实验配置。---\n")
        return

    print("---" * 18)
    print(f"📊 开始分析分组: {group_name}")
    print("---" * 18)

    mse_configs = []
    mae_configs = []

    for config_name, metrics in results_by_config.items():
        mse_list = metrics['mse_list']
        mae_list = metrics['mae_list']
        iter_count = len(mse_list)

        if iter_count == 0:
            continue

        avg_mse = statistics.mean(mse_list)
        avg_mae = statistics.mean(mae_list)

        mse_stdev = statistics.stdev(mse_list) if iter_count > 1 else 0.0
        mae_stdev = statistics.stdev(mae_list) if iter_count > 1 else 0.0

        # 为MSE排序准备数据
        mse_configs.append({
            'name': config_name,
            'avg_mse': avg_mse, 'mse_stdev': mse_stdev,
            'avg_mae': avg_mae, 'mae_stdev': mae_stdev,
            'iter_count': iter_count
        })

        # 为MAE排序准备数据
        mae_configs.append({
            'name': config_name,
            'avg_mae': avg_mae, 'mae_stdev': mae_stdev,
            'avg_mse': avg_mse, 'mse_stdev': mse_stdev,
            'iter_count': iter_count
        })

    # 按MSE和MAE排序（升序，越小越好）
    mse_configs.sort(key=lambda x: x['avg_mse'])
    mae_configs.sort(key=lambda x: x['avg_mae'])

    print(f"✅ 分组分析完成！共处理了 {len(results_by_config)} 个独立的实验配置。")

    # 显示前n个MSE结果
    print(f"\n🏆 [分组: {group_name}] 平均MSE前{n}名结果:")
    if not mse_configs:
        print("  未能找到任何有效的MSE结果。")
    else:
        for i, config in enumerate(mse_configs[:n], 1):
            print(f"  {i}. 配置名称: {config['name']}")
            print(f"     平均MSE: {config['avg_mse']:.6f} (标准差: {config['mse_stdev']:.6f})")
            print(f"     平均MAE: {config['avg_mae']:.6f} (标准差: {config['mae_stdev']:.6f})")
            print(f"     迭代次数: {config['iter_count']}")
            print()

    # 显示前n个MAE结果
    print(f"\n🏆 [分组: {group_name}] 平均MAE前{n}名结果:")
    if not mae_configs:
        print("  未能找到任何有效的MAE结果。")
    else:
        for i, config in enumerate(mae_configs[:n], 1):
            print(f"  {i}. 配置名称: {config['name']}")
            print(f"     平均MAE: {config['avg_mae']:.6f} (标准差: {config['mae_stdev']:.6f})")
            print(f"     平均MSE: {config['avg_mse']:.6f} (标准差: {config['mse_stdev']:.6f})")
            print(f"     迭代次数: {config['iter_count']}")
            print()


def find_top_n_average_runs(search_dir, n=10):
    """
    递归搜索 'metric.json'，按实验配置前缀进行分组，计算每个配置下多次迭代
    的平均值和标准差，并为每个分组找出最优的前n个配置。

    支持按 'TAC-Mixer_P12_sl24_pl24_' 和 'TAC-Mixer_P12_sl36_pl3_' 前缀分组。
    """
    root_path = Path(search_dir)

    if not root_path.is_dir():
        print(f"❌ 错误: 目录 '{search_dir}' 不存在")
        return

    print(f"🚀 开始在目录中搜索: {root_path}")
    print(f"🎯 目标: 查找每个配置下所有迭代的平均MSE/MAE前{n}名及其标准差\n")

    # --- 步骤 1: 数据收集与分组 ---
    # 定义分组前缀
    PREFIXES = {
        'sl72_pl3': 'ST-PPGN_MIMIC_III_sl72_pl3_',
        'sl3000_pl300': 'ST-PPGN_HumanActivity_sl3000_pl300_',
        'sl2160_pl3': 'ST-PPGN_MIMIC_IV_sl2160_pl3_',
        'sl36_pl3': 'ST-PPGN_P12_sl36_pl3_',
        'sl150_pl3': 'ST-PPGN_USHCN_sl150_pl3',
        'sl72_pl3_s': 'ST-PPGN_MIMIC_III_Sens_',
        'sl3000_pl300_s': 'ST-PPGN_HumanActivity_Sens_',
        'sl2160_pl3_s': 'ST-PPGN_MIMIC_IV_Sens_',
        'sl36_pl3_s': 'ST-PPGN_P12_Sens_',
        'sl150_pl3_s': 'ST-PPGN_USHCN_Sens_'
    }

    # 创建用于存储分组结果的字典
    grouped_results = {
        group_key: defaultdict(lambda: {'mse_list': [], 'mae_list': []})
        for group_key in PREFIXES
    }
    # grouped_results['其他'] = defaultdict(lambda: {'mse_list': [], 'mae_list': []})

    metric_files = list(root_path.rglob('metric.json'))
    if not metric_files:
        print("未找到任何 'metric.json' 文件。请检查目录结构和文件名。")
        return

    # 智能地确定配置目录的前缀，例如 'TAC-Mixer_'
    model_name_prefix = root_path.name + "_"

    for metric_file in metric_files:
        try:
            # 从 metric.json 向上查找父目录，直到找到以模型名开头的目录
            config_name = None
            current_path = metric_file.parent
            while current_path != root_path.parent:
                if current_path.name.startswith(model_name_prefix):
                    config_name = current_path.name
                    break
                current_path = current_path.parent

            if config_name is None:
                print(f"⚠️ 警告: 未能在路径 {metric_file} 中找到预期的配置目录，已跳过。")
                continue

            with open(metric_file, 'r') as f:
                data = json.load(f)

            # 根据配置名称前缀进行分组
            matched_group = None
            for group_key, prefix in PREFIXES.items():
                if config_name.startswith(prefix):
                    grouped_results[group_key][config_name]['mse_list'].append(data['MSE'])
                    grouped_results[group_key][config_name]['mae_list'].append(data['MAE'])
                    matched_group = group_key
                    break

            # 如果没有匹配任何已知前缀，则放入“其他”组
            if not matched_group:
                grouped_results['其他'][config_name]['mse_list'].append(data['MSE'])
                grouped_results['其他'][config_name]['mae_list'].append(data['MAE'])

        except (json.JSONDecodeError, KeyError, IndexError) as e:
            # print(f"⚠️ 警告: 跳过文件 {metric_file} (原因: {e})")
            continue

    # --- 步骤 2 & 3: 对每个分组进行计算和展示 ---
    for group_name, results_data in grouped_results.items():
        process_and_display_group_results(group_name, results_data, n)

    print("---" * 18)
    print("🎉 所有分组分析完毕。")
    print("---" * 18)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="查找实验结果中平均MSE/MAE最小的前n个配置，并按配置前缀分组报告。"
    )
    parser.add_argument(
        "search_dir",
        type=str,
        help="要搜索的结果根目录 (例如, storage/results/HumanActivity/TAC-Mixer)。"
    )
    parser.add_argument(
        "-n", "--top_n",
        type=int,
        default=10,
        help="要为每个分组显示的前n个结果数量 (默认为10)。"
    )
    args = parser.parse_args()

    find_top_n_average_runs(args.search_dir, args.top_n)