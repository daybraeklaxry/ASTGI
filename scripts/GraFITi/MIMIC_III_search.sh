#!/bin/bash

# =====================================================================================
#                 GraFITi 在 MIMIC-III 上的超参数搜索脚本 (高效GPU利用)
# =====================================================================================

# --- 1. GPU 配置 ---
# 在这里指定你想使用的GPU编号，用逗号隔开，例如 "0,1,2,3" 或 "4,5,6,7"
GPU_IDS="0,1,2,3,4,5,6,7"

# 将GPU_IDS字符串转换为bash数组
IFS=',' read -r -a GPUS <<< "$GPU_IDS"
NUM_GPUS=${#GPUS[@]}

# [核心机制] 使用关联数组来追踪每个GPU上运行的进程ID (PID)
# 键: GPU ID, 值: 任务的 PID
declare -A gpu_pids
for gpu in "${GPUS[@]}"; do
    gpu_pids[$gpu]=""
done

echo "🚀 将在 ${NUM_GPUS} 个GPU (${GPUS[*]}) 上并行执行 GraFITi 任务，采用动态调度策略。"

# --- 2. 固定参数 (基于MIMIC-III数据集和公平比较原则) ---
model_name="GraFITi"
dataset_root_path="storage/datasets/MIMIC_III"
dataset_name="MIMIC_III"
seq_len=72
pred_len=3
enc_in=96
# 假设 GraFITi 的主入口文件是 main.py，并且使用上述参数

# --- 3. 超参数搜索空间 ---
# [策略] d_model: 参考 GraFITi 论文和代码库，选择常见的嵌入维度
d_models=(64 128 256)
# [策略] batch_size: 考虑到不同模型大小的显存消耗，提供多种选项
batch_sizes=(16 32 64)
# [策略] lrs: 包含 GraFITi 论文中使用的 0.001，并向两侧扩展
lrs=(0.001)
# [策略] n_layers: GraFITi 论文中建议的搜索范围
n_layers_options=(1 2 3 4)
# [策略] n_heads: GraFITi 论文中建议的注意力头数
n_heads_options=(1 2 4)


# --- 4. 随机搜索设置 ---
# 定义您希望运行的总实验次数
TOTAL_RUNS=128

# --- [核心机制] 动态寻找空闲GPU的函数 ---
find_free_gpu() {
    local free_gpu_id=""
    # 循环直到找到一个空闲的GPU
    while [[ -z "$free_gpu_id" ]]; do
        # 遍历所有指定的GPU
        for gpu_id in "${GPUS[@]}"; do
            pid=${gpu_pids[$gpu_id]}
            # 如果PID为空或进程不存在，则认为该GPU空闲
            if [[ -z "$pid" ]] || ! kill -0 "$pid" 2>/dev/null; then
                free_gpu_id=$gpu_id
                unset gpu_pids[$gpu_id] # 清除旧的PID记录
                break
            fi
        done
        # 如果遍历完所有GPU都没有找到空闲的，则等待任何一个后台任务结束
        if [[ -z "$free_gpu_id" ]]; then
            wait -n
        fi
    done
    echo "$free_gpu_id"
}


echo "🚀 开始 GraFITi 在 MIMIC-III 上的随机搜索，总共将运行 ${TOTAL_RUNS} 组实验..."

# --- 5. 随机搜索主循环 ---
for (( i=1; i<=${TOTAL_RUNS}; i++ )); do

  # --- 随机采样超参数 ---
  dm=${d_models[$((RANDOM % ${#d_models[@]}))]}
  bs=${batch_sizes[$((RANDOM % ${#batch_sizes[@]}))]}
  lr=${lrs[$((RANDOM % ${#lrs[@]}))]}
  layers=${n_layers_options[$((RANDOM % ${#n_layers_options[@]}))]}
  heads=${n_heads_options[$((RANDOM % ${#n_heads_options[@]}))]}


  # --- [核心机制] GPU 动态分配 ---
  # 调用函数获取一个当前空闲的GPU ID
  gpu_id=$(find_free_gpu)

  # --- 构建唯一的模型ID (包含关键超参数) ---
  model_id="${model_name}_${dataset_name}_sl${seq_len}_pl${pred_len}_dm${dm}_L${layers}_H${heads}_lr${lr}_bs${bs}_run${i}"

  echo "-----------------------------------------------------------------------"
  echo "📈 启动任务 [${i}/${TOTAL_RUNS}] -> 动态分配至空闲 GPU ${gpu_id}"
  echo "   架构: d_model=${dm}, n_layers=${layers}, n_heads=${heads}"
  echo "   训练: batch_size=${bs}, learning_rate=${lr}"
  echo "   模型 ID: ${model_id}"
  echo "-----------------------------------------------------------------------"

  # --- 在后台启动训练任务 ---
  # 注意：这里的 `main.py` 及其参数需要与您的 GraFITi 项目实际情况匹配
  (
    python main.py \
      --gpu_id $gpu_id \
      --is_training 1 \
      --model_name "$model_name" \
      --model_id "$model_id" \
      --dataset_root_path "$dataset_root_path" \
      --dataset_name "$dataset_name" \
      --features M \
      --seq_len "$seq_len" \
      --pred_len "$pred_len" \
      --enc_in "$enc_in" \
      --loss "MSE" \
      --train_epochs 300 \
      --patience 5 \
      --val_interval 1 \
      --itr 5 \
      --d_model "$dm" \
      --n_layers "$layers" \
      --n_heads "$heads" \
      --batch_size "$bs" \
      --learning_rate "$lr" \
      --use_multi_gpu 0 # 因为我们是手动分配单GPU，所以这里设为0
  ) &

  # --- [核心机制] 记录新任务的 PID ---
  new_pid=$!
  gpu_pids[$gpu_id]=$new_pid
  echo "   任务已启动，PID: ${new_pid}，已绑定至 GPU ${gpu_id}"

  # 短暂休眠，以确保日志顺序和PID被正确捕获
  sleep 1

done

echo "✅ 所有 ${TOTAL_RUNS} 组任务已启动，等待最后批次的任务执行完毕..."
# 等待所有后台子进程完成
wait
echo "🎉🎉🎉 GraFITi 在 MIMIC-III 上的随机搜索任务已全部完成！🎉🎉🎉"