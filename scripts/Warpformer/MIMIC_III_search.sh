#!/bin/bash

# =====================================================================================
#                 Warpformer 在 MIMIC-III 上的超参数搜索脚本 (高效GPU利用)
# =====================================================================================

# --- 1. GPU 配置 ---
# 在这里指定你想使用的GPU编号，用逗号隔开，例如 "0,1,2,3" 或 "4,5,6,7"
GPU_IDS="0,1,2,5,6,7"

# 将GPU_IDS字符串转换为bash数组
IFS=',' read -r -a GPUS <<< "$GPU_IDS"
NUM_GPUS=${#GPUS[@]}

# [核心机制] 使用关联数组来追踪每个GPU上运行的进程ID (PID)
# 键: GPU ID, 值: 任务的 PID
declare -A gpu_pids
for gpu in "${GPUS[@]}"; do
    gpu_pids[$gpu]=""
done

echo "🚀 将在 ${NUM_GPUS} 个GPU (${GPUS[*]}) 上并行执行 Warpformer 任务，采用动态调度策略。"

# --- 2. 固定参数 (基于MIMIC-III数据集和公平比较原则) ---
model_name="Warpformer"
dataset_root_path="storage/datasets/MIMIC_III"
dataset_name="MIMIC_III"
seq_len=72
pred_len=3
enc_in=96
c_out=96

# --- 3. Warpformer 超参数搜索空间 (基于策略分析) ---

# [策略] d_model: 模型的核心维度，是性能和复杂度的关键。探索从较小到较大的范围。
d_models=(64 128 256)

# [策略] batch_size: 考虑到d_model增大可能增加显存消耗，探索 16 和 32 两个常用选项。
batch_sizes=(16 32)

# [策略] learning_rate: 学习率是优化的关键。选择一个经过验证的、在 Transformer 类模型上表现良好的范围。
lrs=(0.001)

# [策略] dropout: Transformer 中标准的正则化方法，防止过拟合。
dropouts=(0.0 0.1 0.2)

# [策略] n_layers: Encoder 层数，决定了模型的深度和捕捉复杂依赖的能力。
n_layers_options=(1 2 3)

# [策略] n_heads: 多头注意力机制的头数，允许模型在不同子空间中关注信息。
n_heads_options=(2 4 8)


# --- 4. 随机搜索设置 ---
# 总共希望运行的实验组数
TOTAL_RUNS=256

# --- [核心机制] 动态寻找空闲GPU的函数 ---
# (此函数与您提供的脚本完全相同，功能是找到一个当前没有任务运行的GPU)
find_free_gpu() {
    local free_gpu_id=""
    # 循环直到找到一个空闲的GPU
    while [[ -z "$free_gpu_id" ]]; do
        for gpu_id in "${GPUS[@]}"; do
            pid=${gpu_pids[$gpu_id]}
            # 如果PID为空或该进程不存在，则认为GPU空闲
            if [[ -z "$pid" ]] || ! kill -0 "$pid" 2>/dev/null; then
                free_gpu_id=$gpu_id
                unset gpu_pids[$gpu_id] # 清除旧的、已结束的PID记录
                break
            fi
        done
        # 如果所有GPU都在忙，等待任何一个后台任务结束
        if [[ -z "$free_gpu_id" ]]; then
            wait -n
        fi
    done
    echo "$free_gpu_id"
}


echo "🚀 开始 Warpformer 在 MIMIC-III 上的随机搜索，总共将运行 ${TOTAL_RUNS} 组实验..."

# --- 5. 随机搜索主循环 ---
for (( i=1; i<=${TOTAL_RUNS}; i++ )); do

  # --- 随机采样超参数 ---
  dm=${d_models[$((RANDOM % ${#d_models[@]}))]}
  bs=${batch_sizes[$((RANDOM % ${#batch_sizes[@]}))]}
  lr=${lrs[$((RANDOM % ${#lrs[@]}))]}
  dp=${dropouts[$((RANDOM % ${#dropouts[@]}))]}
  n_layers=${n_layers_options[$((RANDOM % ${#n_layers_options[@]}))]}
  n_heads=${n_heads_options[$((RANDOM % ${#n_heads_options[@]}))]}


  # --- [核心机制] GPU 动态分配 ---
  gpu_id=$(find_free_gpu)

  # --- 构建唯一的模型ID (包含Warpformer关键参数) ---
  model_id="${model_name}_${dataset_name}_sl${seq_len}_pl${pred_len}_dm${dm}_L${n_layers}_nh${n_heads}_dp${dp}_lr${lr}_bs${bs}_run${i}"

  echo "-----------------------------------------------------------------------"
  echo "📈 启动任务 [${i}/${TOTAL_RUNS}] -> 动态分配至空闲 GPU ${gpu_id}"
  echo "   Arch: d_model=${dm}, n_layers=${n_layers}, n_heads=${n_heads}, dropout=${dp}"
  echo "   Train: batch_size=${bs}, learning_rate=${lr}"
  echo "   Model ID: ${model_id}"
  echo "-----------------------------------------------------------------------"

  # --- 在后台启动训练任务 (注意参数已更新为Warpformer的) ---
  (
    # 将任务绑定到找到的空闲GPU上
    CUDA_VISIBLE_DEVICES=${gpu_id} python main.py \
      --is_training 1 \
      --model_id "$model_id" \
      --model_name "$model_name" \
      --dataset_root_path "$dataset_root_path" \
      --dataset_name "$dataset_name" \
      --features M \
      --seq_len "$seq_len" \
      --pred_len "$pred_len" \
      --enc_in "$enc_in" \
      --c_out "$c_out" \
      --loss "MSE" \
      --train_epochs 300 \
      --patience 5 \
      --val_interval 1 \
      --itr 5 \
      --batch_size "$bs" \
      --learning_rate "$lr" \
      --d_model "$dm" \
      --n_heads "$n_heads" \
      --n_layers "$n_layers" \
      --dropout "$dp"
  ) & # 使用 '&' 将命令置于后台执行

  # --- [核心机制] 记录新任务的 PID ---
  new_pid=$!
  gpu_pids[$gpu_id]=$new_pid
  echo "   任务已启动，PID: ${new_pid}，已绑定至 GPU ${gpu_id}"

  sleep 1 # 短暂休眠，确保日志顺序和PID正确捕获

done

echo "✅ 所有 ${TOTAL_RUNS} 组任务已启动，等待最后批次的任务执行完毕..."
wait # 等待所有后台启动的子进程全部结束
echo "🎉🎉🎉 Warpformer 在 MIMIC-III 上的随机搜索任务已全部完成！🎉🎉🎉"