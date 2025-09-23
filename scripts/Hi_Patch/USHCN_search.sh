#!/bin/bash

# =====================================================================================
#                      Hi-Patch 在 USHCN 上的超参数搜索脚本
# =====================================================================================

# --- 1. GPU 配置 ---
# 在这里指定你想使用的GPU编号，用逗号隔开，例如 "0,1,2,3"
GPU_IDS="0,1,2,3,4,5,6,7"

# 将GPU_IDS字符串转换为bash数组
IFS=',' read -r -a GPUS <<< "$GPU_IDS"
NUM_GPUS=${#GPUS[@]}

# [核心机制] 使用关联数组来追踪每个GPU上运行的进程ID (PID)
declare -A gpu_pids
for gpu in "${GPUS[@]}"; do
    gpu_pids[$gpu]=""
done

echo "🚀 将在 ${NUM_GPUS} 个GPU (${GPUS[*]}) 上并行执行 Hi-Patch 模型的超参数搜索任务。"

# --- 2. 固定参数 (为 USHCN 数据集和模型特性进行设置) ---
model_name="Hi_Patch"
dataset_root_path="storage/datasets/USHCN"
dataset_name="USHCN"
# [公平比较] 遵循通用设置
seq_len=150
pred_len=3
enc_in=5
dec_in=5
c_out=5
# Hi-Patch 使用 'collate_fn_patch'
collate_fn="collate_fn_patch"


# --- 3. 超参数搜索范围 ---
# [调优策略] d_model: 模型维度，探索不同模型容量的影响
d_models=(32 64 128)
# [调优策略] batch_sizes: 批次大小，影响训练稳定性和速度
batch_sizes=(16 32 64)
# [调优策略] lrs: 学习率，是训练中最关键的超参数之一
lrs=(0.0005 0.001 0.002)
# [调优策略] n_layers_options: GAT层数，决定了信息在图上传播的深度
n_layers_options=(1 2 3)
# [调优策略] n_heads_options: 多头注意力头数，让模型关注不同方面的信息
n_heads_options=(1 2 4)
# [调优策略] patch_len_options: 补丁长度。对于长度为150的序列，选择一些合适的因子作为补丁长度。
# 补丁划分是Hi-Patch的核心，直接影响模型如何看待局部信息。
# 注意：patch_stride 将被设置为与 patch_len 相等，以实现论文中描述的非重叠补丁。
patch_len_options=(6 10 15 25)


# --- 4. 随机搜索设置 ---
# [调优策略] 在指定范围内随机组合参数，执行的总实验次数。
TOTAL_RUNS=512

# --- [核心机制] 动态寻找空闲GPU的函数 ---
find_free_gpu() {
    local free_gpu_id=""
    while [[ -z "$free_gpu_id" ]]; do
        for gpu_id in "${GPUS[@]}"; do
            pid=${gpu_pids[$gpu_id]}
            # 检查PID是否为空，或者该进程是否已不存在
            if [[ -z "$pid" ]] || ! kill -0 "$pid" 2>/dev/null; then
                free_gpu_id=$gpu_id
                # 标记该GPU即将被占用
                gpu_pids[$gpu_id]="taken"
                break
            fi
        done
        # 如果没有找到空闲GPU，则等待任何一个正在运行的后台任务结束
        if [[ -z "$free_gpu_id" ]]; then
            wait -n
        fi
    done
    echo "$free_gpu_id"
}


echo "🚀 开始 Hi-Patch 在 USHCN 上的超参数搜索，总共将运行 ${TOTAL_RUNS} 组实验..."

# --- 5. 随机搜索主循环 ---
for (( i=1; i<=${TOTAL_RUNS}; i++ )); do

  # --- 随机采样超参数 ---
  dm=${d_models[$((RANDOM % ${#d_models[@]}))]}
  bs=${batch_sizes[$((RANDOM % ${#batch_sizes[@]}))]}
  lr=${lrs[$((RANDOM % ${#lrs[@]}))]}
  n_layers=${n_layers_options[$((RANDOM % ${#n_layers_options[@]}))]}
  n_heads=${n_heads_options[$((RANDOM % ${#n_heads_options[@]}))]}
  p_len=${patch_len_options[$((RANDOM % ${#patch_len_options[@]}))]}
  # 保持补丁不重叠
  p_stride=$p_len

  # --- [核心机制] GPU 动态分配 ---
  gpu_id=$(find_free_gpu)

  # --- 构建唯一的模型ID，用于结果追溯 ---
  model_id="${model_name}_${dataset_name}_sl${seq_len}_pl${pred_len}_dm${dm}_L${n_layers}_h${n_heads}_plen${p_len}_lr${lr}_bs${bs}_run${i}"

  echo "-----------------------------------------------------------------------"
  echo "📈 启动任务 [${i}/${TOTAL_RUNS}] -> 动态分配至空闲 GPU ${gpu_id}"
  echo "   Arch: d_model=${dm}, n_layers=${n_layers}, n_heads=${n_heads}, patch_len=${p_len}"
  echo "   Train: bs=${bs}, lr=${lr}"
  echo "   Model ID: ${model_id}"
  echo "-----------------------------------------------------------------------"

  # --- 在后台启动训练任务 ---
  (
    python main.py \
      --gpu_id "$gpu_id" \
      --is_training 1 \
      --model_id "$model_id" \
      --model_name "$model_name" \
      --dataset_root_path "$dataset_root_path" \
      --dataset_name "$dataset_name" \
      --collate_fn "$collate_fn" \
      --features M \
      --seq_len "$seq_len" \
      --pred_len "$pred_len" \
      --enc_in "$enc_in" \
      --dec_in "$dec_in" \
      --c_out "$c_out" \
      --loss "MSE" \
      --train_epochs 300 \
      --patience 5 \
      --val_interval 1 \
      --itr 5 \
      --batch_size "$bs" \
      --learning_rate "$lr" \
      --d_model "$dm" \
      --n_layers "$n_layers" \
      --n_heads "$n_heads" \
      --patch_len "$p_len" \
      --patch_stride "$p_stride"
  ) &

  # --- [核心机制] 记录新任务的 PID ---
  new_pid=$!
  gpu_pids[$gpu_id]=$new_pid
  echo "   任务已启动，PID: ${new_pid}，已绑定至 GPU ${gpu_id}"

  # 短暂休眠以避免瞬间启动过多任务导致系统不稳定
  sleep 1

done

echo "✅ 所有 ${TOTAL_RUNS} 组任务已全部启动，等待最后批次的任务执行完毕..."
wait
echo "🎉🎉🎉 Hi-Patch 在 USHCN 上的超参数搜索任务已全部完成！🎉🎉🎉"