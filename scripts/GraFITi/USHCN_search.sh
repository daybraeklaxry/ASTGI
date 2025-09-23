#!/bin/bash

# =====================================================================================
#                 GraFITi 在 USHCN 上的超参数随机搜索脚本
# =====================================================================================

# --- 1. GPU 配置 ---
# 在这里指定你想使用的GPU编号，用逗号隔开，例如 "0,1,2,3"
GPU_IDS="0,1,2,3"

# 将GPU_IDS字符串转换为bash数组
IFS=',' read -r -a GPUS <<< "$GPU_IDS"
NUM_GPUS=${#GPUS[@]}

# [核心机制] 使用关联数组来追踪每个GPU上运行的进程ID (PID)
declare -A gpu_pids
for gpu in "${GPUS[@]}"; do
    gpu_pids[$gpu]=""
done

echo "🚀 将在 ${NUM_GPUS} 个GPU (${GPUS[*]}) 上并行执行 GraFITi 任务，采用随机搜索策略。"

# --- 2. 固定参数 (为 USHCN 数据集和公平比较进行调整) ---
model_name="GraFITi"
dataset_root_path="storage/datasets/USHCN"
dataset_name="USHCN"
# [公平比较] 遵循论文及公开代码的设置
seq_len=150
pred_len=3
enc_in=5
dec_in=5
c_out=5

# --- 3. 超参数随机搜索范围 ---
# [调优策略] d_models: 对应论文中的 "hidden nodes in dense layers"，覆盖从小到大的模型容量
d_models=(16 32 64 128 256)
# [调优策略] n_layers: 对应论文中的 "L"，即 GNN 的层数
n_layers=(1 2 3 4)
# [调优策略] n_heads: 对应论文中的 "#heads in MAB"，即多头注意力机制的头数
n_heads=(1 2 4)
# [调优策略] batch_sizes: 探索不同批次大小对模型收敛和性能的影响
batch_sizes=(16 32 64)
# [调优策略] lrs: 覆盖一个较广的学习率范围，以寻找最优值
lrs=(0.001)


# --- 4. 随机搜索设置 ---
# [调优策略] 在指定搜索空间内随机组合超参数，运行指定的总次数
TOTAL_RUNS=256

# --- [核心机制] 动态寻找空闲GPU的函数 ---
find_free_gpu() {
    local free_gpu_id=""
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
        # 如果没有找到空闲GPU，则等待任意一个后台任务结束
        if [[ -z "$free_gpu_id" ]]; then
            wait -n
        fi
    done
    echo "$free_gpu_id"
}


echo "🚀 开始 GraFITi 在 USHCN 上的超参数搜索，总共将运行 ${TOTAL_RUNS} 组实验..."

# --- 5. 随机搜索主循环 ---
for (( i=1; i<=${TOTAL_RUNS}; i++ )); do

  # --- 随机采样超参数 ---
  dm=${d_models[$((RANDOM % ${#d_models[@]}))]}
  n_l=${n_layers[$((RANDOM % ${#n_layers[@]}))]}
  n_h=${n_heads[$((RANDOM % ${#n_heads[@]}))]}
  bs=${batch_sizes[$((RANDOM % ${#batch_sizes[@]}))]}
  lr=${lrs[$((RANDOM % ${#lrs[@]}))]}

  # --- [核心机制] GPU 动态分配 ---
  gpu_id=$(find_free_gpu)

  # --- 构建唯一的模型ID ---
  model_id="${model_name}_${dataset_name}_sl${seq_len}_pl${pred_len}_dm${dm}_nl${n_l}_nh${n_h}_lr${lr}_bs${bs}_run${i}"

  echo "-----------------------------------------------------------------------"
  echo "📈 启动任务 [${i}/${TOTAL_RUNS}] -> 动态分配至空闲 GPU ${gpu_id}"
  echo "   Arch: d_model=${dm}, n_layers=${n_l}, n_heads=${n_h}"
  echo "   Train: bs=${bs}, lr=${lr}"
  echo "   Model ID: ${model_id}"
  echo "-----------------------------------------------------------------------"

  # --- 在后台启动训练任务 ---
  # 注意：这里的 `main.py` 脚本参数需要与 GraFITi 项目的实际接收参数相匹配
  (
    python main.py \
      --gpu_id $gpu_id \
      --is_training 1 \
      --model_id "$model_id" \
      --model_name "$model_name" \
      --dataset_root_path "$dataset_root_path" \
      --dataset_name "$dataset_name" \
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
      --n_layers "$n_l" \
      --n_heads "$n_h"
  ) &

  # --- [核心机制] 记录新任务的 PID ---
  new_pid=$!
  gpu_pids[$gpu_id]=$new_pid
  echo "   任务已启动，PID: ${new_pid}，已绑定至 GPU ${gpu_id}"

  sleep 1 # 短暂休眠以避免瞬间启动过多任务

done

echo "✅ 所有 ${TOTAL_RUNS} 组任务已启动，等待最后批次的任务执行完毕..."
wait
echo "🎉🎉🎉 GraFITi 在 USHCN 上的超参数搜索任务已全部完成！🎉🎉🎉"