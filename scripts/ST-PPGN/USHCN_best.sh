#!/bin/bash

# =====================================================================================
#                 ST-PPGN 在 USHCN 上的超参数精调脚本 (基于已知最优参数)
# =====================================================================================

# --- 1. GPU 配置 ---
# 在这里指定你想使用的GPU编号，用逗号隔开，例如 "0,1,2,3"
GPU_IDS="6"

# 将GPU_IDS字符串转换为bash数组
IFS=',' read -r -a GPUS <<< "$GPU_IDS"
NUM_GPUS=${#GPUS[@]}

# [核心机制] 使用关联数组来追踪每个GPU上运行的进程ID (PID)
declare -A gpu_pids
for gpu in "${GPUS[@]}"; do
    gpu_pids[$gpu]=""
done

echo "🚀 将在 ${NUM_GPUS} 个GPU (${GPUS[*]}) 上并行执行 ST-PPGN 任务，采用基于最优参数的精调策略。"

# --- 2. 固定参数 (为 USHCN 数据集和公平比较进行调整) ---
ABLATION_NAME="Ablation_4"
model_name="ST-PPGN"
dataset_root_path="storage/datasets/USHCN"
dataset_name="USHCN"
# [公平比较] 遵循 HyperIMTS 的设置
seq_len=150
pred_len=3
enc_in=5
dec_in=5
c_out=5

# --- 3.1 基于最优参数的精调搜索范围 ---
# 最优参数: dm=128, bs=32, lr=0.002, dp=0.25, k=48, L=1, dc=16, dt=32, mlp_r=2.0, w_c=1.0

# [调优策略] d_model: 以 128 为中心，探索略微调整模型容量带来的影响。
d_models=(64)
# [调优策略] batch_size: 以 32 为中心，探索邻近的批次大小。
batch_sizes=(32)
# [调优策略] lrs: 以 0.002 为中心，进行更精细的学习率搜索。
lrs=(0.001)
# [调优策略] dropouts: 以 0.25 为中心，这是一个非常适合精调的范围。
dropouts=(0.25)

# --- 3.2 ST-PPGN 专属超参数精调 ---
# [调优策略] k_neighbors: 以 48 为中心，测试邻域大小的敏感性。
stppgn_k_neighbors_options=(80)
# [调优策略] prop_layers: 最优值为 1，我们依然探索 2 层是否可能带来提升。
stppgn_prop_layers_options=(3)
# [调优策略] channel_dims: 最优值为 16，对于5个变量，我们聚焦于较小的嵌入维度。
stppgn_channel_dims=(24)
# [调优策略] time_dims: 以 32 为中心，继续验证最优时间编码维度。
stppgn_time_dims=(24)
# [调优策略] mlp_ratios: 以 2.0 为中心，这是一个鲁棒的结构参数，值得再次确认。
stppgn_mlp_ratios=(2.0)
# [调优策略] channel_dist_weight (w_c): 最优值为 1.0 (无偏好)。我们新增一些值来探索施加“同通道优先”的软约束是否能提升性能。
stppgn_channel_dist_weights=(1.0)


# --- 4. 随机搜索设置 ---
# [调优策略] 由于搜索空间已显著缩小且更有针对性，可以适当减少总运行次数。
TOTAL_RUNS=1

# --- [核心机制] 动态寻找空闲GPU的函数 ---
find_free_gpu() {
    local free_gpu_id=""
    while [[ -z "$free_gpu_id" ]]; do
        for gpu_id in "${GPUS[@]}"; do
            pid=${gpu_pids[$gpu_id]}
            if [[ -z "$pid" ]] || ! kill -0 "$pid" 2>/dev/null; then
                free_gpu_id=$gpu_id
                unset gpu_pids[$gpu_id]
                break
            fi
        done
        if [[ -z "$free_gpu_id" ]]; then
            wait -n
        fi
    done
    echo "$free_gpu_id"
}


echo "🚀 开始 ST-PPGN 在 USHCN 上的精细调优，总共将运行 ${TOTAL_RUNS} 组实验..."

# --- 5. 随机搜索主循环 ---
for (( i=1; i<=${TOTAL_RUNS}; i++ )); do

  # --- 随机采样超参数 ---
  dm=${d_models[$((RANDOM % ${#d_models[@]}))]}
  bs=${batch_sizes[$((RANDOM % ${#batch_sizes[@]}))]}
  lr=${lrs[$((RANDOM % ${#lrs[@]}))]}
  dp=${dropouts[$((RANDOM % ${#dropouts[@]}))]}
  k_nn=${stppgn_k_neighbors_options[$((RANDOM % ${#stppgn_k_neighbors_options[@]}))]}
  n_prop=${stppgn_prop_layers_options[$((RANDOM % ${#stppgn_prop_layers_options[@]}))]}
  c_dim=${stppgn_channel_dims[$((RANDOM % ${#stppgn_channel_dims[@]}))]}
  t_dim=${stppgn_time_dims[$((RANDOM % ${#stppgn_time_dims[@]}))]}
  mlp_r=${stppgn_mlp_ratios[$((RANDOM % ${#stppgn_mlp_ratios[@]}))]}
  w_c=${stppgn_channel_dist_weights[$((RANDOM % ${#stppgn_channel_dist_weights[@]}))]}

  # --- [核心机制] GPU 动态分配 ---
  gpu_id=$(find_free_gpu)

  # --- 构建唯一的模型ID ---
  model_id="${model_name}_${dataset_name}_sl${seq_len}_pl${pred_len}_dm${dm}_L${n_prop}_k${k_nn}_dc${c_dim}_dt${t_dim}_wc${w_c}_dp${dp}_lr${lr}_bs${bs}_mlp_${mlp_r}_run${i}"

  echo "-----------------------------------------------------------------------"
  echo "📈 启动任务 [${i}/${TOTAL_RUNS}] -> 动态分配至空闲 GPU ${gpu_id}"
  echo "   Arch: d_model=${dm}, prop_layers=${n_prop}, dropout=${dp}"
  echo "   ST-PPGN: k_neighbors=${k_nn}, channel_dim=${c_dim}, time_dim=${t_dim}, mlp_ratio=${mlp_r}, channel_dist_weight=${w_c}"
  echo "   Train: bs=${bs}, lr=${lr}"
  echo "   Model ID: ${model_id}"
  echo "-----------------------------------------------------------------------"

  # --- 在后台启动训练任务 ---
  (
    CUDA_VISIBLE_DEVICES=${gpu_id} python main.py \
      --ablation_name "$ABLATION_NAME" \
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
      --dropout "$dp" \
      --stppgn_k_neighbors "$k_nn" \
      --stppgn_prop_layers "$n_prop" \
      --stppgn_channel_dim "$c_dim" \
      --stppgn_time_dim "$t_dim" \
      --stppgn_mlp_ratio "$mlp_r" \
      --stppgn_channel_dist_weight "$w_c"
  ) &

  # --- [核心机制] 记录新任务的 PID ---
  new_pid=$!
  gpu_pids[$gpu_id]=$new_pid
  echo "   任务已启动，PID: ${new_pid}，已绑定至 GPU ${gpu_id}"

  sleep 1

done

echo "✅ 所有 ${TOTAL_RUNS} 组任务已启动，等待最后批次的任务执行完毕..."
wait
echo "🎉🎉🎉 ST-PPGN 在 USHCN 上的超参数精调任务已全部完成！🎉🎉🎉"