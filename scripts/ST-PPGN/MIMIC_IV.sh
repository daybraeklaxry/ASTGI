#!/bin/bash

# =====================================================================================
#                 ST-PPGN 在 MIMIC-IV 上的超参数搜索脚本 (高效GPU利用)
# =====================================================================================

# --- 1. GPU 配置 ---
# 在这里指定你想使用的GPU编号，用逗号隔开，例如 "0,1,2,3" 或 "4,5,6,7"
GPU_IDS="0,1,2,3,4,5,6,7" # 假设使用4个GPU

# 将GPU_IDS字符串转换为bash数组
IFS=',' read -r -a GPUS <<< "$GPU_IDS"
NUM_GPUS=${#GPUS[@]}

# [核心机制] 使用关联数组来追踪每个GPU上运行的进程ID (PID)
declare -A gpu_pids
for gpu in "${GPUS[@]}"; do
    gpu_pids[$gpu]=""
done

echo "🚀 将在 ${NUM_GPUS} 个GPU (${GPUS[*]}) 上并行执行 ST-PPGN 在 MIMIC-IV 上的任务，采用动态调度策略。"

# --- 2. 固定参数 (基于MIMIC-IV和公平比较原则) ---
model_name="ST-PPGN"
dataset_root_path="storage/datasets/MIMIC_IV"
dataset_name="MIMIC_IV"
# 为了与 HyperIMTS 等模型公平比较，采用以下设置
seq_len=2160
pred_len=3
enc_in=100  # MIMIC-IV 变量数
dec_in=100
c_out=100

# --- 3.1 通用超参数搜索空间 (针对MIMIC-IV进行调整) ---
# [策略] d_model: MIMIC-IV更复杂，需要更大模型容量
d_models=(96 128 192)
# [策略] batch_size: MIMIC-IV点云密度大，为防止OOM，采用更小的batch size
batch_sizes=(8 16)
# [策略] lrs: 较大的模型和数据集可能需要更精细的LR，向更小的值探索
lrs=(0.001 0.0005 0.0001)
# [策略] dropouts: 保持标准范围，防止大型模型过拟合
dropouts=(0.1 0.2 0.3)

# --- 3.2 ST-PPGN 专属超参数 (针对MIMIC-IV进行调整) ---
# [策略] k_neighbors: 点云规模和密度大增，需要更大的邻域来捕获有效信息
stppgn_k_neighbors_options=(64 96 128)
# [策略] prop_layers: 更复杂的动态可能需要更深的信息传播
stppgn_prop_layers_options=(2 3 4)
# [策略] channel_dims: 变量数从36增至100，需要更高维的通道嵌入空间
stppgn_channel_dims=(64 96)
# [策略] time_dims: 时间序列长度大幅增加，同样需要更强的时序表达能力
stppgn_time_dims=(64 128)
# [策略] mlp_ratios: 保持标准范围，这是一个相对稳健的参数
stppgn_mlp_ratios=(2.0 3.0 4.0)
stppgn_channel_dist_weights=(1.0)

# --- 4. 随机搜索设置 ---
TOTAL_RUNS=256 # 您可以根据计算资源调整总运行次数

# --- [核心机制] 动态寻找空闲GPU的函数 (与P12脚本完全相同) ---
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

echo "🚀 开始 ST-PPGN 在 MIMIC-IV 上的随机搜索，总共将运行 ${TOTAL_RUNS} 组实验..."

# --- 5. 随机搜索主循环 (与P12脚本完全相同) ---
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
  # --- GPU 动态分配 ---
  gpu_id=$(find_free_gpu)

  # --- 构建唯一的模型ID ---
  model_id="${model_name}_${dataset_name}_sl${seq_len}_pl${pred_len}_dm${dm}_L${n_prop}_k${k_nn}_dc${c_dim}_dt${t_dim}_wc${w_c}_dp${dp}_lr${lr}_bs${bs}_run${i}"

  echo "-----------------------------------------------------------------------"
  echo "📈 启动任务 [${i}/${TOTAL_RUNS}] -> 动态分配至空闲 GPU ${gpu_id}"
  echo "   Arch: d_model=${dm}, prop_layers=${n_prop}, dropout=${dp}"
  echo "   ST-PPGN: k_neighbors=${k_nn}, channel_dim=${c_dim}, time_dim=${t_dim}, mlp_ratio=${mlp_r}"
  echo "   Train: bs=${bs}, lr=${lr}"
  echo "   Model ID: ${model_id}"
  echo "-----------------------------------------------------------------------"

  # --- 在后台启动训练任务 ---
  (
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
      --dec_in "$dec_in" \
      --c_out "$c_out" \
      --loss "MSE" \
      --train_epochs 50 \
      --patience 5 \
      --val_interval 1 \
      --itr 1 \
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

  # --- 记录新任务的 PID ---
  new_pid=$!
  gpu_pids[$gpu_id]=$new_pid
  echo "   任务已启动，PID: ${new_pid}，已绑定至 GPU ${gpu_id}"
  sleep 1

done

echo "✅ 所有 ${TOTAL_RUNS} 组任务已启动，等待最后批次的任务执行完毕..."
wait
echo "🎉🎉🎉 ST-PPGN 在 MIMIC-IV 上的随机搜索任务已全部完成！🎉🎉🎉"