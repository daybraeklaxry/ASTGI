#!/bin/bash

# =====================================================================================
#                 ST-PPGN 在 MIMIC-III 上的超参数搜索脚本 (高效GPU利用)
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

echo "🚀 将在 ${NUM_GPUS} 个GPU (${GPUS[*]}) 上并行执行 ST-PPGN 任务，采用动态调度策略。"

# --- 2. 固定参数 (基于MIMIC-III数据集和公平比较原则) ---
model_name="ST-PPGN"
dataset_root_path="storage/datasets/MIMIC_III"
dataset_name="MIMIC_III"
seq_len=72
pred_len=3
enc_in=96
dec_in=96
c_out=96

# --- 3. 超参数搜索空间 (基于策略分析) ---

# [策略] d_model: 以P12的SOTA(96)为基础，向上探索以适应更多的变量
d_models=(96 128 192)
# [策略] batch_size: 考虑到d_model增大可能增加显存消耗，暂时固定为32，可根据实际情况调整
batch_sizes=(32)
# [策略] lrs: 沿用P12的有效范围，对于大型数据集是稳健的选择
lrs=(0.0001 0.0002 0.0005)
# [策略] dropouts: 沿用P12的有效范围
dropouts=(0.1 0.2 0.3)

# --- 3.2 ST-PPGN 专属超参数 ---
# [策略] k_neighbors: 针对MIMIC-III的稀疏性，探索更广的范围。32(局部)、64(P12 SOTA)、96(更广视野)
stppgn_k_neighbors_options=(80 96 112)
# [策略] prop_layers: 保持不变，层数是核心架构，其与其它参数的交互值得探索
stppgn_prop_layers_options=(1 2 3)
# [策略] channel_dims: 以64为基础，增加96以更好地编码96个变量
stppgn_channel_dims=(64 96)
# [策略] time_dims: 沿用P12的有效范围
stppgn_time_dims=(64 128)
# [策略] mlp_ratios: 沿用P12的有效范围
stppgn_mlp_ratios=(2.0 3.0 4.0)
stppgn_channel_dist_weights=(1.0)
# --- 4. 随机搜索设置 ---
TOTAL_RUNS=256

# --- [核心机制] 动态寻找空闲GPU的函数 ---
# (此函数与您的P12脚本完全相同，无需修改)
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


echo "🚀 开始 ST-PPGN 在 MIMIC-III 上的随机搜索，总共将运行 ${TOTAL_RUNS} 组实验..."

# --- 5. 随机搜索主循环 ---
for (( i=1; i<=${TOTAL_RUNS}; i++ )); do

  # --- 随机采样超参数 ---
  # 通用参数
  dm=${d_models[$((RANDOM % ${#d_models[@]}))]}
  bs=${batch_sizes[$((RANDOM % ${#batch_sizes[@]}))]}
  lr=${lrs[$((RANDOM % ${#lrs[@]}))]}
  dp=${dropouts[$((RANDOM % ${#dropouts[@]}))]}

  # ST-PPGN 专属参数
  k_nn=${stppgn_k_neighbors_options[$((RANDOM % ${#stppgn_k_neighbors_options[@]}))]}
  n_prop=${stppgn_prop_layers_options[$((RANDOM % ${#stppgn_prop_layers_options[@]}))]}
  c_dim=${stppgn_channel_dims[$((RANDOM % ${#stppgn_channel_dims[@]}))]}
  t_dim=${stppgn_time_dims[$((RANDOM % ${#stppgn_time_dims[@]}))]}
  mlp_r=${stppgn_mlp_ratios[$((RANDOM % ${#stppgn_mlp_ratios[@]}))]}
  w_c=${stppgn_channel_dist_weights[$((RANDOM % ${#stppgn_channel_dist_weights[@]}))]}

  # --- [核心机制] GPU 动态分配 ---
  gpu_id=$(find_free_gpu)

  # --- 构建唯一的模型ID (包含ST-PPGN关键参数) ---
  model_id="${model_name}_${dataset_name}_sl${seq_len}_pl${pred_len}_dm${dm}_L${n_prop}_k${k_nn}_dc${c_dim}_dt${t_dim}_wc${w_c}_dp${dp}_lr${lr}_bs${bs}_mlp${mlp_r}_run${i}"

  echo "-----------------------------------------------------------------------"
  echo "📈 启动任务 [${i}/${TOTAL_RUNS}] -> 动态分配至空闲 GPU ${gpu_id}"
  echo "   Arch: d_model=${dm}, prop_layers=${n_prop}, dropout=${dp}"
  echo "   ST-PPGN: k_neighbors=${k_nn}, channel_dim=${c_dim}, time_dim=${t_dim}, mlp_ratio=${mlp_r}"
  echo "   Train: bs=${bs}, lr=${lr}"
  echo "   Model ID: ${model_id}"
  echo "-----------------------------------------------------------------------"

  # --- 在后台启动训练任务 (注意参数已更新为ST-PPGN的) ---
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

  sleep 1 # 短暂休眠，确保日志顺序和PID正确捕获

done

echo "✅ 所有 ${TOTAL_RUNS} 组任务已启动，等待最后批次的任务执行完毕..."
wait
echo "🎉🎉🎉 ST-PPGN 在 MIMIC-III 上的随机搜索任务已全部完成！🎉🎉🎉"