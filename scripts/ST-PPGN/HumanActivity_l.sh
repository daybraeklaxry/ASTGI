#!/bin/bash

# =====================================================================================
#           ST-PPGN 在 HumanActivity 上的超参数搜索脚本 (高效GPU利用)
# =====================================================================================

# --- 1. GPU 配置 ---
# 在这里指定你想使用的GPU编号，用逗号隔开，例如 "0,1,2,3" 或 "4,5,6,7"
GPU_IDS="0,4,5,6,7" # 假设您有8个GPU可用

# 将GPU_IDS字符串转换为bash数组
IFS=',' read -r -a GPUS <<< "$GPU_IDS"
NUM_GPUS=${#GPUS[@]}

# [核心机制] 使用关联数组来追踪每个GPU上运行的进程ID (PID)
declare -A gpu_pids
for gpu in "${GPUS[@]}"; do
    gpu_pids[$gpu]=""
done

echo "🚀 将在 ${NUM_GPUS} 个GPU (${GPUS[*]}) 上并行执行 ST-PPGN 任务，采用动态调度策略。"

# --- 2. 固定参数 (已为 HumanActivity 和公平比较进行调整) ---
ABLATION_NAME="Sensitivity_Analysis_l"
model_name="ST-PPGN"
dataset_root_path="storage/datasets/HumanActivity"
dataset_name="HumanActivity"
seq_len=3000   # << 遵从公平比较基准
pred_len=300   # << 遵从公平比较基准
enc_in=12      # << 根据数据集变量数调整
dec_in=12      # << 根据数据集变量数调整
c_out=12       # << 根据数据集变量数调整

# --- 3.1 通用超参数搜索空间 (针对 HumanActivity 优化) ---
# [策略] d_model: 由于变量数减少，从较小的值开始探索，但保留较大值以应对长序列复杂性
d_models=(128)
# [策略] batch_size: seq_len=3000 内存开销巨大，必须使用更小的 batch_size 防止 OOM
batch_sizes=(8)
# [策略] lrs: 保持一个稳健的搜索范围
lrs=(0.002)
# [策略] dropouts: 样本量小，过拟合风险高，探索更高的 dropout 值
dropouts=(0.1)

# --- 3.2 ST-PPGN 专属超参数搜索空间 (针对 HumanActivity 优化) ---
# [策略] k_neighbors: 序列非常长，需要更大的 k 来捕获更广阔的局部时空上下文
stppgn_k_neighbors_options=(96)
# [策略] prop_layers: 保持不变。层数过多在长序列上计算成本高且可能导致过平滑
stppgn_prop_layers_options=(1 2 3 4 5)
# [策略] channel_dims: 变量数只有12，channel_dim 不需要太大，有助于降低模型复杂度和过拟合
stppgn_channel_dims=(64)
# [策略] time_dims: 时间维度长，需要足够的容量来编码时间信息，聚焦于表现好的值
stppgn_time_dims=(128)
# [策略] mlp_ratios: 保持不变，探索不同容量的MLP
stppgn_mlp_ratios=(4.0)
### 新增 ###
# [策略] channel_dist_weight (w_c): 探索不同强度的“同通道优先”先验。
# 1.0表示无偏好，值越大，对同通道邻居的偏好越强。
stppgn_channel_dist_weights=(1.0)


# --- 4. 随机搜索设置 ---
# 由于每个任务耗时更长，适当减少总运行次数
TOTAL_RUNS=1

# --- [核心机制] 动态寻找空闲GPU的函数 (保持不变) ---
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

echo "🚀 开始 ST-PPGN 在 HumanActivity 上的随机搜索，总共将运行 ${TOTAL_RUNS} 组实验..."

# --- 5. 随机搜索主循环 ---
for l_val in "${stppgn_prop_layers_options[@]}"; do

  # --- 随机采样超参数 ---
  dm=${d_models[$((RANDOM % ${#d_models[@]}))]}
  bs=${batch_sizes[$((RANDOM % ${#batch_sizes[@]}))]}
  lr=${lrs[$((RANDOM % ${#lrs[@]}))]}
  dp=${dropouts[$((RANDOM % ${#dropouts[@]}))]}

  k_nn=${stppgn_k_neighbors_options[$((RANDOM % ${#stppgn_k_neighbors_options[@]}))]}
  n_prop=${l_val}
  c_dim=${stppgn_channel_dims[$((RANDOM % ${#stppgn_channel_dims[@]}))]}
  t_dim=${stppgn_time_dims[$((RANDOM % ${#stppgn_time_dims[@]}))]}
  mlp_r=${stppgn_mlp_ratios[$((RANDOM % ${#stppgn_mlp_ratios[@]}))]}
  ### 新增 ###
  w_c=${stppgn_channel_dist_weights[$((RANDOM % ${#stppgn_channel_dist_weights[@]}))]}


  # --- GPU 动态分配 ---
  gpu_id=$(find_free_gpu)

  ### 修改 ###
  # --- 构建唯一的模型ID ---
  model_id="${model_name}_${dataset_name}_Sens_L${n_prop}_sl${seq_len}_pl${pred_len}_dm${dm}_K${k_nn}_dc${c_dim}_dt${t_dim}"
  echo "-----------------------------------------------------------------------"
  echo "📈 启动 L值敏感性测试 -> prop_layers=${n_prop}，分配至空闲 GPU ${gpu_id}"
  echo "   Arch: d_model=${dm}, prop_layers=${n_prop}, dropout=${dp}"
  ### 修改 ###
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

  # --- 记录新任务的 PID ---
  new_pid=$!
  gpu_pids[$gpu_id]=$new_pid
  echo "   任务已启动，PID: ${new_pid}，已绑定至 GPU ${gpu_id}"

  sleep 1

done

echo "✅ 所有 ${TOTAL_RUNS} 组任务已启动，等待最后批次的任务执行完毕..."
wait
echo "🎉🎉🎉 ST-PPGN 在 HumanActivity 上的随机搜索任务已全部完成！🎉🎉🎉"