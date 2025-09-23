#!/bin/bash

# =====================================================================================
#                 ST-PPGN 在 P12 上的超参数搜索脚本 (高效GPU利用)
# =====================================================================================

# --- 1. GPU 配置 ---
# 在这里指定你想使用的GPU编号，用逗号隔开，例如 "0,1,2,3" 或 "4,5,6,7"
GPU_IDS="1,2,3"

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

# --- 2. 固定参数 (请根据您的数据集进行调整) ---
ABLATION_NAME="Sensitivity_Analysis_l"
model_name="ST-PPGN"
dataset_root_path="storage/datasets/P12"
dataset_name="P12"
seq_len=36
pred_len=3
enc_in=36
dec_in=36
c_out=36

# --- 3.1 通用超参数 ---
# [策略] d_model: 聚焦于表现最好的64，并探索其附近区域
d_models=(64)
# [策略] batch_size: 固定为表现更好的32
batch_sizes=(8)
# [策略] lrs: 围绕0.001进行更精细的搜索
lrs=(0.0005)
# [策略] dropouts: 保持不变，继续探索
dropouts=(0.1)

# --- 3.2 ST-PPGN 专属超参数 ---
# [策略] k_neighbors: 基于k=32的成功，向上探索更大的感受野
stppgn_k_neighbors_options=(80)
# [策略] prop_layers: 保持不变，继续探索其与其它参数的交互
stppgn_prop_layers_options=(1 2 3 4 5)
# [策略] channel_dims: 暂时固定为表现最好的64，以减少变量
stppgn_channel_dims=(96)
# [策略] time_dims: 聚焦于表现好的64和128
stppgn_time_dims=(128)
# [策略] mlp_ratios: 保持不变，继续探索
stppgn_mlp_ratios=(3.0)

### 新增 ###
# [策略] channel_dist_weight (w_c): 探索不同强度的“同通道优先”先验。
# 1.0表示无偏好，值越大，对同通道邻居的偏好越强。
stppgn_channel_dist_weights=(1.0)


# --- 4. 随机搜索设置 ---
TOTAL_RUNS=1

# --- [核心机制] 动态寻找空闲GPU的函数 ---
# 该函数会循环检查，直到找到一个空闲的GPU并返回其ID
find_free_gpu() {
    local free_gpu_id=""
    while [[ -z "$free_gpu_id" ]]; do
        # 遍历所有可用的GPU
        for gpu_id in "${GPUS[@]}"; do
            pid=${gpu_pids[$gpu_id]}
            # 检查条件：
            # 1. PID为空 (从未分配过任务或任务已结束并被清理)
            # 2. PID不为空，但该进程已不存在 (kill -0 失败)
            if [[ -z "$pid" ]] || ! kill -0 "$pid" 2>/dev/null; then
                free_gpu_id=$gpu_id
                # 清理旧的PID记录，以防万一
                unset gpu_pids[$gpu_id]
                break # 找到了，跳出内层 for 循环
            fi
        done

        # 如果遍历完所有GPU都正忙，则等待任意一个后台任务结束
        if [[ -z "$free_gpu_id" ]]; then
            # echo "⏳ 所有GPU正忙，等待一个任务完成以释放资源..."
            wait -n
        fi
    done
    # 将找到的空闲GPU ID返回给调用者
    echo "$free_gpu_id"
}


echo "🚀 开始 ST-PPGN 在 P12 上的随机搜索，总共将运行 ${TOTAL_RUNS} 组实验..."

# --- 5. 随机搜索主循环 ---
for l_val in "${stppgn_prop_layers_options[@]}"; do

  # --- 随机采样超参数 ---
  # 通用参数
  dm=${d_models[$((RANDOM % ${#d_models[@]}))]}
  bs=${batch_sizes[$((RANDOM % ${#batch_sizes[@]}))]}
  lr=${lrs[$((RANDOM % ${#lrs[@]}))]}
  dp=${dropouts[$((RANDOM % ${#dropouts[@]}))]}

  # ST-PPGN 专属参数
  k_nn=${stppgn_k_neighbors_options[$((RANDOM % ${#stppgn_k_neighbors_options[@]}))]}
  n_prop=${l_val}
  c_dim=${stppgn_channel_dims[$((RANDOM % ${#stppgn_channel_dims[@]}))]}
  t_dim=${stppgn_time_dims[$((RANDOM % ${#stppgn_time_dims[@]}))]}
  mlp_r=${stppgn_mlp_ratios[$((RANDOM % ${#stppgn_mlp_ratios[@]}))]}
  ### 新增 ###
  w_c=${stppgn_channel_dist_weights[$((RANDOM % ${#stppgn_channel_dist_weights[@]}))]}


  # --- [核心机制] GPU 动态分配 ---
  # 调用函数来获取一个当前空闲的GPU ID
  gpu_id=$(find_free_gpu)

  ### 修改 ###
  # --- 构建唯一的模型ID (包含ST-PPGN关键参数) ---
  model_id="${model_name}_${dataset_name}_Sens_L${n_prop}_sl${seq_len}_pl${pred_len}_dm${dm}_K${k_nn}_dc${c_dim}_dt${t_dim}"
  echo "-----------------------------------------------------------------------"
  echo "📈 启动 L值敏感性测试 -> prop_layers=${n_prop}，分配至空闲 GPU ${gpu_id}"
  echo "   Arch: d_model=${dm}, prop_layers=${n_prop}, dropout=${dp}"
  ### 修改 ###
  echo "   ST-PPGN: k_neighbors=${k_nn}, channel_dim=${c_dim}, time_dim=${t_dim}, mlp_ratio=${mlp_r}, channel_dist_weight=${w_c}"
  echo "   Train: bs=${bs}, lr=${lr}"
  echo "   Model ID: ${model_id}"
  echo "-----------------------------------------------------------------------"

  # --- 在后台启动训练任务 (注意参数已更新为ST-PPGN的) ---
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
  # $! 是 bash 的一个特殊变量，它会保存最后一个被放到后台的进程的PID
  new_pid=$!
  gpu_pids[$gpu_id]=$new_pid
  echo "   任务已启动，PID: ${new_pid}，已绑定至 GPU ${gpu_id}"

  sleep 1 # 短暂休眠，确保日志顺序和PID正确捕获

done

echo "✅ 所有 ${TOTAL_RUNS} 组任务已启动，等待最后批次的任务执行完毕..."
wait
echo "🎉🎉🎉 ST-PPGN 在 P12 上的随机搜索任务已全部完成！🎉🎉🎉"