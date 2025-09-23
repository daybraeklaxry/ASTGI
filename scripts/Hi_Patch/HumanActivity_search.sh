#!/bin/bash

# =====================================================================================
#           Hi-Patch 在 HumanActivity 上的超参数搜索脚本 (高效GPU利用)
# =====================================================================================

# --- 1. GPU 配置 ---
# 在这里指定你想使用的GPU编号，用逗号隔开，例如 "0,1,2,3" 或 "4,5,6,7"
GPU_IDS="0,1,2,3,4,5,6,7" # 假设您有8个GPU可用

# 将GPU_IDS字符串转换为bash数组
IFS=',' read -r -a GPUS <<< "$GPU_IDS"
NUM_GPUS=${#GPUS[@]}

# [核心机制] 使用关联数组来追踪每个GPU上运行的进程ID (PID)
declare -A gpu_pids
for gpu in "${GPUS[@]}"; do
    gpu_pids[$gpu]=""
done

echo "🚀 将在 ${NUM_GPUS} 个GPU (${GPUS[*]}) 上并行执行 Hi-Patch 任务，采用动态调度策略。"

# --- 2. 固定参数 (已为 HumanActivity 和公平比较进行调整) ---
model_name="Hi_Patch"
dataset_root_path="storage/datasets/HumanActivity"
dataset_name="HumanActivity"
seq_len=3000   # << 遵从公平比较基准
pred_len=300  # << 遵从公平比较基准
enc_in=12      # << 根据数据集变量数调整
dec_in=12      # << 根据数据集变量数调整
c_out=12       # << 根据数据集变量数调整

# --- 3.1 通用超参数搜索空间 ---
# [策略] d_model: 探索不同的模型容量
d_models=(32 64 128)
# [策略] batch_size: seq_len=3000 内存开销巨大，必须使用更小的 batch_size 防止 OOM
batch_sizes=(16 32)
# [策略] lrs: 保持一个稳健的搜索范围
lrs=(0.001 0.0005 0.0001)

# --- 3.2 Hi-Patch 专属超参数搜索空间 ---
# [策略] n_layers: GAT 层的数量，探索不同深度的图网络
n_layers_options=(1 2 3)
# [策略] n_heads: 多头注意力的头数
n_heads_options=(2 4 8)
# [策略] patch_len 和 patch_stride: 这是Hi-Patch的核心。
# 我们将 patch_len 和 patch_stride 设置为相等的值，以创建非重叠的补丁。
# 补丁大小的选择决定了模型的“分辨率”和层级结构的数量。
patch_len_options=(60 75 125 250 500 750) # 3000的因子，分别产生12, 6, 4个补丁

# --- 4. 随机搜索设置 ---
# 根据您的计算资源调整总运行次数
TOTAL_RUNNS=512

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
            # 如果没有立即可用的GPU，等待任何一个后台任务完成
            wait -n
        fi
    done
    echo "$free_gpu_id"
}

echo "🚀 开始 Hi-Patch 在 HumanActivity 上的随机搜索，总共将运行 ${TOTAL_RUNNS} 组实验..."

# --- 5. 随机搜索主循环 ---
for (( i=1; i<=${TOTAL_RUNNS}; i++ )); do

  # --- 随机采样超参数 ---
  dm=${d_models[$((RANDOM % ${#d_models[@]}))]}
  bs=${batch_sizes[$((RANDOM % ${#batch_sizes[@]}))]}
  lr=${lrs[$((RANDOM % ${#lrs[@]}))]}

  n_layers=${n_layers_options[$((RANDOM % ${#n_layers_options[@]}))]}
  n_heads=${n_heads_options[$((RANDOM % ${#n_heads_options[@]}))]}
  p_len=${patch_len_options[$((RANDOM % ${#patch_len_options[@]}))]}
  p_stride=$p_len # 设置为非重叠补丁

  # --- GPU 动态分配 ---
  gpu_id=$(find_free_gpu)

  # --- 构建唯一的模型ID ---
  model_id="${model_name}_${dataset_name}_sl${seq_len}_pl${pred_len}_dm${dm}_L${n_layers}_nh${n_heads}_pl${p_len}_ps${p_stride}_lr${lr}_bs${bs}_run${i}"

  echo "-----------------------------------------------------------------------"
  echo "📈 启动任务 [${i}/${TOTAL_RUNNS}] -> 动态分配至空闲 GPU ${gpu_id}"
  echo "   Arch: d_model=${dm}, n_layers=${n_layers}, n_heads=${n_heads}"
  echo "   Patching: patch_len=${p_len}, patch_stride=${p_stride}"
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
      --collate_fn "collate_fn_patch" \
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

  # --- 记录新任务的 PID ---
  new_pid=$!
  gpu_pids[$gpu_id]=$new_pid
  echo "   任务已启动，PID: ${new_pid}，已绑定至 GPU ${gpu_id}"

  # 短暂休眠以避免瞬间启动过多任务导致日志混乱
  sleep 1

done

echo "✅ 所有 ${TOTAL_RUNNS} 组任务已启动，等待最后批次的任务执行完毕..."
wait
echo "🎉🎉🎉 Hi-Patch 在 HumanActivity 上的随机搜索任务已全部完成！🎉🎉🎉"