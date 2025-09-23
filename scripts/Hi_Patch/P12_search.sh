#!/bin/bash

# =====================================================================================
#                 Hi-Patch 在 P12 数据集上的超参数随机搜索脚本
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

# --- 2. 固定参数 (为 P12 数据集和公平比较进行调整) ---
model_name="Hi_Patch"
dataset_root_path="storage/datasets/P12" # 请确保路径正确
dataset_name="P12"
# [数据集特定参数] 根据 P12 数据集特性设置
seq_len=36
pred_len=3 # 预测长度，根据您的任务需求设定
enc_in=36  # 输入变量数
c_out=36   # 输出变量数

# --- 3. Hi-Patch 超参数搜索范围 ---

# [模型结构] d_model: 节点嵌入维度
d_models=(32 64 128)
# [模型结构] n_layers: GAT 层的数量
n_layers_options=(1 2 3)
# [模型结构] n_heads: 多头注意力的头数
n_heads_options=(1 2 4)

# [Patching策略] patch_len: 每个 patch 的长度 (时间步数)
# 注意：patch_stride 将与 patch_len 保持一致以实现无重叠分块
patch_lens=(4 6 9 12)

# [训练参数] batch_size: 批处理大小
batch_sizes=(16 32 64)
# [训练参数] lrs: 学习率
lrs=(0.0005 0.001 0.002)

# --- 4. 随机搜索设置 ---
# 定义总共要运行的实验次数
TOTAL_RUNS=512

# --- [核心机制] 动态寻找空闲GPU的函数 ---
find_free_gpu() {
    local free_gpu_id=""
    while [[ -z "$free_gpu_id" ]]; do
        for gpu_id in "${GPUS[@]}"; do
            pid=${gpu_pids[$gpu_id]}
            # 检查PID是否为空，或者对应的进程是否已不存在
            if [[ -z "$pid" ]] || ! kill -0 "$pid" 2>/dev/null; then
                free_gpu_id=$gpu_id
                unset gpu_pids[$gpu_id] # 标记为"已分配"（通过清空pid）
                break
            fi
        done
        # 如果所有GPU都在忙，则等待任何一个后台任务结束
        if [[ -z "$free_gpu_id" ]]; then
            wait -n
        fi
    done
    echo "$free_gpu_id"
}

echo "🚀 开始 Hi-Patch 在 P12 上的超参数搜索，总共将运行 ${TOTAL_RUNS} 组实验..."

# --- 5. 随机搜索主循环 ---
for (( i=1; i<=${TOTAL_RUNS}; i++ )); do

  # --- 随机采样超参数 ---
  dm=${d_models[$((RANDOM % ${#d_models[@]}))]}
  n_layers=${n_layers_options[$((RANDOM % ${#n_layers_options[@]}))]}
  n_heads=${n_heads_options[$((RANDOM % ${#n_heads_options[@]}))]}
  p_len=${patch_lens[$((RANDOM % ${#patch_lens[@]}))]}
  p_stride=$p_len # 保持步幅与长度一致
  bs=${batch_sizes[$((RANDOM % ${#batch_sizes[@]}))]}
  lr=${lrs[$((RANDOM % ${#lrs[@]}))]}

  # --- [核心机制] GPU 动态分配 ---
  gpu_id=$(find_free_gpu)

  # --- 构建唯一的模型ID ---
  model_id="${model_name}_${dataset_name}_sl${seq_len}_pl${pred_len}_dm${dm}_nl${n_layers}_nh${n_heads}_plen${p_len}_bs${bs}_lr${lr}_run${i}"

  echo "-----------------------------------------------------------------------"
  echo "📈 启动任务 [${i}/${TOTAL_RUNS}] -> 动态分配至空闲 GPU ${gpu_id}"
  echo "   Arch: d_model=${dm}, n_layers=${n_layers}, n_heads=${n_heads}"
  echo "   Patch: patch_len=${p_len}, patch_stride=${p_stride}"
  echo "   Train: batch_size=${bs}, learning_rate=${lr}"
  echo "   Model ID: ${model_id}"
  echo "-----------------------------------------------------------------------"

  # --- 在后台启动训练任务 ---
  (
    # 假设您的主程序是 main.py，并且接受以下参数
    # 您需要根据 Hi-Patch 的实际接收参数名进行调整
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

  # --- [核心机制] 记录新任务的 PID ---
  new_pid=$!
  gpu_pids[$gpu_id]=$new_pid
  echo "   任务已启动，PID: ${new_pid}，已绑定至 GPU ${gpu_id}"

  # 短暂休眠，避免瞬间启动过多任务导致系统不稳定
  sleep 1

done

echo "✅ 所有 ${TOTAL_RUNS} 组任务已启动，等待最后批次的任务执行完毕..."
wait
echo "🎉🎉🎉 Hi-Patch 在 P12 上的超参数搜索任务已全部完成！🎉🎉🎉"