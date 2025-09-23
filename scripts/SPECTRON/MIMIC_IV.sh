#!/bin.bash

# ======================= [ 用户配置区域 ] =======================
# 在下面的括号中填入您希望使用的GPU编号，用空格隔开。
# 示例: GPUS_TO_USE=(0 1 2 3)
GPUS_TO_USE=(0 5 6)
# =================================================================

# --- GPU并行设置 (自动计算) ---
NUM_GPUS=${#GPUS_TO_USE[@]}
job_count=0

# --- 固定参数 (针对 MIMIC-IV 任务) ---
model_name="$(basename "$(dirname "$(readlink -f "$0")")")"
dataset_root_path=storage/datasets/MIMIC_IV
dataset_name=$(basename "$0" .sh)
seq_len=2160
pred_len=720
enc_in=100
# 注意：对于SPECTRON这类模型，dec_in 和 c_out 参数可能不直接使用，
# 但为了脚本兼容性，我们根据变量数进行设置
dec_in=100
c_out=100

# =================================================================================
# --- 严谨优化的超参数搜索空间 (针对 MIMIC-IV 的稀疏性和复杂性) ---
d_models=(64 128)                     # [优化] 匹配并探索必要的模型容量
n_heads_options=(4 8)                 # 保持不变，是合理的范围
batch_sizes=(16 32)                   # [优化] 探索不同批量大小，兼顾性能与显存
lrs=(0.001 0.0005 0.0001)              # [关键优化] 必须搜索学习率
num_kernels=(32 64 96 128)            # [最关键优化] 大幅提升谱分辨率以应对复杂信号
num_intra_layers=(1 2 3)              # [优化] 探索合适的谱交互深度
dropouts=(0.1 0.2 0.3)                # 保持不变，是很好的正则化范围
# =================================================================================

# --- 随机搜索设置 ---
TOTAL_RUNS=128 # 考虑到更大的搜索空间，可以适当增加运行次数

echo "🚀 开始 SPECTRON 在 MIMIC-IV 上的【严谨优化版】随机搜索..."
echo "   将在 ${NUM_GPUS} 个指定GPU上运行: (${GPUS_TO_USE[*]})"
echo "   总共将启动 ${TOTAL_RUNS} 组实验..."

# --- 随机搜索循环 ---
for (( i=0; i<${TOTAL_RUNS}; i++ )); do

  # --- 随机采样超参数 ---
  dm=${d_models[$((RANDOM % ${#d_models[@]}))]}
  bs=${batch_sizes[$((RANDOM % ${#batch_sizes[@]}))]}
  lr=${lrs[$((RANDOM % ${#lrs[@]}))]}
  nk=${num_kernels[$((RANDOM % ${#num_kernels[@]}))]}
  nil=${num_intra_layers[$((RANDOM % ${#num_intra_layers[@]}))]}
  dp=${dropouts[$((RANDOM % ${#dropouts[@]}))]}

  while true; do
    nh=${n_heads_options[$((RANDOM % ${#n_heads_options[@]}))]}
    if [ $((dm % nh)) -eq 0 ]; then
      break
    fi
  done

  # --- 从指定的GPU列表中分配GPU ---
  index=$((job_count % NUM_GPUS))
  gpu_id=${GPUS_TO_USE[$index]}

  model_id="${model_name}_${dataset_name}_sl${seq_len}_pl${pred_len}_dm${dm}_nh${nh}_nk${nk}_nil${nil}_dp${dp}_lr${lr}_bs${bs}_run${i}"

  echo "----------------------------------------------------"
  echo "📈 启动随机搜索任务 [${job_count}] -> 分配至指定GPU ${gpu_id}"
  echo "   Config: d_model=${dm}, n_heads=${nh}, bs=${bs}, lr=${lr}, kernels=${nk}, intra_layers=${nil}, dropout=${dp}"
  echo "   Model ID: ${model_id}"
  echo "----------------------------------------------------"

  # 在后台启动训练任务，并设置CUDA_VISIBLE_DEVICES
  CUDA_VISIBLE_DEVICES=${gpu_id} python main.py \
    --is_training 1 \
    --model_id "$model_id" \
    --model_name "$model_name" \
    --d_model "$dm" \
    --n_heads "$nh" \
    --dropout "$dp" \
    --spectron_num_kernels "$nk" \
    --spectron_d_max 5.0 \
    --spectron_num_intra_layers "$nil" \
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
    --use_multi_gpu 0 &

  job_count=$((job_count + 1))
  sleep 2

  if [[ $(jobs -r -p | wc -l) -ge $NUM_GPUS ]]; then
      wait -n
  fi
done

echo "✅ 所有任务已启动，等待最后批次的任务执行完毕..."
wait
echo "🎉🎉🎉 MIMIC-IV 随机搜索任务已全部完成！🎉🎉🎉"