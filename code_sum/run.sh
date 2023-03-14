export CUDA_VISIBLE_DEVICES=0
TEST=checkpoint-18000-3.27/whole_model.bin
MODEL_NAME=uclanlp/plbart-base  # roberta-base, microsoft/codebert-base, microsoft/graphcodebert-base
MODEL_NAME_ALIAS=${MODEL_NAME/'/'/-}
MODEL_TYPE=plbart
SAVED_PATH=../../result/bartrepre-1w
LANGUAGE=java
OUTPUT=result/java_codesum
TRAIN_FILE=./dataset/tlcodesum/data
EVAL_FILE=./dataset/tlcodesum/data
NODE_INDEX=0 && echo NODE_INDEX: ${NODE_INDEX}
PER_NODE_GPU=4 && echo PER_NODE_GPU: ${PER_NODE_GPU}
NUM_NODE=1 && echo NUM_NODE: ${NUM_NODE}
mkdir -p ${OUTPUT}
BLOCK_SIZE=300 # sentence length
TRAIN_BATCH_SIZE=32 #12 #32 # per gpu batch
EVAL_BATCH_SIZE=32 #12 #32
ACCUMULATE_STEPS=2 #6
LEARNING_RATE=2e-5
WEIGHT_DECAY=0.01
ADAM_EPS=1e-6
MAX_STEPS=10000
WARMUP_STEPS=1000 # 0.1 of max steps
SAVE_STEPS=2000  #
BEAM_SIZE=1
TEST_STEP=10000

CUDA_LAUNCH_BLOCKING=1 python run.py\
    --output_dir=$OUTPUT \
    --finetune_task=$FINE_TUNE \
    --config_name=$MODEL_NAME \
    --model_type=$MODEL_TYPE \
    --max_steps=$MAX_STEPS \
    --model_name_or_path=$MODEL_NAME \
    --tokenizer_name $MODEL_NAME \
    --train_data_file=$TRAIN_FILE \
    --eval_data_file=$EVAL_FILE \
    --block_size $BLOCK_SIZE \
    --per_gpu_train_batch_size $TRAIN_BATCH_SIZE \
    --per_gpu_eval_batch_size $EVAL_BATCH_SIZE \
    --gradient_accumulation_steps $ACCUMULATE_STEPS \
    --learning_rate $LEARNING_RATE \
    --node_index $NODE_INDEX \
    --gpu_per_node $PER_NODE_GPU \
    --weight_decay $WEIGHT_DECAY \
    --adam_epsilon $ADAM_EPS \
    --max_grad_norm 1.0 \
    --warmup_steps $WARMUP_STEPS \
    --save_steps $SAVE_STEPS \
    --seed 123456 \
    --lang $LANGUAGE \
    --beam_size $BEAM_SIZE