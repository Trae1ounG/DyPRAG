
model_name=qwen2.5-1.5b-instruct
python3 src/inference.py \
    --model_name=$model_name \
    --dataset=2wikimultihopqa \
    --sample=-1 \
    --num_train_epochs=1 \
    --learning_rate=0.0003 \
    --lora_rank=2 \
    --lora_alpha=32 \
    --max_new_tokens=128 \
    --inference_method=prag \
    --with_cot \
