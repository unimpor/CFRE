# wandb sweep --project RAG-cwq configs/cwq_tune.yaml
# wandb sweep --project RAG-webqsp configs/webqsp_tune.yaml


GPUs="1"
AGENTS_PER_GPU=1
# SWEEP_ID="g-com/RAG-cwq/8ujkkvpw"
SWEEP_ID="g-com/RAG-webqsp/7rprefsd"


for ((i=0; i<${#GPUs}; i++)); do
    gpu_id=${GPUs:$i:1}

    for ((agent=0; agent<$AGENTS_PER_GPU; agent++)); do
        echo "Launching agent $agent on GPU $gpu_id"
        CUDA_VISIBLE_DEVICES=$gpu_id wandb agent $SWEEP_ID &> /dev/null &
        pid=$!
        current_time=$(date "+%Y-%m-%d %H:%M:%S")
        echo "$current_time: Launched agent $agent (PID: $pid) on GPU $gpu_id" >> "./wandb/agents.log"
    done
done


echo "All agents launched."
