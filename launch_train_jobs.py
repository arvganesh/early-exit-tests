import argparse
import time
import subprocess

parser = argparse.ArgumentParser(description="Launch jobs")

start_layer = 7
end_layer = 15


#--ft_last_transformer \

for layer in range(start_layer, end_layer):
    layer_script = f"""#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -p gh
#SBATCH -t 36:00:00
#SBATCH --mail-type=all
#SBATCH --mail-user=arvind.cganesh@gmail.com


GITHUB_REPO=$SCRATCH/early-exit-tests

module load tacc-apptainer
singularity exec -H $SCRATCH/tmp_home --nv $SCRATCH/pytorch_25.01-py3.sif \
python train.py \
--model_path meta-llama/Llama-3.2-1B \
--target_layer {layer} \
--seed 0 \
--loss_type kl_divergence \
--batch_size 4 \
--grad_accumulate_steps 8 \
--max_steps 100000 \
--learning_rate 0.00002 \
--warmup_step_ratio 0.1 \
--max_length 4096 \
--device cuda \
--output_dir $SCRATCH/models/ \
--notes "100k Steps" \
--wandb
exit
    """
    script_name = "tmp_job_script"
    with open(script_name, "w+") as f:
        f.write(layer_script)
    
    subprocess.run(["sbatch", script_name])
    time.sleep(1)
    

    
    
