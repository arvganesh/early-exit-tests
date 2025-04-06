import argparse
import time
import subprocess

parser = argparse.ArgumentParser(description="Launch jobs")
parser.add_argument(
    "--print_cmd",
    action="store_true",
)
args = parser.parse_args()

start_layer = 14
end_layer = 15

for layer in range(start_layer, end_layer):
    python_cmd = f"""python train.py \
--model_path meta-llama/Llama-3.2-1B \
--target_layer {layer} \
--seed 123141 \
--loss_type kl_divergence \
--batch_size 4 \
--grad_accumulate_steps 8 \
--max_steps 50000 \
--learning_rate 0.00002 \
--warmup_step_ratio 0.1 \
--max_length 4096 \
--device cuda \
--output_dir $SCRATCH/models/ \
--notes "50k Steps, planning to train more later" \
--wandb \
--run_type just_head"""

    layer_script = f"""#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -p gh
#SBATCH -t 24:00:00
#SBATCH --mail-type=all
#SBATCH --mail-user=arvind.cganesh@gmail.com


GITHUB_REPO=$SCRATCH/early-exit-tests

module load tacc-apptainer
singularity exec -H $SCRATCH/tmp_home --nv $SCRATCH/pytorch_25.01-py3.sif \
{python_cmd}
exit"""

    if args.print_cmd:
        print(python_cmd)
        break
    else:
        script_name = "tmp_job_script"
        with open(script_name, "w+") as f:
            f.write(layer_script)
        
        subprocess.run(["sbatch", script_name])
    time.sleep(1)
    

    
    
