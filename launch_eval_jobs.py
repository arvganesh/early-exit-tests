import argparse
import hashlib
import time
import subprocess

parser = argparse.ArgumentParser(description="Launch jobs")


def create_script(layer, temp):
    script = f"""#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -p gh-dev
#SBATCH -t 2:00:00
#SBATCH --mail-type=all
#SBATCH --mail-user=arvind.cganesh@gmail.com


GITHUB_REPO=$SCRATCH/early-exit-tests

module load tacc-apptainer
singularity exec -H $SCRATCH/tmp_home --nv $SCRATCH/pytorch_25.01-py3.sif \
python $GITHUB_REPO/evaluate_average_agreement_len.py \
--model_path meta-llama/Llama-3.2-1B \
--target_layer {layer} \
--softmax_temperature {temp} \
--weights_to_load layer0_50000steps_begin1741241521/model_19000_5832.63.pt \
--sample_from_models \
--device cuda \
--run_type head_and_last
exit
"""
    return script

def jobs_running():
    result = subprocess.run(
            ["squeue", "-u", "arvganesh"],
            stdout=subprocess.PIPE,
        )
    exp_hash = "9d6165647e3088be8f976972879b1c58f3191452a898bb7fe5d3ddcafa3c6b36"
    cur_hash = hashlib.sha256(result.stdout).hexdigest()
    return cur_hash != exp_hash

#temps = [0.0, 0.2, 1.0]
temps = [0.0]
for layer in range(1, 15):
    if layer == 10:
        continue
    for temp in temps:
        script = create_script(layer, temp)
        script_name = "tmp_eval_script"
        with open(script_name, "w+") as f:
            f.write(script)
        subprocess.run(["sbatch", script_name])
        while jobs_running():
            pass

    
    
