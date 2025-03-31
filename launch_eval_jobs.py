import argparse
import time
import subprocess

parser = argparse.ArgumentParser(description="Launch jobs")

start_layer = 0
end_layer = 8

for layer in range(start_layer, end_layer):
    script = f"""#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -p gh
#SBATCH -t 24:00:00
#SBATCH --mail-type=all
#SBATCH --mail-user=arvind.cganesh@gmail.com


GITHUB_REPO=$SCRATCH/early-exit-tests

module load tacc-apptainer
singularity exec -H $SCRATCH/tmp_home --nv $SCRATCH/pytorch_25.01-py3.sif \
python $GITHUB_REPO/evaluate_average_agreement_len.py \
--model_path meta-llama/Llama-3.2-1B \
--target_layer {layer} \
--softmax_temperature 0.0 \
--weights_to_load layer0_50000steps_begin1741241521/model_19000_5832.63.pt \
--sample_from_models \
--device cuda
exit
"""
    script_name = "tmp_eval_script"
    with open(script_name, "w+") as f:
        f.write(script)
    
    subprocess.run(["sbatch", script_name])
    time.sleep(1)
        

    
    
