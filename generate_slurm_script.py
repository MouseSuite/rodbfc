import os

def generate_slurm_script(channels, augmentation, lr, num_epochs, save_interval):
    script_content = f"""#!/usr/bin/bash
#SBATCH --job-name=training_{channels}_{augmentation}_{lr}
#SBATCH --output=training_{channels}_{augmentation}_{lr}.out
#SBATCH --error=training_{channels}_{augmentation}_{lr}.err
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=47:00:00
#SBATCH --account=ajoshi_27
#SBATCH --partition=gpu 

eval "$(conda shell.bash hook)"

module load gcc/11.3.0 python/3.9.12

cd /project/ajoshi_27/code_farm/rodbfc

ulimit -n 2880

echo "Checking Cuda, GPU USED?"
python -c 'import torch; print(torch.cuda.is_available()); print(torch.cuda.current_device()); print(torch.cuda.get_device_name(0))'
nvidia-smi

python main_training_modularized_param.py --channels {channels} --augmentation {augmentation} --lr {lr} --num_epochs {num_epochs} --save_interval {save_interval}
"""
    
    script_filename = f"slurm_script_channels_{channels}_aug_{augmentation}_lr_{lr}.sh"
    
    with open(script_filename, 'w') as file:
        file.write(script_content)

    return script_filename


# Define your parameter combinations
channels_list = ["16,64,64,128,256", "2,8,8,16,32"]
augmentation_list = [True, False]
lr_list = [1e-3, 1e-4]
num_epochs_list = [20002]
save_interval_list = [500]

# Generate Slurm scripts for each parameter combination
for channels in channels_list:
    for augmentation in augmentation_list:
        for lr in lr_list:
            for num_epochs in num_epochs_list:
                for save_interval in save_interval_list:
                    script_filename = generate_slurm_script(channels, augmentation, lr, num_epochs, save_interval)
                    
                    # Submit the Slurm script to the HPC queue
                    os.system(f"sbatch {script_filename}")
