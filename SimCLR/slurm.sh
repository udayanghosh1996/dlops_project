#!/bin/bash
#SBATCH --job-name=simCLR 	# Job name
#SBATCH --partition=gpu2 	#Partition name can be test/small/medium/large/gpu #Partition “gpu” should be used only for gpu jobs
#SBATCH --nodes=1 			# Run all processes on a single node
#SBATCH --ntasks=1 			# Run a single task
#SBATCH --cpus-per-task=4 	# Number of CPU cores per task
#SBATCH --gres=gpu:1 		# Include gpu for the task (only for GPU jobs)


module load python/3.8
cd ~/dlops_proj/dlops_project_
python3 run.py
