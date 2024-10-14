#!/bin/bash

##NECESSARY JOB SPECIFICATIONS
#SBATCH --job-name=JointTrack     #Set the job name to "JobExample1"
#SBATCH --time=15:00:00            #Set the wall clock limit to 1hr and 30min
#SBATCH --ntasks=48                 #Request 1 task
#SBATCH --ntasks-per-node=48        #Request 1 task/core per node
#SBATCH --mem=80G                #Request 2560MB (2.5GB) per node
#SBATCH --output=./running_log/Example1Out.%j    #Send stdout/err to "Example1Out.[jobID]"
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=gpu
# SBATCH --nodes=1

#First Executable Line
# conda init
# conda activate joint
cd $SCRATCH/JointNLT
# python tracking/test.py jointnlt swin_b_ep300_track --dataset tnl2k --threads 16 --num_gpus 4 --params__model JointNLT_ep0300.pth.tar
python tracking/analysis_results.py --dataset_name tnl2k --tracker_param swin_b_ep300_track

cd $SCRATCH/JointNLT/leo_experiment
python viz.py
python experiment.py 