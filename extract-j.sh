#!/bin/bash
#SBATCH --job-name=SLD-EXTRACT
#SBATCH --account=ddt_acc23
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Make sure logs directory exists
mkdir -p logs

# Make sure temp_frames directory exists
mkdir -p temp_frames

# Load required modules
module load cuda/11.8.0
module load python

# Print environment information
module list
source /home/21010294/ComputerVision/CVEnv/bin/activate
which python
python --version
nvidia-smi

# Create output directory if it doesn't exist
mkdir -p /work/21010294/SLD/sld_feature_test

# Navigate to the directory with the script
cd /home/21010294/ComputerVision/SL_Dataset

# Run the feature extraction with corrected configuration
python create_sl_dataset.py -i /work/21010294/SLD/cropped -o /work/21010294/SLD/sld_feature \
  --checkpoint-file-rgb tsn_r50_320p_1x1x3_100e_kinetics400_rgb_20200702-cc665e2a.pth \
  --checkpoint-file-flow tsn_r50_320p_1x1x3_110e_kinetics400_flow_20200705-3036bab6.pth \
  --num-augmentations 50 \
  --device cuda:0 \
  --top-words-file /home/21010294/ComputerVision/SL_Dataset/top_words.txt \
  --num-top-words 20 \
  --word-to-sign-dict /home/21010294/ComputerVision/SL_Dataset/final_dict.pkl \
  --export-skeleton