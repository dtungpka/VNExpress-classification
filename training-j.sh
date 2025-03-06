#!/bin/bash
#SBATCH --job-name=VNE-MODELS
#SBATCH --account=ddt_acc23
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --nodelist=hpc22

# Create necessary directories
if [ -d "experiments" ]; then
    # Get current date and time for backup naming
    TIMESTAMP=$(date +"%H_%M_%d_%m")
    echo "Experiments directory already exists, backing up to experiments_before_${TIMESTAMP}"
    mv experiments experiments_before_${TIMESTAMP}
fi

mkdir -p logs
mkdir -p experiments
mkdir -p experiments/models
mkdir -p experiments/features
mkdir -p experiments/combined


# Load required modules and environment
echo "=== Loading modules and environment ==="
module purge
module load cuda
module load python
source /home/21010294/VSR/VSREnv/bin/activate
export XLA_FLAGS="--xla_gpu_cuda_data_dir=/opt/hpc/cuda/11.5.2"


# Print environment information for reproducibility
echo "=== Environment Information ==="
module list
python -c "import sys; print('Python path:', sys.path)"
which python
python --version
python -c "import tensorflow as tf; print('TensorFlow version:', tf.__version__)"
python -c "import gensim; print('Gensim version:', gensim.__version__)"
python -c "import numpy as np; print('NumPy version:', np.__version__)"
python -c "import pandas as pd; print('Pandas version:', pd.__version__)"
python -c "from underthesea import __version__; print('Underthesea version:', __version__)"

# Verify GPU availability
echo "=== GPU Information ==="
python /home/21010294/VSR/cudacheck.py
nvidia-smi

# Navigate to working directory
cd /home/21010294/NLP/VNExpress

# Define model configurations
declare -A model_configs=(
    ["transformer"]="--model_type transformer --num_heads 4 --ff_dim 128"
    ["cnn"]="--model_type cnn --filters 128 --kernel_size 5"
    ["lstm"]="--model_type lstm --lstm_units 128"
    ["bilstm"]="--model_type bilstm --lstm_units 128"
    ["hybrid"]="--model_type hybrid --filters 128 --kernel_size 5 --lstm_units 128"
    ["svm"]="--model_type svm --svm_c 1.0"
)

# Define feature configurations
declare -A feature_configs=(
    ["word2vec"]="--feature_type word2vec --embed_dim 300"
    ["tfidf"]="--feature_type tfidf --max_features 20000"
    ["bow"]="--feature_type bow --max_features 20000"
    ["one-hot"]="--feature_type one-hot --max_features 20000"
)

# General parameters
GENERAL_PARAMS="--dropout_rate 0.2 --batch_size 16 --epochs 150 --max_len 1000 --min_articles 30"

# Data paths
DATA_DIR="articles"
WORD2VEC_PATH="word2vec_vi_words_300dims_final.bin"
STOPWORDS_PATH="vietnamese-stopwords.txt"

# Run experiments: Model comparison (using word2vec features)
echo "=== Running Model Comparison Experiments ==="
for model_name in "${!model_configs[@]}"; do
    echo "=== Training model: $model_name with word2vec features ==="
    OUTPUT_DIR="experiments/models/${model_name}"
    
    echo "Start time: $(date)"
    python attention.py \
        --data_dir "$DATA_DIR" \
        --word2vec_path "$WORD2VEC_PATH" \
        --stopwords_path "$STOPWORDS_PATH" \
        --output_dir "$OUTPUT_DIR" \
        ${model_configs[$model_name]} \
        ${feature_configs["word2vec"]} \
        $GENERAL_PARAMS \
        --experiment_id "${model_name}_vs_word2vec" \
        2>&1 | tee "${OUTPUT_DIR}/${model_name}_word2vec.log"
    echo "End time: $(date)"
done

# Run experiments: Feature comparison (using transformer model)
echo "=== Running Feature Comparison Experiments ==="
for feature_name in "${!feature_configs[@]}"; do
    echo "=== Training transformer with feature: $feature_name ==="
    OUTPUT_DIR="experiments/features/${feature_name}"
    
    echo "Start time: $(date)"
    python attention.py \
        --data_dir "$DATA_DIR" \
        --word2vec_path "$WORD2VEC_PATH" \
        --stopwords_path "$STOPWORDS_PATH" \
        --output_dir "$OUTPUT_DIR" \
        ${model_configs["transformer"]} \
        ${feature_configs[$feature_name]} \
        $GENERAL_PARAMS \
        --experiment_id "transformer_vs_${feature_name}" \
        2>&1 | tee "${OUTPUT_DIR}/transformer_${feature_name}.log"
    echo "End time: $(date)"
done

# Run specific model-feature combinations
echo "=== Running Selected Model-Feature Combinations ==="

# Best expected combinations
COMBINATIONS=(
    "bilstm:word2vec"
    "cnn:tfidf"
    "hybrid:word2vec"
    "svm:tfidf"
)

for combo in "${COMBINATIONS[@]}"; do
    IFS=':' read -r model feature <<< "$combo"
    echo "=== Training $model with $feature features ==="
    OUTPUT_DIR="experiments/combined/${model}_${feature}"
    
    echo "Start time: $(date)"
    python attention.py \
        --data_dir "$DATA_DIR" \
        --word2vec_path "$WORD2VEC_PATH" \
        --stopwords_path "$STOPWORDS_PATH" \
        --output_dir "$OUTPUT_DIR" \
        ${model_configs[$model]} \
        ${feature_configs[$feature]} \
        $GENERAL_PARAMS \
        --experiment_id "${model}_${feature}" \
        2>&1 | tee "${OUTPUT_DIR}/${model}_${feature}.log"
    echo "End time: $(date)"
done

# Generate summary of all experiments
echo "=== Generating Results Summary ==="

python - <<EOF
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Function to collect results from experiment directories
def collect_results(base_dir):
    results = []
    
    for root, dirs, files in os.walk(base_dir):
        if 'summary.json' in files:
            try:
                with open(os.path.join(root, 'summary.json'), 'r') as f:
                    data = json.load(f)
                
                # Extract experiment name from directory
                exp_name = os.path.basename(root)
                data['experiment'] = exp_name
                
                results.append(data)
            except:
                print(f"Error processing: {os.path.join(root, 'summary.json')}")
    
    return results

# Collect results from all experiment directories
all_results = collect_results('experiments')

# Convert to DataFrame
df = pd.DataFrame(all_results)

# Save to CSV
df.to_csv('experiments/all_results.csv', index=False)
print(f"Saved summary of {len(df)} experiments to experiments/all_results.csv")

# Create visualization of test accuracy by model and feature type
plt.figure(figsize=(15, 10))

# Model comparison
plt.subplot(2, 1, 1)
model_results = df.sort_values('test_accuracy', ascending=False)
sns.barplot(x='model_type', y='test_accuracy', hue='feature_type', data=model_results, palette='viridis')
plt.title('Test Accuracy by Model and Feature Type', fontsize=14)
plt.xlabel('Model Type')
plt.ylabel('Test Accuracy')
plt.ylim(0, 1.0)
plt.xticks(rotation=45)
plt.legend(title='Feature Type')
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Feature comparison
plt.subplot(2, 1, 2)
feature_results = df.sort_values('weighted_f1_score', ascending=False)
sns.barplot(x='feature_type', y='weighted_f1_score', hue='model_type', data=feature_results, palette='viridis')
plt.title('Weighted F1-Score by Feature Type and Model', fontsize=14)
plt.xlabel('Feature Type')
plt.ylabel('Weighted F1-Score')
plt.ylim(0, 1.0)
plt.xticks(rotation=45)
plt.legend(title='Model Type')
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig('experiments/performance_comparison.png', dpi=300)

# Find best performing model-feature combination
best = df.loc[df['test_accuracy'].idxmax()]
print(f"Best model: {best['model_type']} with {best['feature_type']} features")
print(f"Test accuracy: {best['test_accuracy']:.4f}, F1-score: {best['weighted_f1_score']:.4f}")

# Generate HTML report
html_report = f"""
<html>
<head>
    <title>Vietnamese Author Classification Experiment Results</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        tr:nth-child(even) {{ background-color: #f2f2f2; }}
        th {{ background-color: #4CAF50; color: white; }}
        .best {{ background-color: #ffffcc; }}
        h1, h2 {{ color: #333; }}
        img {{ max-width: 100%; height: auto; }}
    </style>
</head>
<body>
    <h1>Vietnamese Author Classification Experiment Results</h1>
    <p>Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}</p>
    
    <h2>Best Performing Configuration</h2>
    <p>Model: <strong>{best['model_type']}</strong> with <strong>{best['feature_type']}</strong> features</p>
    <p>Test accuracy: <strong>{best['test_accuracy']:.4f}</strong>, F1-score: <strong>{best['weighted_f1_score']:.4f}</strong></p>
    
    <h2>Performance Comparison</h2>
    <img src="performance_comparison.png" alt="Performance comparison chart">
    
    <h2>All Experiment Results</h2>
    <table>
        <tr>
            <th>Experiment</th>
            <th>Model Type</th>
            <th>Feature Type</th>
            <th>Test Accuracy</th>
            <th>F1-Score</th>
            <th>Classes</th>
            <th>Authors</th>
            <th>Train Samples</th>
            <th>Test Samples</th>
        </tr>
"""

# Add rows for each experiment
for _, row in df.sort_values('test_accuracy', ascending=False).iterrows():
    best_class = " class='best'" if row['test_accuracy'] == best['test_accuracy'] else ""
    html_report += f"""
        <tr{best_class}>
            <td>{row['experiment']}</td>
            <td>{row['model_type']}</td>
            <td>{row['feature_type']}</td>
            <td>{row['test_accuracy']:.4f}</td>
            <td>{row['weighted_f1_score']:.4f}</td>
            <td>{row['num_classes']}</td>
            <td>{row['num_authors']}</td>
            <td>{row['train_samples']}</td>
            <td>{row['test_samples']}</td>
        </tr>
"""

html_report += """
    </table>
</body>
</html>
"""

with open('experiments/results_report.html', 'w') as f:
    f.write(html_report)

print("Generated HTML report at experiments/results_report.html")
EOF

echo "=== All experiments completed ==="
echo "Results summary saved to experiments/all_results.csv"
echo "Visual comparison saved to experiments/performance_comparison.png"
echo "HTML report generated at experiments/results_report.html"
echo "End time: $(date)"