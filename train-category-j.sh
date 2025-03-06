#!/bin/bash
#SBATCH --job-name=VNE-CATEGORY
#SBATCH --account=ddt_acc23
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --nodelist=hpc22

# Create necessary directories
if [ -d "category_experiments" ]; then
    # Get current date and time for backup naming
    TIMESTAMP=$(date +"%H_%M_%d_%m")
    echo "Category experiments directory already exists, backing up to category_experiments_before_${TIMESTAMP}"
    mv category_experiments category_experiments_before_${TIMESTAMP}
fi

mkdir -p logs
mkdir -p category_experiments
mkdir -p category_experiments/models
mkdir -p category_experiments/features
mkdir -p category_experiments/best

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
GENERAL_PARAMS="--dropout_rate 0.2 --batch_size 16 --epochs 50 --max_len 1000 --min_articles 30 --classification_task category"

# Data paths
DATA_DIR="articles"
WORD2VEC_PATH="word2vec_vi_words_300dims_final.bin"
STOPWORDS_PATH="vietnamese-stopwords.txt"

# ===== EXPERIMENT 1: Model Comparison =====
echo "=== Running Category Classification Model Comparison Experiments ==="

for model_name in "${!model_configs[@]}"; do
    echo "=== Training model: $model_name for category classification with word2vec features ==="
    OUTPUT_DIR="category_experiments/models/${model_name}"
    
    echo "Start time: $(date)"
    python attention_cate.py \
        --data_dir "$DATA_DIR" \
        --word2vec_path "$WORD2VEC_PATH" \
        --stopwords_path "$STOPWORDS_PATH" \
        --output_dir "$OUTPUT_DIR" \
        ${model_configs[$model_name]} \
        ${feature_configs["word2vec"]} \
        $GENERAL_PARAMS \
        --experiment_id "category_${model_name}_word2vec" \
        2>&1 | tee "${OUTPUT_DIR}/${model_name}_word2vec_category.log"
    echo "End time: $(date)"
done

# ===== EXPERIMENT 2: Feature Comparison =====
echo "=== Running Category Classification Feature Comparison Experiments ==="

for feature_name in "${!feature_configs[@]}"; do
    echo "=== Training transformer for category classification with feature: $feature_name ==="
    OUTPUT_DIR="category_experiments/features/${feature_name}"
    
    echo "Start time: $(date)"
    python attention_cate.py \
        --data_dir "$DATA_DIR" \
        --word2vec_path "$WORD2VEC_PATH" \
        --stopwords_path "$STOPWORDS_PATH" \
        --output_dir "$OUTPUT_DIR" \
        ${model_configs["transformer"]} \
        ${feature_configs[$feature_name]} \
        $GENERAL_PARAMS \
        --experiment_id "category_transformer_${feature_name}" \
        2>&1 | tee "${OUTPUT_DIR}/transformer_${feature_name}_category.log"
    echo "End time: $(date)"
done

# ===== EXPERIMENT 3: Selected Combinations =====
echo "=== Running Selected Category Classification Model-Feature Combinations ==="

# Best expected combinations based on previous experiments
COMBINATIONS=(
    "bilstm:word2vec"
    "cnn:tfidf"
    "hybrid:word2vec"
    "svm:tfidf"
)

for combo in "${COMBINATIONS[@]}"; do
    IFS=':' read -r model feature <<< "$combo"
    echo "=== Training $model with $feature features for category classification ==="
    OUTPUT_DIR="category_experiments/best/${model}_${feature}"
    
    echo "Start time: $(date)"
    python attention_cate.py \
        --data_dir "$DATA_DIR" \
        --word2vec_path "$WORD2VEC_PATH" \
        --stopwords_path "$STOPWORDS_PATH" \
        --output_dir "$OUTPUT_DIR" \
        ${model_configs[$model]} \
        ${feature_configs[$feature]} \
        $GENERAL_PARAMS \
        --experiment_id "category_${model}_${feature}" \
        2>&1 | tee "${OUTPUT_DIR}/${model}_${feature}_category.log"
    echo "End time: $(date)"
done

# ===== EXPERIMENT 4: Hyperparameter Tuning for Best Model =====
echo "=== Running Hyperparameter Tuning for the Best Model ==="

# We'll assume transformer with word2vec might be the best, but this can be adjusted
# based on initial results from Experiments 1-3
BEST_MODEL="transformer"
BEST_FEATURE="word2vec"
OUTPUT_DIR="category_experiments/tuning"
mkdir -p "$OUTPUT_DIR"

# Try different dropout rates
for dropout in 0.1 0.2 0.3 0.4; do
    echo "=== Training $BEST_MODEL with $BEST_FEATURE features, dropout=$dropout ==="
    
    echo "Start time: $(date)"
    python attention_cate.py \
        --data_dir "$DATA_DIR" \
        --word2vec_path "$WORD2VEC_PATH" \
        --stopwords_path "$STOPWORDS_PATH" \
        --output_dir "${OUTPUT_DIR}/dropout_${dropout}" \
        ${model_configs[$BEST_MODEL]} \
        ${feature_configs[$BEST_FEATURE]} \
        $GENERAL_PARAMS \
        --dropout_rate $dropout \
        --experiment_id "category_${BEST_MODEL}_${BEST_FEATURE}_dropout_${dropout}" \
        2>&1 | tee "${OUTPUT_DIR}/dropout_${dropout}.log"
    echo "End time: $(date)"
done

# Try different numbers of attention heads (for transformer model)
if [ "$BEST_MODEL" = "transformer" ]; then
    for heads in 2 4 8; do
        echo "=== Training transformer with $BEST_FEATURE features, heads=$heads ==="
        
        echo "Start time: $(date)"
        python attention_cate.py \
            --data_dir "$DATA_DIR" \
            --word2vec_path "$WORD2VEC_PATH" \
            --stopwords_path "$STOPWORDS_PATH" \
            --output_dir "${OUTPUT_DIR}/heads_${heads}" \
            --model_type transformer \
            --num_heads $heads \
            --ff_dim 128 \
            ${feature_configs[$BEST_FEATURE]} \
            $GENERAL_PARAMS \
            --experiment_id "category_transformer_${BEST_FEATURE}_heads_${heads}" \
            2>&1 | tee "${OUTPUT_DIR}/heads_${heads}.log"
        echo "End time: $(date)"
    done
fi

# Try different batch sizes
for batch_size in 8 16 32; do
    echo "=== Training $BEST_MODEL with $BEST_FEATURE features, batch_size=$batch_size ==="
    
    echo "Start time: $(date)"
    python attention_cate.py \
        --data_dir "$DATA_DIR" \
        --word2vec_path "$WORD2VEC_PATH" \
        --stopwords_path "$STOPWORDS_PATH" \
        --output_dir "${OUTPUT_DIR}/batch_${batch_size}" \
        ${model_configs[$BEST_MODEL]} \
        ${feature_configs[$BEST_FEATURE]} \
        $GENERAL_PARAMS \
        --batch_size $batch_size \
        --experiment_id "category_${BEST_MODEL}_${BEST_FEATURE}_batch_${batch_size}" \
        2>&1 | tee "${OUTPUT_DIR}/batch_${batch_size}.log"
    echo "End time: $(date)"
done

# ===== Generate Summary Report =====
echo "=== Generating Category Classification Results Summary ==="

python - <<EOF
import os
import json
import numpy as np
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
                
                # Extract experiment group from directory structure
                path_parts = root.split('/')
                if len(path_parts) >= 3:
                    data['experiment_group'] = path_parts[1]  # models, features, best, or tuning
                    data['experiment_name'] = path_parts[2]   # specific experiment name
                
                # Parse experiment_id to extract meaningful components
                if 'experiment_id' in data:
                    exp_id = data['experiment_id']
                    parts = exp_id.split('_')
                    if len(parts) >= 3:
                        # Extract parameters from experiment_id if available
                        for i, part in enumerate(parts[2:], 2):
                            if part in ['dropout', 'heads', 'batch'] and i+1 < len(parts):
                                data[part] = parts[i+1]
                
                results.append(data)
            except Exception as e:
                print(f"Error processing {os.path.join(root, 'summary.json')}: {e}")
    
    return results

# Collect results from all experiment directories
all_results = collect_results('category_experiments')

if not all_results:
    print("No results found. Make sure the experiments have completed successfully.")
    exit(0)

# Convert to DataFrame
df = pd.DataFrame(all_results)

# Clean up the DataFrame
if 'num_classes' in df.columns and 'num_authors' not in df.columns:
    df = df.rename(columns={'num_classes': 'num_categories'})
elif 'num_authors' in df.columns and 'num_categories' not in df.columns:
    df = df.rename(columns={'num_authors': 'num_categories'})

# Ensure numeric columns are numeric
numeric_cols = ['test_accuracy', 'weighted_f1_score', 'train_samples', 'test_samples']
for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

# Save to CSV
df.to_csv('category_experiments/category_results.csv', index=False)
print(f"Saved summary of {len(df)} experiments to category_experiments/category_results.csv")

# Create visualizations
plt.figure(figsize=(20, 15))

# 1. Model comparison (if we have model data)
plt.subplot(2, 2, 1)
if 'experiment_group' in df.columns and 'models' in df['experiment_group'].values:
    model_data = df[df['experiment_group'] == 'models']
    if not model_data.empty:
        sns.barplot(x='model_type', y='test_accuracy', data=model_data.sort_values('test_accuracy'), palette='viridis')
        plt.title('Category Classification: Model Comparison', fontsize=14)
        plt.xlabel('Model Type')
        plt.ylabel('Test Accuracy')
        plt.ylim(0, 1.0)
        plt.xticks(rotation=45)
        plt.grid(axis='y', linestyle='--', alpha=0.7)

# 2. Feature comparison (if we have feature data)
plt.subplot(2, 2, 2)
if 'experiment_group' in df.columns and 'features' in df['experiment_group'].values:
    feature_data = df[df['experiment_group'] == 'features']
    if not feature_data.empty:
        sns.barplot(x='feature_type', y='test_accuracy', data=feature_data.sort_values('test_accuracy'), palette='viridis')
        plt.title('Category Classification: Feature Comparison', fontsize=14)
        plt.xlabel('Feature Type')
        plt.ylabel('Test Accuracy')
        plt.ylim(0, 1.0)
        plt.grid(axis='y', linestyle='--', alpha=0.7)

# 3. F1 scores comparison
plt.subplot(2, 2, 3)
if 'weighted_f1_score' in df.columns and 'model_type' in df.columns:
    model_f1 = df.groupby('model_type')['weighted_f1_score'].mean().reset_index()
    sns.barplot(x='model_type', y='weighted_f1_score', data=model_f1.sort_values('weighted_f1_score'), palette='viridis')
    plt.title('Category Classification: F1 Score by Model Type', fontsize=14)
    plt.xlabel('Model Type')
    plt.ylabel('Weighted F1 Score')
    plt.ylim(0, 1.0)
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

# 4. Hyperparameter tuning results (if available)
plt.subplot(2, 2, 4)
if 'experiment_group' in df.columns and 'tuning' in df['experiment_group'].values:
    tuning_data = df[df['experiment_group'] == 'tuning']
    if 'dropout' in tuning_data.columns:
        dropout_data = tuning_data.dropna(subset=['dropout']).copy()
        if not dropout_data.empty:
            dropout_data['dropout'] = pd.to_numeric(dropout_data['dropout'])
            sns.lineplot(x='dropout', y='test_accuracy', data=dropout_data.sort_values('dropout'), marker='o')
            plt.title('Category Classification: Effect of Dropout Rate', fontsize=14)
            plt.xlabel('Dropout Rate')
            plt.ylabel('Test Accuracy')
            plt.grid(True, alpha=0.7)
    elif 'batch' in tuning_data.columns:
        batch_data = tuning_data.dropna(subset=['batch']).copy()
        if not batch_data.empty:
            batch_data['batch'] = pd.to_numeric(batch_data['batch'])
            sns.lineplot(x='batch', y='test_accuracy', data=batch_data.sort_values('batch'), marker='o')
            plt.title('Category Classification: Effect of Batch Size', fontsize=14)
            plt.xlabel('Batch Size')
            plt.ylabel('Test Accuracy')
            plt.grid(True, alpha=0.7)

plt.tight_layout()
plt.savefig('category_experiments/category_performance_comparison.png', dpi=300)

# Find best performing configuration
best = df.loc[df['test_accuracy'].idxmax()]
print(f"Best configuration for category classification:")
print(f"Model: {best.get('model_type', 'N/A')} with {best.get('feature_type', 'N/A')} features")
print(f"Test accuracy: {best.get('test_accuracy', 0):.4f}, F1-score: {best.get('weighted_f1_score', 0):.4f}")

# Generate HTML report
html_report = f"""
<html>
<head>
    <title>Vietnamese Article Category Classification Results</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        tr:nth-child(even) {{ background-color: #f2f2f2; }}
        th {{ background-color: #4CAF50; color: white; }}
        .best {{ background-color: #ffffcc; }}
        h1, h2 {{ color: #333; }}
        img {{ max-width: 100%; height: auto; }}
        .summary-box {{ 
            background-color: #f9f9f9;
            border: 1px solid #ddd;
            padding: 15px;
            margin-bottom: 20px;
            border-radius: 5px;
        }}
    </style>
</head>
<body>
    <h1>Vietnamese Article Category Classification Experiment Results</h1>
    <p>Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}</p>
    
    <div class="summary-box">
        <h2>Best Performing Configuration</h2>
        <p>Model: <strong>{best.get('model_type', 'N/A')}</strong> with <strong>{best.get('feature_type', 'N/A')}</strong> features</p>
        <p>Test accuracy: <strong>{best.get('test_accuracy', 0):.4f}</strong>, F1-score: <strong>{best.get('weighted_f1_score', 0):.4f}</strong></p>
        <p>Experiment ID: {best.get('experiment_id', 'N/A')}</p>
        <p>Number of categories: {best.get('num_categories', 'N/A')}</p>
        <p>Training samples: {best.get('train_samples', 'N/A')}, Test samples: {best.get('test_samples', 'N/A')}</p>
    </div>
    
    <h2>Performance Comparison</h2>
    <img src="category_performance_comparison.png" alt="Category classification performance comparison">
    
    <h2>All Experiment Results</h2>
    <table>
        <tr>
            <th>Experiment ID</th>
            <th>Model Type</th>
            <th>Feature Type</th>
            <th>Test Accuracy</th>
            <th>F1-Score</th>
            <th>Categories</th>
            <th>Train Samples</th>
            <th>Test Samples</th>
            <th>Group</th>
            <th>Name</th>
        </tr>
"""

# Add rows for each experiment
for _, row in df.sort_values('test_accuracy', ascending=False).iterrows():
    best_class = " class='best'" if row.get('test_accuracy') == best.get('test_accuracy') else ""
    html_report += f"""
        <tr{best_class}>
            <td>{row.get('experiment_id', 'N/A')}</td>
            <td>{row.get('model_type', 'N/A')}</td>
            <td>{row.get('feature_type', 'N/A')}</td>
            <td>{row.get('test_accuracy', 0):.4f}</td>
            <td>{row.get('weighted_f1_score', 0):.4f}</td>
            <td>{row.get('num_categories', 'N/A')}</td>
            <td>{row.get('train_samples', 0)}</td>
            <td>{row.get('test_samples', 0)}</td>
            <td>{row.get('experiment_group', 'N/A')}</td>
            <td>{row.get('experiment_name', 'N/A')}</td>
        </tr>
"""

html_report += """
    </table>
    
    <h2>Analysis and Recommendations</h2>
    <p>Based on these experiments, we can make the following observations:</p>
    <ul>
        <li>The best model for category classification appears to be a combination of model architecture and feature engineering.</li>
        <li>Hyperparameter tuning shows that optimal settings vary by model type.</li>
        <li>Future work should focus on refining the best configuration and possibly exploring ensemble methods.</li>
    </ul>
    
    <h2>Next Steps</h2>
    <ol>
        <li>Further optimize the best performing model</li>
        <li>Investigate error patterns on misclassified categories</li>
        <li>Consider hierarchical classification for related categories</li>
        <li>Explore transfer learning from pretrained Vietnamese language models</li>
    </ol>
</body>
</html>
"""

with open('category_experiments/category_results_report.html', 'w') as f:
    f.write(html_report)

print("Generated HTML report at category_experiments/category_results_report.html")
EOF

echo "=== All category classification experiments completed ==="
echo "Results summary saved to category_experiments/category_results.csv"
echo "Visual comparison saved to category_experiments/category_performance_comparison.png"
echo "HTML report generated at category_experiments/category_results_report.html"
echo "End time: $(date)"