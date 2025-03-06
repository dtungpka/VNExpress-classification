#!/bin/bash
#SBATCH --job-name=VNE-TRANSFORMER
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
mkdir -p experiments/transformer_features
mkdir -p experiments/transformer_heads
mkdir -p experiments/transformer_ff_dims
mkdir -p experiments/transformer_dropouts
mkdir -p experiments/transformer_batch_sizes

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

# Data paths
DATA_DIR="articles"
WORD2VEC_PATH="word2vec_vi_words_300dims_final.bin"
STOPWORDS_PATH="vietnamese-stopwords.txt"

# Base parameters (default)
BASE_PARAMS="--model_type transformer --batch_size 1 --epochs 150 --max_len 1000 --min_articles 30"

# ===== EXPERIMENT 1: Different Feature Types =====
echo "=== Running Transformer with Different Features ==="

declare -A feature_configs=(
    ["word2vec"]="--feature_type word2vec --embed_dim 300"
    ["tfidf"]="--feature_type tfidf --max_features 20000"
    ["bow"]="--feature_type bow --max_features 20000"
    ["one-hot"]="--feature_type one-hot --max_features 20000"
)

for feature_name in "${!feature_configs[@]}"; do
    echo "=== Training transformer with feature: $feature_name ==="
    OUTPUT_DIR="experiments/transformer_features/${feature_name}"
    
    echo "Start time: $(date)"
    python attention.py \
        --data_dir "$DATA_DIR" \
        --word2vec_path "$WORD2VEC_PATH" \
        --stopwords_path "$STOPWORDS_PATH" \
        --output_dir "$OUTPUT_DIR" \
        --num_heads 4 \
        --ff_dim 128 \
        --dropout_rate 0.2 \
        ${feature_configs[$feature_name]} \
        $BASE_PARAMS \
        --experiment_id "transformer_${feature_name}" \
        2>&1 | tee "${OUTPUT_DIR}/transformer_${feature_name}.log"
    echo "End time: $(date)"
done

# ===== EXPERIMENT 2: Different Number of Attention Heads =====
echo "=== Running Transformer with Different Numbers of Attention Heads ==="

# Use best feature type from first experiment (assuming word2vec for now)
FEATURE_CONFIG=${feature_configs["word2vec"]}

for num_heads in 1 2 4 8 16; do
    echo "=== Training transformer with $num_heads attention heads ==="
    OUTPUT_DIR="experiments/transformer_heads/heads_${num_heads}"
    
    echo "Start time: $(date)"
    python attention.py \
        --data_dir "$DATA_DIR" \
        --word2vec_path "$WORD2VEC_PATH" \
        --stopwords_path "$STOPWORDS_PATH" \
        --output_dir "$OUTPUT_DIR" \
        --num_heads $num_heads \
        --ff_dim 128 \
        --dropout_rate 0.2 \
        $FEATURE_CONFIG \
        $BASE_PARAMS \
        --experiment_id "transformer_heads_${num_heads}" \
        2>&1 | tee "${OUTPUT_DIR}/transformer_heads_${num_heads}.log"
    echo "End time: $(date)"
done

# ===== EXPERIMENT 3: Different Feed-Forward Dimensions =====
echo "=== Running Transformer with Different Feed-Forward Dimensions ==="

for ff_dim in 64 128 256 512 1024; do
    echo "=== Training transformer with feed-forward dimension $ff_dim ==="
    OUTPUT_DIR="experiments/transformer_ff_dims/ff_${ff_dim}"
    
    echo "Start time: $(date)"
    python attention.py \
        --data_dir "$DATA_DIR" \
        --word2vec_path "$WORD2VEC_PATH" \
        --stopwords_path "$STOPWORDS_PATH" \
        --output_dir "$OUTPUT_DIR" \
        --num_heads 4 \
        --ff_dim $ff_dim \
        --dropout_rate 0.2 \
        $FEATURE_CONFIG \
        $BASE_PARAMS \
        --experiment_id "transformer_ff_${ff_dim}" \
        2>&1 | tee "${OUTPUT_DIR}/transformer_ff_${ff_dim}.log"
    echo "End time: $(date)"
done

# ===== EXPERIMENT 4: Different Dropout Rates =====
echo "=== Running Transformer with Different Dropout Rates ==="

for dropout in 0.1 0.2 0.3 0.4 0.5; do
    echo "=== Training transformer with dropout rate $dropout ==="
    OUTPUT_DIR="experiments/transformer_dropouts/dropout_${dropout}"
    
    echo "Start time: $(date)"
    python attention.py \
        --data_dir "$DATA_DIR" \
        --word2vec_path "$WORD2VEC_PATH" \
        --stopwords_path "$STOPWORDS_PATH" \
        --output_dir "$OUTPUT_DIR" \
        --num_heads 4 \
        --ff_dim 128 \
        --dropout_rate $dropout \
        $FEATURE_CONFIG \
        $BASE_PARAMS \
        --experiment_id "transformer_dropout_${dropout}" \
        2>&1 | tee "${OUTPUT_DIR}/transformer_dropout_${dropout}.log"
    echo "End time: $(date)"
done

# ===== EXPERIMENT 5: Different Batch Sizes =====
echo "=== Running Transformer with Different Batch Sizes ==="

for batch_size in 8 16 32 64; do
    echo "=== Training transformer with batch size $batch_size ==="
    OUTPUT_DIR="experiments/transformer_batch_sizes/batch_${batch_size}"
    
    echo "Start time: $(date)"
    python attention.py \
        --data_dir "$DATA_DIR" \
        --word2vec_path "$WORD2VEC_PATH" \
        --stopwords_path "$STOPWORDS_PATH" \
        --output_dir "$OUTPUT_DIR" \
        --num_heads 4 \
        --ff_dim 128 \
        --dropout_rate 0.2 \
        --batch_size $batch_size \
        $FEATURE_CONFIG \
        $BASE_PARAMS \
        --experiment_id "transformer_batch_${batch_size}" \
        2>&1 | tee "${OUTPUT_DIR}/transformer_batch_${batch_size}.log"
    echo "End time: $(date)"
done

# ===== EXPERIMENT 6: Best Configurations Combined =====
echo "=== Running Best Transformer Configuration ==="

# Note: Replace these values with the best from previous experiments
BEST_FEATURE="word2vec"  # Placeholder - replace after experiments
BEST_HEADS=4             # Placeholder - replace after experiments
BEST_FF_DIM=128          # Placeholder - replace after experiments
BEST_DROPOUT=0.2         # Placeholder - replace after experiments
BEST_BATCH=16            # Placeholder - replace after experiments

OUTPUT_DIR="experiments/transformer_best"
mkdir -p "$OUTPUT_DIR"

echo "=== Training transformer with best configuration ==="
echo "Start time: $(date)"
python attention.py \
    --data_dir "$DATA_DIR" \
    --word2vec_path "$WORD2VEC_PATH" \
    --stopwords_path "$STOPWORDS_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --feature_type $BEST_FEATURE \
    --embed_dim 300 \
    --num_heads $BEST_HEADS \
    --ff_dim $BEST_FF_DIM \
    --dropout_rate $BEST_DROPOUT \
    --batch_size $BEST_BATCH \
    --epochs 150 \
    --max_len 1000 \
    --min_articles 30 \
    --model_type transformer \
    --experiment_id "transformer_best_config" \
    2>&1 | tee "${OUTPUT_DIR}/transformer_best_config.log"
echo "End time: $(date)"

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
                
                # Get experiment category from directory structure
                experiment_path = os.path.relpath(root, base_dir)
                parts = experiment_path.split('/')
                
                if len(parts) >= 2:
                    experiment_category = parts[0]  # e.g., transformer_features
                    experiment_variant = parts[1]   # e.g., word2vec
                    
                    data['experiment_category'] = experiment_category
                    data['experiment_variant'] = experiment_variant
                
                results.append(data)
            except Exception as e:
                print(f"Error processing {os.path.join(root, 'summary.json')}: {e}")
    
    return results

# Collect results from all experiment directories
all_results = collect_results('experiments')

# Convert to DataFrame
df = pd.DataFrame(all_results)

# Save to CSV
df.to_csv('experiments/transformer_results.csv', index=False)
print(f"Saved summary of {len(df)} experiments to experiments/transformer_results.csv")

# Create visualizations by experiment type
plt.figure(figsize=(15, 20))

# 1. Feature comparison
plt.subplot(3, 2, 1)
feature_data = df[df['experiment_category'].str.contains('transformer_features', na=False)]
if not feature_data.empty:
    sns.barplot(x='feature_type', y='test_accuracy', data=feature_data, palette='viridis')
    plt.title('Transformer Performance by Feature Type', fontsize=14)
    plt.xlabel('Feature Type')
    plt.ylabel('Test Accuracy')
    plt.ylim(0, 1.0)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

# 2. Number of heads comparison
plt.subplot(3, 2, 2)
heads_data = df[df['experiment_category'].str.contains('transformer_heads', na=False)]
if not heads_data.empty:
    # Extract number of heads from experiment_variant
    heads_data['num_heads'] = heads_data['experiment_variant'].str.extract(r'heads_(\d+)').astype(int)
    sns.barplot(x='num_heads', y='test_accuracy', data=heads_data.sort_values('num_heads'), palette='viridis')
    plt.title('Transformer Performance by Number of Attention Heads', fontsize=14)
    plt.xlabel('Number of Heads')
    plt.ylabel('Test Accuracy')
    plt.ylim(0, 1.0)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

# 3. Feed-forward dimension comparison
plt.subplot(3, 2, 3)
ff_data = df[df['experiment_category'].str.contains('transformer_ff_dims', na=False)]
if not ff_data.empty:
    # Extract ff_dim from experiment_variant
    ff_data['ff_dim'] = ff_data['experiment_variant'].str.extract(r'ff_(\d+)').astype(int)
    sns.barplot(x='ff_dim', y='test_accuracy', data=ff_data.sort_values('ff_dim'), palette='viridis')
    plt.title('Transformer Performance by Feed-Forward Dimension', fontsize=14)
    plt.xlabel('Feed-Forward Dimension')
    plt.ylabel('Test Accuracy')
    plt.ylim(0, 1.0)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

# 4. Dropout rate comparison
plt.subplot(3, 2, 4)
dropout_data = df[df['experiment_category'].str.contains('transformer_dropouts', na=False)]
if not dropout_data.empty:
    # Extract dropout rate from experiment_variant
    dropout_data['dropout'] = dropout_data['experiment_variant'].str.extract(r'dropout_(0\.\d+)').astype(float)
    sns.barplot(x='dropout', y='test_accuracy', data=dropout_data.sort_values('dropout'), palette='viridis')
    plt.title('Transformer Performance by Dropout Rate', fontsize=14)
    plt.xlabel('Dropout Rate')
    plt.ylabel('Test Accuracy')
    plt.ylim(0, 1.0)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

# 5. Batch size comparison
plt.subplot(3, 2, 5)
batch_data = df[df['experiment_category'].str.contains('transformer_batch_sizes', na=False)]
if not batch_data.empty:
    # Extract batch size from experiment_variant
    batch_data['batch_size'] = batch_data['experiment_variant'].str.extract(r'batch_(\d+)').astype(int)
    sns.barplot(x='batch_size', y='test_accuracy', data=batch_data.sort_values('batch_size'), palette='viridis')
    plt.title('Transformer Performance by Batch Size', fontsize=14)
    plt.xlabel('Batch Size')
    plt.ylabel('Test Accuracy')
    plt.ylim(0, 1.0)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig('experiments/transformer_comparisons.png', dpi=300)

# Find best performing configuration
if not df.empty:
    best = df.loc[df['test_accuracy'].idxmax()]
    print(f"Best configuration: {best['experiment_id']}")
    print(f"Test accuracy: {best['test_accuracy']:.4f}, F1-score: {best['weighted_f1_score']:.4f}")

    # Generate HTML report
    html_report = f"""
    <html>
    <head>
        <title>Transformer Model Experiments for Vietnamese Author Classification</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            table {{ border-collapse: collapse; width: 100%; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            tr:nth-child(even) {{ background-color: #f2f2f2; }}
            th {{ background-color: #4CAF50; color: white; }}
            .best {{ background-color: #ffffcc; }}
            h1, h2 {{ color: #333; }}
            img {{ max-width: 100%; height: auto; }}
            .card {{ border: 1px solid #ddd; padding: 20px; margin-bottom: 20px; border-radius: 5px; }}
        </style>
    </head>
    <body>
        <h1>Transformer Model Experiments for Vietnamese Author Classification</h1>
        <p>Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}</p>
        
        <div class="card">
            <h2>Best Performing Configuration</h2>
            <p>Experiment ID: <strong>{best['experiment_id']}</strong></p>
            <p>Test accuracy: <strong>{best['test_accuracy']:.4f}</strong>, F1-score: <strong>{best['weighted_f1_score']:.4f}</strong></p>
        </div>
        
        <h2>Performance Comparisons</h2>
        <img src="transformer_comparisons.png" alt="Transformer performance comparisons">
        
        <h2>All Experiment Results</h2>
        <table>
            <tr>
                <th>Experiment ID</th>
                <th>Category</th>
                <th>Variant</th>
                <th>Feature Type</th>
                <th>Test Accuracy</th>
                <th>F1-Score</th>
                <th>Train Samples</th>
                <th>Test Samples</th>
            </tr>
    """

    # Add rows for each experiment
    for _, row in df.sort_values('test_accuracy', ascending=False).iterrows():
        best_class = " class='best'" if row['test_accuracy'] == best['test_accuracy'] else ""
        html_report += f"""
            <tr{best_class}>
                <td>{row.get('experiment_id', 'N/A')}</td>
                <td>{row.get('experiment_category', 'N/A')}</td>
                <td>{row.get('experiment_variant', 'N/A')}</td>
                <td>{row.get('feature_type', 'N/A')}</td>
                <td>{row.get('test_accuracy', 0):.4f}</td>
                <td>{row.get('weighted_f1_score', 0):.4f}</td>
                <td>{row.get('train_samples', 0)}</td>
                <td>{row.get('test_samples', 0)}</td>
            </tr>
    """

    html_report += """
        </table>
        
        <h2>Recommendations for Future Experiments</h2>
        <p>Based on these results, consider:</p>
        <ul>
            <li>Fine-tuning the best feature type with different preprocessing steps</li>
            <li>Experimenting with more attention layers or different model depths</li>
            <li>Trying learning rate schedules and different optimizers</li>
            <li>Adding regularization techniques beyond dropout</li>
        </ul>
    </body>
    </html>
    """

    with open('experiments/transformer_report.html', 'w') as f:
        f.write(html_report)

    print("Generated HTML report at experiments/transformer_report.html")
EOF

echo "=== All transformer experiments completed ==="
echo "Results summary saved to experiments/transformer_results.csv"
echo "Visual comparison saved to experiments/transformer_comparisons.png"
echo "HTML report generated at experiments/transformer_report.html"
echo "End time: $(date)"