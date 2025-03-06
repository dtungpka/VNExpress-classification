# VNExpress Text Classification Project

## Overview
This repository contains a comprehensive text classification system for Vietnamese news articles from VNExpress. The project implements various natural language processing models and feature extraction techniques to categorize Vietnamese news content.

## Features
- Multiple text classification models:
  - Transformer-based architecture
  - CNN
  - LSTM
  - BiLSTM
  - Hybrid models (CNN-LSTM)
  - SVM

- Various text feature extraction methods:
  - Word2Vec (300-dimensional embeddings)
  - TF-IDF
  - Bag-of-Words (BoW)
  - One-hot encoding

- Vietnamese language processing with:
  - Underthesea library integration
  - Vietnamese stopwords filtering

## Environment Setup
The project is configured to run on HPC environment with GPU acceleration:
- CUDA 11.5/11.8
- TensorFlow
- Gensim (for Word2Vec)
- NumPy
- Pandas
- Underthesea (Vietnamese NLP toolkit)

## Directory Structure
```
VNExpress-classification/
├── articles/              # Dataset directory
├── logs/                  # Training and execution logs
├── experiments/           # Experimental results
│   ├── models/            # Saved models
│   ├── features/          # Extracted features
│   └── combined/          # Combined model outputs
├── *.sh                   # Execution scripts for HPC
└── word2vec_vi_words_300dims_final.bin  # Pre-trained Word2Vec embeddings
```

## Usage
The project contains several scripts for running on HPC environments:

### Training
```bash
sbatch training-j.sh       # Full training pipeline with all models
sbatch training-trans-j.sh # Transformer-specific training
sbatch train-category-j.sh # Category-specific training
```

### Configuration
The training scripts support various configuration options:
- Model type (transformer, cnn, lstm, bilstm, hybrid, svm)
- Feature types (word2vec, tfidf, bow, one-hot)
- Hyperparameters (batch size, epochs, embedding dimensions, etc.)

### Example Commands
```bash
# Example for running a CNN model with Word2Vec features
python train.py --model_type cnn --filters 128 --kernel_size 5 --feature_type word2vec --embed_dim 300 --batch_size 32 --epochs 50
```

## Requirements
- Python 3.x
- CUDA compatible GPU
- TensorFlow
- Gensim
- NumPy
- Pandas
- Underthesea

## License
This project is released under the MIT License.