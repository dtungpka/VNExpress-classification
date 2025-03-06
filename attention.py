#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
os.environ['XLA_FLAGS'] = '--xla_gpu_cuda_data_dir=/opt/hpc/cuda/11.5.2'
import re
import json
import numpy as np
import pandas as pd
import requests
import matplotlib.pyplot as plt
import gensim
import joblib  # Add this direct import
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.layers import (
    Embedding, GlobalAveragePooling1D, Dense, Input, Conv1D, MaxPooling1D,
    Dropout, LayerNormalization, MultiHeadAttention, LSTM, Bidirectional,
    Concatenate
)
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from underthesea import word_tokenize

# Define argument parser
def parse_args():
    parser = argparse.ArgumentParser(description='Author classification with multiple model architectures')
    parser.add_argument('--data_dir', type=str, default='articles', 
                        help='Directory containing article JSON files')
    parser.add_argument('--word2vec_path', type=str, default='word2vec_vi_words_300dims_final.bin',
                        help='Path to Word2Vec model file')
    parser.add_argument('--stopwords_path', type=str, default='vietnamese-stopwords.txt',
                        help='Path to stopwords file')
    parser.add_argument('--max_len', type=int, default=1000,
                        help='Maximum sequence length')
    parser.add_argument('--min_articles', type=int, default=30,
                        help='Minimum number of articles per author')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Training batch size')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Maximum number of training epochs')
    parser.add_argument('--embed_dim', type=int, default=300,
                        help='Embedding dimension size')
    parser.add_argument('--num_heads', type=int, default=4,
                        help='Number of attention heads for transformer')
    parser.add_argument('--ff_dim', type=int, default=128,
                        help='Feed forward dimension')
    parser.add_argument('--dropout_rate', type=float, default=0.2,
                        help='Dropout rate')
    parser.add_argument('--output_dir', type=str, default='output',
                        help='Directory for saving outputs')
    parser.add_argument('--model_type', type=str, default='transformer',
                        choices=['transformer', 'cnn', 'lstm', 'bilstm', 'svm', 'hybrid'],
                        help='Model architecture to use')
    parser.add_argument('--feature_type', type=str, default='word2vec',
                        choices=['word2vec', 'tfidf', 'bow', 'fasttext', 'one-hot'],
                        help='Type of text features to use')
    
    # Model-specific parameters
    parser.add_argument('--filters', type=int, default=128,
                        help='Number of filters for CNN')
    parser.add_argument('--kernel_size', type=int, default=5,
                        help='Kernel size for CNN')
    parser.add_argument('--lstm_units', type=int, default=128,
                        help='Number of units for LSTM')
    parser.add_argument('--svm_c', type=float, default=1.0,
                        help='Regularization parameter for SVM')
    parser.add_argument('--max_features', type=int, default=20000,
                        help='Max number of features for TF-IDF or BOW')
    
    # Experiment ID for better organization
    parser.add_argument('--experiment_id', type=str, default=None,
                        help='Unique identifier for experiment')
    
    return parser.parse_args()

# Download stopwords if not available
def get_stopwords(stopwords_path):
    stop_words_url = 'https://raw.githubusercontent.com/stopwords/vietnamese-stopwords/refs/heads/master/vietnamese-stopwords.txt'
    if not os.path.exists(stopwords_path):
        print(f"Downloading stopwords to {stopwords_path}")
        r = requests.get(stop_words_url)
        os.makedirs(os.path.dirname(stopwords_path), exist_ok=True)
        with open(stopwords_path, 'wb') as f:
            f.write(r.content)
    
    stop_words = []
    with open(stopwords_path, 'r', encoding='utf-8') as f:
        stop_words = f.read().split('\n')
    print(f'Number of stop words: {len(stop_words)}')
    return stop_words

# Text cleaning functions
def clean_text(text):
    text = re.sub(r'[^\w\s]', '', text)  # Remove special characters
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    return text

def replace_all(text):
    dict_map = {
        "òa": "oà", "Òa": "Oà", "ÒA": "OÀ",
        "óa": "oá", "Óa": "Oá", "ÓA": "OÁ",
        "ỏa": "oả", "Ỏa": "Oả", "ỎA": "OẢ",
        "õa": "oã", "Õa": "Oã", "ÕA": "OÃ",
        "ọa": "oạ", "Ọa": "Oạ", "ỌA": "OẠ",
        "òe": "oè", "Òe": "Oè", "ÒE": "OÈ",
        "óe": "oé", "Óe": "Oé", "ÓE": "OÉ",
        "ỏe": "oẻ", "Ỏe": "Oẻ", "ỎE": "OẺ",
        "õe": "oẽ", "Õe": "Oẽ", "ÕE": "OẼ",
        "ọe": "oẹ", "Ọe": "Oẹ", "ỌE": "OẸ",
        "ùy": "uỳ", "Ùy": "Uỳ", "ÙY": "UỲ",
        "úy": "uý", "Úy": "Uý", "ÚY": "UÝ",
        "ủy": "uỷ", "Ủy": "Uỷ", "ỦY": "UỶ",
        "ũy": "uỹ", "Ũy": "Uỹ", "ŨY": "UỸ",
        "ụy": "uỵ", "Ụy": "Uỵ", "ỤY": "UỴ",
    }
    for i, j in dict_map.items():
        text = text.replace(i, j)
    return text

# Load data from JSON files
def get_json_data(folder_path):
    articles_data = []
    articles = os.listdir(folder_path)
    for article in tqdm(articles, desc="Loading data"):
        with open(os.path.join(folder_path, article), "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except:
                print(f"Error loading JSON file: {article}")
                continue
           
            title = data.get("title", None)
            author = data.get("author_name", None)
            category = data.get("category", None)
            posted_date = data.get("posted_date", None)
            content = data.get("content", "")
            content = content.strip().replace("\n", " ")
            
            content = clean_text(content)
            if content == "":
                continue
            articles_data.append([title, author, category, posted_date, content])
    
    # Convert to DataFrame
    articles_data = pd.DataFrame(articles_data, columns=["title", "author", "category", "posted_date", "content"])
    return articles_data

# Custom Transformer model components
class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1, **kwargs):
        super().__init__(**kwargs)  # This properly handles the 'name' parameter
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.rate = rate
        self.att = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential(
            [Dense(ff_dim, activation="relu"), Dense(embed_dim),]
        )
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)

    def call(self, inputs, training=False):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)
    
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "ff_dim": self.ff_dim,
            "rate": self.rate
        })
        return config

# Feature extraction methods
def extract_word2vec_features(texts, w2v_model, max_len, embed_dim):
    """Convert texts to Word2Vec vector matrices."""
    matrices = []
    for text in tqdm(texts, desc="Extracting Word2Vec features"):
        words = text.split()
        matrix = np.zeros((max_len, embed_dim))
        for i, word in enumerate(words[:max_len]):
            if word in w2v_model:
                matrix[i] = w2v_model[word]
        matrices.append(matrix)
    return np.array(matrices)

def extract_tfidf_features(train_texts, val_texts=None, test_texts=None, max_features=20000):
    """Extract TF-IDF features from texts."""
    vectorizer = TfidfVectorizer(max_features=max_features)
    train_features = vectorizer.fit_transform(train_texts)
    
    result = {"train": train_features, "vectorizer": vectorizer}
    
    if val_texts is not None:
        val_features = vectorizer.transform(val_texts)
        result["val"] = val_features
    
    if test_texts is not None:
        test_features = vectorizer.transform(test_texts)
        result["test"] = test_features
    
    return result

def extract_bow_features(train_texts, val_texts=None, test_texts=None, max_features=20000):
    """Extract Bag-of-Words features from texts."""
    vectorizer = CountVectorizer(max_features=max_features)
    train_features = vectorizer.fit_transform(train_texts)
    
    result = {"train": train_features, "vectorizer": vectorizer}
    
    if val_texts is not None:
        val_features = vectorizer.transform(val_texts)
        result["val"] = val_features
    
    if test_texts is not None:
        test_features = vectorizer.transform(test_texts)
        result["test"] = test_features
    
    return result

def extract_one_hot_features(train_texts, val_texts=None, test_texts=None, max_len=1000, vocab_size=20000):
    """Extract one-hot encoding features from texts."""
    tokenizer = Tokenizer(num_words=vocab_size)
    tokenizer.fit_on_texts(train_texts)
    
    train_sequences = tokenizer.texts_to_sequences(train_texts)
    train_features = pad_sequences(train_sequences, maxlen=max_len)
    
    result = {"train": train_features, "tokenizer": tokenizer}
    
    if val_texts is not None:
        val_sequences = tokenizer.texts_to_sequences(val_texts)
        val_features = pad_sequences(val_sequences, maxlen=max_len)
        result["val"] = val_features
    
    if test_texts is not None:
        test_sequences = tokenizer.texts_to_sequences(test_texts)
        test_features = pad_sequences(test_sequences, maxlen=max_len)
        result["test"] = test_features
    
    return result

# Add this function after your existing feature extraction functions
def prepare_features_for_model(features, model_type):
    """Prepare features in the correct format based on the model type."""
    if model_type == 'svm':
        # For SVM, if features are 3D, convert to 2D by taking the mean across the sequence dimension
        if len(features.shape) == 3:
            return np.mean(features, axis=1)  # Average word vectors for each document
        return features
    else:
        # For neural network models, ensure proper dimension
        return features

# Model building functions
def build_transformer_model(input_shape, num_classes, args):
    """Build a transformer model with attention."""
    inputs = Input(shape=input_shape)
    
    # Add transformer block with attention
    transformer_block = TransformerBlock(
        embed_dim=args.embed_dim, 
        num_heads=args.num_heads, 
        ff_dim=args.ff_dim,
        rate=args.dropout_rate
    )
    x = transformer_block(inputs)
    
    # Global pooling and classification layers
    x = GlobalAveragePooling1D()(x)
    x = Dropout(args.dropout_rate)(x)
    x = Dense(64, activation="relu")(x)
    x = Dropout(args.dropout_rate)(x)
    outputs = Dense(num_classes, activation="softmax")(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

def build_cnn_model(input_shape, num_classes, args):
    """Build a CNN model."""
    inputs = Input(shape=input_shape)
    
    # CNN layers
    x = Conv1D(args.filters, args.kernel_size, activation="relu")(inputs)
    x = MaxPooling1D(pool_size=2)(x)
    x = Conv1D(args.filters * 2, args.kernel_size, activation="relu")(x)
    x = MaxPooling1D(pool_size=2)(x)
    
    # Global pooling and classification layers
    x = GlobalAveragePooling1D()(x)
    x = Dropout(args.dropout_rate)(x)
    x = Dense(64, activation="relu")(x)
    x = Dropout(args.dropout_rate)(x)
    outputs = Dense(num_classes, activation="softmax")(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

def build_lstm_model(input_shape, num_classes, args):
    """Build an LSTM model."""
    inputs = Input(shape=input_shape)
    
    # LSTM layers
    x = LSTM(args.lstm_units, return_sequences=True)(inputs)
    x = LSTM(args.lstm_units)(x)
    
    # Classification layers
    x = Dropout(args.dropout_rate)(x)
    x = Dense(64, activation="relu")(x)
    x = Dropout(args.dropout_rate)(x)
    outputs = Dense(num_classes, activation="softmax")(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

def build_bilstm_model(input_shape, num_classes, args):
    """Build a Bidirectional LSTM model."""
    inputs = Input(shape=input_shape)
    
    # BiLSTM layers
    x = Bidirectional(LSTM(args.lstm_units, return_sequences=True))(inputs)
    x = Bidirectional(LSTM(args.lstm_units))(x)
    
    # Classification layers
    x = Dropout(args.dropout_rate)(x)
    x = Dense(64, activation="relu")(x)
    x = Dropout(args.dropout_rate)(x)
    outputs = Dense(num_classes, activation="softmax")(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

def build_hybrid_model(input_shape, num_classes, args):
    """Build a hybrid model combining CNN and BiLSTM."""
    inputs = Input(shape=input_shape)
    
    # CNN branch
    conv1 = Conv1D(args.filters, args.kernel_size, activation="relu")(inputs)
    pool1 = MaxPooling1D(pool_size=2)(conv1)
    
    # BiLSTM branch
    bilstm = Bidirectional(LSTM(args.lstm_units, return_sequences=True))(inputs)
    bilstm = Bidirectional(LSTM(args.lstm_units // 2))(bilstm)
    
    # Merge branches
    concat = Concatenate()([GlobalAveragePooling1D()(pool1), bilstm])
    
    # Classification layers
    x = Dropout(args.dropout_rate)(concat)
    x = Dense(64, activation="relu")(x)
    x = Dropout(args.dropout_rate)(x)
    outputs = Dense(num_classes, activation="softmax")(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

def build_svm_model(args):
    """Build an SVM model with TF-IDF features."""
    return SVC(C=args.svm_c, kernel='linear', probability=True)
def load_transformer_model(model_path):
    """Load saved transformer model with custom objects."""
    with keras.utils.custom_object_scope({'TransformerBlock': TransformerBlock}):
        model = keras.models.load_model(model_path)
    return model
# Define main function
def main():
    # Parse arguments
    args = parse_args()
    
    # Create experiment identifier if not provided
    if args.experiment_id is None:
        args.experiment_id = f"{args.model_type}_{args.feature_type}"
    
    # Create output directory
    output_dir = os.path.join(args.output_dir, args.experiment_id)
    os.makedirs(output_dir, exist_ok=True)
    
    # Save configuration
    config = vars(args)
    with open(os.path.join(output_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=4)
    
    # Get stopwords
    stop_words = get_stopwords(args.stopwords_path)
    
    # Define stopwords removal function
    def remove_stopwords(text):
        return ' '.join([word for word in text.split() if word not in stop_words])
    
    # Get dataset
    print("Loading and processing article data...")
    sen_data = get_json_data(args.data_dir)
    
    # Define tokenization function
    def tokenize(text):
        return word_tokenize(text, format="text")
    
    # Apply tokenization and cleaning
    tqdm.pandas(desc="Tokenizing")
    sen_data['tokenized_text'] = (
        sen_data['content']
        .progress_apply(tokenize)
        .apply(remove_stopwords)
        .apply(replace_all)
    )
    
    # Filter authors with sufficient articles
    X = sen_data['tokenized_text']
    y = sen_data['author']
    author_counts = y.value_counts()
    authors = author_counts[author_counts >= args.min_articles].index
    sen_data_filtered = sen_data[sen_data['author'].isin(authors)]
    X = sen_data_filtered['tokenized_text']
    y = sen_data_filtered['author']
    print(f"Number of authors with at least {args.min_articles} articles: {len(authors)}")
    
    # Split data into train, dev, test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    print(f"Train size: {len(X_train)}")
    print(f"Validation size: {len(X_val)}")
    print(f"Test size: {len(X_test)}")
    
    # Create label mapping
    label_map = {label: i for i, label in enumerate(y.unique())}
    with open(os.path.join(output_dir, 'label_map.json'), 'w') as f:
        json.dump(label_map, f, indent=4)
    print(f"Label map: {label_map}")
    
    # Convert labels to one-hot encoding for deep learning models
    y_train_enc = keras.utils.to_categorical([label_map[label] for label in y_train], num_classes=len(label_map))
    y_val_enc = keras.utils.to_categorical([label_map[label] for label in y_val], num_classes=len(label_map))
    y_test_enc = keras.utils.to_categorical([label_map[label] for label in y_test], num_classes=len(label_map))
    
    # Feature extraction based on selected feature type
    print(f"Extracting {args.feature_type} features...")
    
    if args.feature_type == 'word2vec':
        # Load Word2Vec model
        print("Loading Word2Vec model...")
        w2v_model = gensim.models.KeyedVectors.load_word2vec_format(args.word2vec_path, binary=True)
        embedding_dim = w2v_model.vector_size
        print(f"Word2Vec embedding dimension: {embedding_dim}")
        
        # Extract Word2Vec features
        X_train_features = extract_word2vec_features(X_train, w2v_model, args.max_len, embedding_dim)
        X_val_features = extract_word2vec_features(X_val, w2v_model, args.max_len, embedding_dim)
        X_test_features = extract_word2vec_features(X_test, w2v_model, args.max_len, embedding_dim)
        
        input_shape = (args.max_len, embedding_dim)
        
    elif args.feature_type == 'tfidf':
        # Extract TF-IDF features
        features = extract_tfidf_features(
            X_train, X_val, X_test, 
            max_features=args.max_features
        )
        X_train_features = features['train']
        X_val_features = features['val']
        X_test_features = features['test']
        
        # Save vectorizer
        joblib.dump(features['vectorizer'], os.path.join(output_dir, 'tfidf_vectorizer.joblib'))
        
        # For neural networks, convert sparse matrices to dense
        if args.model_type != 'svm':
            X_train_features = X_train_features.toarray()
            X_val_features = X_val_features.toarray()
            X_test_features = X_test_features.toarray()
            input_shape = (X_train_features.shape[1],)
            # Add a dimension for 1D convolution
            if args.model_type in ['cnn', 'lstm', 'bilstm', 'transformer', 'hybrid']:
                X_train_features = np.expand_dims(X_train_features, axis=2)
                X_val_features = np.expand_dims(X_val_features, axis=2)
                X_test_features = np.expand_dims(X_test_features, axis=2)
                input_shape = (X_train_features.shape[1], 1)
        
    elif args.feature_type == 'one-hot':
        # Extract one-hot encoding features
        features = extract_one_hot_features(
            X_train, X_val, X_test, 
            max_len=args.max_len,
            vocab_size=args.max_features
        )
        X_train_features = features['train']
        X_val_features = features['val']
        X_test_features = features['test']
        
        # Save tokenizer
        with open(os.path.join(output_dir, 'tokenizer.json'), 'w') as f:
            f.write(features['tokenizer'].to_json())
        
        # For neural networks, create embedding layer
        if args.model_type != 'svm':
            vocab_size = min(args.max_features, len(features['tokenizer'].word_index) + 1)
            embedding_layer = Embedding(
                vocab_size,
                args.embed_dim,
                input_length=args.max_len
            )
            input_shape = (args.max_len,)
        
    else:  # 'bow' or other
        # Use BOW as default
        from sklearn.feature_extraction.text import CountVectorizer
        features = extract_bow_features(
            X_train, X_val, X_test, 
            max_features=args.max_features
        )
        X_train_features = features['train']
        X_val_features = features['val']
        X_test_features = features['test']
        
        # Save vectorizer
        joblib.dump(features['vectorizer'], os.path.join(output_dir, 'bow_vectorizer.joblib'))
        
        # For neural networks, convert sparse matrices to dense
        if args.model_type != 'svm':
            X_train_features = X_train_features.toarray()
            X_val_features = X_val_features.toarray()
            X_test_features = X_test_features.toarray()
            input_shape = (X_train_features.shape[1],)
            # Add a dimension for 1D convolution
            if args.model_type in ['cnn', 'lstm', 'bilstm', 'transformer', 'hybrid']:
                X_train_features = np.expand_dims(X_train_features, axis=2)
                X_val_features = np.expand_dims(X_val_features, axis=2)
                X_test_features = np.expand_dims(X_test_features, axis=2)
                input_shape = (X_train_features.shape[1], 1)
    
    print(f"Train features shape: {X_train_features.shape}")
    print(f"Validation features shape: {X_val_features.shape}")
    print(f"Test features shape: {X_test_features.shape}")
    
    # Prepare features for the model
    X_train_features = prepare_features_for_model(X_train_features, args.model_type)
    X_val_features = prepare_features_for_model(X_val_features, args.model_type)
    X_test_features = prepare_features_for_model(X_test_features, args.model_type)
    
    # Model building based on selected model type
    print(f"Building {args.model_type} model...")
    
    if args.model_type == 'transformer':
        model = build_transformer_model(input_shape, len(label_map), args)
    elif args.model_type == 'cnn':
        model = build_cnn_model(input_shape, len(label_map), args)
    elif args.model_type == 'lstm':
        model = build_lstm_model(input_shape, len(label_map), args)
    elif args.model_type == 'bilstm':
        model = build_bilstm_model(input_shape, len(label_map), args)
    elif args.model_type == 'hybrid':
        model = build_hybrid_model(input_shape, len(label_map), args)
    elif args.model_type == 'svm':
        model = build_svm_model(args)
    else:
        raise ValueError(f"Unsupported model type: {args.model_type}")
    
    # For deep learning models
    if args.model_type != 'svm':
        # Display model summary
        model.summary()
        
        # Set up callbacks
        checkpoint_cb = ModelCheckpoint(
            os.path.join(output_dir, 'best_model.h5'), 
            save_best_only=True,
            monitor='val_accuracy',
            mode='max'
        )
        
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        tensorboard_cb = TensorBoard(log_dir=os.path.join(output_dir, 'logs'))
                # Train model
        print("Training model...")
        history = model.fit(
            X_train_features, 
            y_train_enc,
            validation_data=(X_val_features, y_val_enc),
            epochs=args.epochs,
            batch_size=args.batch_size,
            callbacks=[checkpoint_cb, early_stopping, tensorboard_cb]
        )
        
        # Plot training history
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('Model accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'training_history.png'), dpi=300)
        
        # Save training history
        with open(os.path.join(output_dir, 'training_history.json'), 'w') as f:
            history_dict = {
                'accuracy': [float(x) for x in history.history['accuracy']],
                'val_accuracy': [float(x) for x in history.history['val_accuracy']],
                'loss': [float(x) for x in history.history['loss']],
                'val_loss': [float(x) for x in history.history['val_loss']]
            }
            json.dump(history_dict, f, indent=4)
        
        # Load best model for evaluation
        best_model = keras.models.load_model(
            os.path.join(output_dir, 'best_model.h5'),
            custom_objects={'TransformerBlock': TransformerBlock}
        )
        
        # Evaluate on test set
        test_loss, test_acc = best_model.evaluate(X_test_features, y_test_enc)
        print(f"Test accuracy: {test_acc:.4f}")
        
        # Get predictions
        y_pred_prob = best_model.predict(X_test_features)
        y_pred = np.argmax(y_pred_prob, axis=1)
        y_true = np.argmax(y_test_enc, axis=1)
        
        # Get readable labels
        reverse_label_map = {v: k for k, v in label_map.items()}
        y_pred_labels = [reverse_label_map[i] for i in y_pred]
        y_true_labels = [reverse_label_map[i] for i in y_true]
        
    else:  # SVM model
        # Prepare features for SVM
        X_train_svm = prepare_features_for_model(X_train_features, 'svm')
        print(f"SVM input shape: {X_train_svm.shape}")
        model.fit(X_train_svm, y_train)
        
        # For evaluation, also transform validation and test features
        X_val_svm = prepare_features_for_model(X_val_features, 'svm')
        X_test_svm = prepare_features_for_model(X_test_features, 'svm')
        
        # Evaluate on test set
        test_acc = model.score(X_test_svm, y_test)
        print(f"Test accuracy: {test_acc:.4f}")
        
        y_pred = model.predict(X_test_svm)
        y_pred_labels = y_pred
        y_true_labels = y_test
        
    # Generate classification report
    print("Generating classification report...")
    from sklearn.metrics import classification_report, confusion_matrix
    import seaborn as sns
    
    report = classification_report(y_true_labels, y_pred_labels, output_dict=True)
    print(classification_report(y_true_labels, y_pred_labels))
    
    with open(os.path.join(output_dir, 'classification_report.json'), 'w') as f:
        json.dump(report, f, indent=4)
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_true_labels, y_pred_labels)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=sorted(label_map.keys()),
                yticklabels=sorted(label_map.keys()))
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), dpi=300)
    
    # Save model summary
    if args.model_type != 'svm':
        # Save model summary to file
        from contextlib import redirect_stdout
        with open(os.path.join(output_dir, 'model_summary.txt'), 'w') as f:
            with redirect_stdout(f):
                model.summary()
    else:
        # Save SVM parameters
        with open(os.path.join(output_dir, 'model_params.json'), 'w') as f:
            params = {
                'C': args.svm_c,
                'kernel': 'linear',
                'feature_type': args.feature_type,
                'max_features': args.max_features
            }
            json.dump(params, f, indent=4)
    
    # Generate summary metrics
    summary = {
        'model_type': args.model_type,
        'feature_type': args.feature_type,
        'test_accuracy': float(test_acc),
        'weighted_f1_score': report['weighted avg']['f1-score'],
        'num_classes': len(label_map),
        'num_authors': len(authors),
        'train_samples': len(X_train),
        'test_samples': len(X_test)
    }
    
    with open(os.path.join(output_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=4)
    
    print(f"All results saved in {output_dir}")

# Add to the end of the file
if __name__ == "__main__":
    main()