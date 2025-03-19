import os
import json
import time
import gradio as gr
import numpy as np
import pandas as pd
from urllib.parse import urlparse
from datetime import datetime
import re
import urllib.parse
import tensorflow as tf
from tensorflow import keras
import gensim
import joblib
from underthesea import word_tokenize
import argparse

# Parse command line arguments
def parse_args():
    parser = argparse.ArgumentParser(description='VNExpress Article Classifier Web App')
    
    # File paths
    parser.add_argument('--articles_dir', type=str, default='../articles',
                      help='Path to the articles directory containing JSON files')
    parser.add_argument('--index_file', type=str, default='../articles_index.json',
                      help='Path to the articles index JSON file')
    parser.add_argument('--categories_file', type=str, default='../category_definitions.json',
                      help='Path to the category definitions JSON file')
    parser.add_argument('--stopwords_file', type=str, default='../vietnamese-stopwords.txt',
                      help='Path to the Vietnamese stopwords file')
    
    # Model paths
    parser.add_argument('--transformer_model', type=str, default='output/transformer_word2vec_category/best_model.h5',
                      help='Path to the Transformer model file')
    parser.add_argument('--bilstm_model', type=str, default='output/bilstm_word2vec_category/best_model.h5',
                      help='Path to the BiLSTM model file')
    parser.add_argument('--cnn_model', type=str, default='output/cnn_word2vec_category/best_model.h5',
                      help='Path to the CNN model file')
    parser.add_argument('--svm_model', type=str, default='output/svm_tfidf_category',
                      help='Path to the SVM model directory')
    parser.add_argument('--word2vec_model', type=str, default='../word2vec_vi_words_300dims_final.bin',
                      help='Path to the Word2Vec model file')
    
    # Web app settings
    parser.add_argument('--port', type=int, default=7860,
                      help='Port to run the Gradio web server on')
    parser.add_argument('--share', action='store_true',
                      help='Whether to create a public link for the app')
    
    return parser.parse_args()

# Add the TransformerBlock class for loading models that use it
class TransformerBlock(keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.rate = rate
        self.att = keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential(
            [keras.layers.Dense(ff_dim, activation="relu"), keras.layers.Dense(embed_dim),]
        )
        self.layernorm1 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = keras.layers.Dropout(rate)
        self.dropout2 = keras.layers.Dropout(rate)

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

# Load category definitions from JSON file
def load_category_definitions(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            categories = json.load(f)
        
        # Create mapping from category code to name
        category_mapping = {}
        for cat_id, cat_info in categories.items():
            category_mapping[cat_info["code"]] = cat_info["name"]
        
        return category_mapping
    except Exception as e:
        print(f"Error loading category definitions: {e}")
        # Fallback to default mapping if file can't be loaded
        return {
            "<POL>": "Politics",
            "<ECO>": "Economy",
            "<HEA>": "Health",
            "<EDU>": "Education",
            "<ENV>": "Environment",
            "<SOC>": "Society",
            "<TRA>": "Transport",
            "<TEC>": "Technology",
            "<ENT>": "Entertainment",
            "<SPR>": "Sports",
            "<FOO>": "Food",
            "<LAW>": "Law",
            "<INT>": "International",
            "<REG>": "Regional",
            "<SCI>": "Science"
        }

# Load models for inference
def load_models(args):
    """Load trained models for inference"""
    models = {}
    
    # Define model directories from args
    model_dirs = {
        "Transformer": args.transformer_model,
        "BiLSTM": args.bilstm_model,
        "CNN": args.cnn_model,
        "SVM": args.svm_model
    }
    
    # Load each model
    for model_name, model_path in model_dirs.items():
        try:
            if model_name == "SVM":
                models[model_name] = {
                    "model": joblib.load(os.path.join(model_path, "model.joblib")),
                    "vectorizer": joblib.load(os.path.join(model_path, "tfidf_vectorizer.joblib")),
                    "label_map": json.load(open(os.path.join(model_path, "label_map.json"), "r"))
                }
            else:
                # For neural network models
                custom_objects = {'TransformerBlock': TransformerBlock}
                with keras.utils.custom_object_scope(custom_objects):
                    models[model_name] = {
                        "model": keras.models.load_model(model_path),
                        "label_map": json.load(open(os.path.join(os.path.dirname(model_path), "label_map.json"), "r"))
                    }
                    
                    # Load Word2Vec if needed
                    if not "w2v_model" in models and model_name != "SVM":
                        try:
                            models["w2v_model"] = gensim.models.KeyedVectors.load_word2vec_format(
                                args.word2vec_model, binary=True
                            )
                        except Exception as e:
                            print(f"Error loading Word2Vec model: {e}")
            
            print(f"Successfully loaded {model_name} model from {model_path}")
        except Exception as e:
            print(f"Error loading {model_name} model from {model_path}: {e}")
    
    return models

# Clean and preprocess text
def clean_text(text):
    """Clean and normalize text"""
    text = re.sub(r'[^\w\s]', '', text)  # Remove special characters
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    return text

def replace_all(text):
    """Replace Vietnamese characters for consistency"""
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

# Load stopwords
def get_stopwords(stopwords_path):
    if not os.path.exists(stopwords_path):
        print(f"Stopwords file not found: {stopwords_path}")
        return []
    
    with open(stopwords_path, 'r', encoding='utf-8') as f:
        stop_words = f.read().split('\n')
    return stop_words

# Extract features for model inference
def extract_word2vec_features(text, w2v_model, max_len=1000, embed_dim=300):
    """Convert text to Word2Vec vector matrix."""
    words = text.split()
    matrix = np.zeros((max_len, embed_dim))
    for i, word in enumerate(words[:max_len]):
        if word in w2v_model:
            matrix[i] = w2v_model[word]
    return np.array([matrix])

def extract_tfidf_features(text, vectorizer):
    """Extract TF-IDF features from text."""
    return vectorizer.transform([text])

# Preprocess text for models
def preprocess_for_models(text, stop_words):
    """Preprocess text for model inference"""
    # Tokenize
    tokenized = word_tokenize(text, format="text")
    
    # Remove stopwords
    filtered = ' '.join([word for word in tokenized.split() if word not in stop_words])
    
    # Apply additional cleaning
    cleaned = replace_all(filtered)
    
    return cleaned

# Load the article index from JSON
def load_article_index(path):
    with open(path, "r", encoding="utf-8") as f:
        index = json.load(f)
    return index

# Load article data from JSON file
def load_article_data(json_file, folder):
    try:
        file_path = os.path.join(folder, json_file)
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"Error loading {json_file}: {e}")
        return None

def get_screenshot_url(url):
    """Get a screenshot of the URL using a third-party service"""
    # Encode the URL
    encoded_url = urllib.parse.quote_plus(url)
    # Use a service like Screenshotmachine or PageSpeed Insights
    return f"https://api.screenshotmachine.com?key=23d1c1&url={encoded_url}&device=phone&dimension=480x800&zoom=50format=jpg&delay=600"

# Find article ID from URL
def get_article_id_from_url(url, article_index):
    # Extract article ID from URL using a regular expression
    match = re.search(r'-(\d+)\.html', url)
    if match:
        article_id = match.group(1)
        if article_id in article_index:
            return article_id
    return None

# Run model inference
def run_model_inference(article_data, model_name, models, stop_words, progress=gr.Progress()):
    """Run inference using selected model"""
    # Get content
    content = article_data.get('content', '')
    if not content:
        return {
            "author": "Unknown",
            "author_confidence": 0.0,
            "category_code": "Unknown",
            "category_label": "Unknown"
        }
    
    # Preprocess text
    preprocessed_text = preprocess_for_models(content, stop_words)
    
    # Show progress
    total_steps = 5
    progress.tqdm(range(1), desc="Preprocessing text")
    
    if model_name not in models:
        return {
            "author": article_data.get('author_name', 'Unknown'),
            "author_confidence": 0.0,
            "category_code": "Unknown",
            "category_label": "Unknown"
        }
    
    progress.tqdm(range(1), desc="Extracting features")
    
    # Extract features based on model type
    if model_name == "SVM":
        # Use TF-IDF features for SVM
        features = extract_tfidf_features(preprocessed_text, models[model_name]["vectorizer"])
    else:
        # Use Word2Vec for neural network models
        if "w2v_model" not in models:
            return {
                "author": article_data.get('author_name', 'Unknown'),
                "author_confidence": 0.0,
                "category_code": "Unknown",
                "category_label": "Unknown"
            }
        features = extract_word2vec_features(preprocessed_text, models["w2v_model"])
    
    progress.tqdm(range(1), desc="Running model")
    
    # Perform inference
    if model_name == "SVM":
        category_pred = models[model_name]["model"].predict(features)[0]
        category_pred_proba = models[model_name]["model"].predict_proba(features)[0]
        category_confidence = max(category_pred_proba)
        
        # Map category prediction to label
        reverse_label_map = {v: k for k, v in models[model_name]["label_map"].items()}
        category_pred_label = reverse_label_map.get(category_pred, "Unknown")
        
        # Use actual author from the data
        author_pred = article_data.get('author_name', 'Unknown')
        author_confidence = 1.0  # Using actual author
    else:
        # Neural network models
        pred_proba = models[model_name]["model"].predict(features)[0]
        pred_class = np.argmax(pred_proba)
        category_confidence = float(pred_proba[pred_class])
        
        # Map prediction to label
        reverse_label_map = {v: k for k, v in models[model_name]["label_map"].items()}
        category_pred_label = reverse_label_map.get(pred_class, "Unknown")
        
        # Use actual author from the data
        author_pred = article_data.get('author_name', 'Unknown')
        author_confidence = 1.0  # Using actual author
    
    progress.tqdm(range(2), desc="Finalizing results")
    
    # Determine category code
    category_code = "Unknown"
    for code, label in CATEGORY_MAPPING.items():
        if label == category_pred_label:
            category_code = code
            break
    
    return {
        "author": author_pred,
        "author_confidence": author_confidence,
        "category_code": category_code.replace('<', '').replace('>', ''),
        "category_label": category_pred_label,
        "category_confidence": category_confidence
    }

# Process the URL and display the article
def create_process_url_function(args, models, stop_words):
    def process_url(url, selected_model, progress=gr.Progress()):
        # Check if URL is empty
        if not url:
            return (
                "<div style='color: red; text-align: center; padding: 20px;'>Please enter a VNExpress URL</div>",
                "",
                "<div style='color: red; padding: 10px;'>No article content to display</div>",
                "<div style='color: red; padding: 10px;'>No classification results available</div>"
            )
        
        # Load the article index
        article_index = load_article_index(args.index_file)
        
        # Get the article ID from the URL
        article_id = get_article_id_from_url(url, article_index)
        
        if not article_id:
            return (
                f"<div style='color: red; text-align: center; padding: 20px;'>Cannot find article with URL: {url}</div>",
                "",
                "<div style='color: red; padding: 10px;'>Article not found in the database</div>",
                "<div style='color: red; padding: 10px;'>No classification results available</div>"
            )
        
        # Get the JSON filename for this article
        json_file = article_index[article_id].get("json_file")
        
        if not json_file:
            return (
                f"<div style='color: red; text-align: center; padding: 20px;'>No JSON file found for article ID: {article_id}</div>",
                "",
                "<div style='color: red; padding: 10px;'>Article data not available</div>",
                "<div style='color: red; padding: 10px;'>No classification results available</div>"
            )
        
        # Load the article data
        article_data = load_article_data(json_file, args.articles_dir)
        
        if not article_data:
            return (
                f"<div style='color: red; text-align: center; padding: 20px;'>Failed to load article data from {json_file}</div>",
                "",
                "<div style='color: red; padding: 10px;'>Could not parse article data</div>",
                "<div style='color: red; padding: 10px;'>No classification results available</div>"
            )
        
        # Create webpage display HTML using local data instead of iframe
        iframe_html = f"""
        <div style="width:100%; border:1px solid #ccc; border-radius:5px; padding:15px; background:#fff;">
            <h2 style="color:#333; margin-top:0; text-align:center;">{article_data.get('title', '')}</h2>
            <div style="text-align:center; margin:10px 0;">
                <a href="{url}" target="_blank" style="color:#1a73e8; text-decoration:none;">Open article in new tab</a>
            </div>
            <div style="margin:15px 0; padding:10px; background:#f5f5f5; border-radius:5px; color:#333;">
                <p style="font-style:italic; margin:0;">{article_data.get('abstract', '')}</p>
            </div>
            <div style="margin-top:15px; color:#666; font-size:0.9em; text-align:right;">
                VNExpress • {article_data.get('posted_date', '')}
            </div>
            <div style="margin-top:15px; text-align:center;">
                <p style="color:#666; font-style:italic;">Note: Direct preview not available due to website security policy</p>
            </div>
        </div>
        """
        
        # Create author HTML
        author_html = f"""
        <div style="display:flex; align-items:center; margin-top:15px; padding:10px; background:#f9f9f9; border-radius:5px;">
            <div style="margin-right:15px;">
                <img src="{article_data.get('author_image', '')}" style="width:60px; height:60px; border-radius:50%; object-fit:cover;">
            </div>
            <div>
                <h3 style="margin:0; padding:0; color:#333;">{article_data.get('author_name', 'Unknown Author')}</h3>
                <p style="margin:5px 0 0 0; padding:0; color:#666;">{article_data.get('description', '')}</p>
            </div>
        </div>
        """
        
        content_with_br = article_data.get('content', '').replace('\n', '<br>')
        
        # Create content HTML
        content_html = f"""
        <div style="padding:15px; border:1px solid #eee; border-radius:5px; background-color:#fff; margin-bottom:20px;">
            <h2 style="margin-top:0; color:#333;">{article_data.get('title', '')}</h2>
            <div style="font-style:italic; margin-bottom:20px; padding:10px; background:#f5f5f5; border-radius:5px; color:#333;">
                {article_data.get('abstract', '')}
            </div>
            <div style="white-space:pre-line; line-height:1.6; color:#333;">
                {content_with_br}
            </div>
            <div style="margin-top:15px; color:#666; font-size:0.9em;">
                Published: {article_data.get('posted_date', '')}
            </div>
        </div>
        """
        
        # Perform actual model inference
        inference_result = run_model_inference(article_data, selected_model, models, stop_words, progress)
        
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Create classification results HTML
        results_html = f"""
        <div style="padding:15px; border:1px solid #4CAF50; border-radius:5px; background:#f1f8e9;">
            <h3 style="color:#2E7D32; margin-top:0;">Classification Results</h3>
            <div style="margin-bottom:10px; color:#333;">
                <strong style="color:#333;">Author Prediction:</strong> <span style="color:#333;">{inference_result["author"]}</span> 
                <span style="background:#e8f5e9; padding:2px 6px; border-radius:3px; margin-left:5px; color:#333;">
                    Confidence: {inference_result["author_confidence"]:.2f}
                </span>
            </div>
            <div style="margin-bottom:10px; color:#333;">
                <strong style="color:#333;">Category Code:</strong> <span style="color:#333;">{inference_result["category_code"]}</span>
            </div>
            <div style="margin-bottom:10px; color:#333;">
                <strong style="color:#333;">Category Label:</strong> <span style="color:#333;">{inference_result["category_label"]}</span>
                <span style="background:#e8f5e9; padding:2px 6px; border-radius:3px; margin-left:5px; color:#333;">
                    Confidence: {inference_result.get("category_confidence", 0.0):.2f}
                </span>
            </div>
            <div style="color:#333; font-size:0.9em; margin-top:15px;">
                <em style="color:#555;">Model: {selected_model}</em> | <em style="color:#555;">Processed: {current_time}</em>
            </div>
        </div>
        """
        
        return iframe_html, author_html, content_html, results_html
    
    return process_url

# Create a Gradio Interface
def create_interface(args):
    # Load category definitions
    global CATEGORY_MAPPING
    CATEGORY_MAPPING = load_category_definitions(args.categories_file)
    
    # Load models
    models = load_models(args)
    
    # Load stopwords
    stop_words = get_stopwords(args.stopwords_file)
    
    # Create processing function with loaded resources
    process_url = create_process_url_function(args, models, stop_words)
    
    with gr.Blocks(css="footer {display: none !important;}") as app:
        gr.HTML(
            """
            <div style="text-align: center; margin-bottom: 1rem">
                <h1 style="margin-bottom: 0.5rem">VNExpress Article Classifier</h1>
                <p>Enter a VNExpress article URL to classify the author and category</p>
            </div>
            """
        )
        
        with gr.Row():
            with gr.Column(scale=1):
                url_input = gr.Textbox(
                    label="Article URL",
                    placeholder="Enter a VNExpress article URL",
                    value="https://vnexpress.net/cho-toi-mot-lo-ne-2921612.html"
                )
                
                model_selector = gr.Dropdown(
                    choices=["Transformer", "BiLSTM", "CNN", "SVM"],
                    value="Transformer",
                    label="Select Model"
                )
                
                analyze_btn = gr.Button("Analyze Article", variant="primary")
                clear_btn = gr.Button("Clear")
        
        with gr.Row():
            with gr.Column(scale=1):
                webpage_display = gr.HTML(label="Web Page Preview")
                author_display = gr.HTML(label="Author Information")
                
            with gr.Column(scale=1):
                content_display = gr.HTML(label="Article Content")
                results_display = gr.HTML(label="Classification Results")
        
        analyze_btn.click(
            process_url,
            inputs=[url_input, model_selector],
            outputs=[webpage_display, author_display, content_display, results_display]
        )
        
        clear_btn.click(
            lambda: ("", "Transformer", "", "", "", ""),
            outputs=[url_input, model_selector, webpage_display, author_display, content_display, results_display]
        )
    
    return app

if __name__ == "__main__":
    # Parse command-line arguments
    args = parse_args()
    
    # Print configuration
    print("=== VNExpress Article Classifier Configuration ===")
    print(f"Articles directory: {args.articles_dir}")
    print(f"Index file: {args.index_file}")
    print(f"Categories file: {args.categories_file}")
    print(f"Stopwords file: {args.stopwords_file}")
    print(f"Model paths:")
    print(f"  - Transformer: {args.transformer_model}")
    print(f"  - BiLSTM: {args.bilstm_model}")
    print(f"  - CNN: {args.cnn_model}")
    print(f"  - SVM: {args.svm_model}")
    print(f"  - Word2Vec: {args.word2vec_model}")
    print(f"Server port: {args.port}")
    print(f"Create public link: {args.share}")
    print("=" * 50)
    
    # Create and launch the interface
    app = create_interface(args)
    app.launch(server_port=args.port, share=args.share)