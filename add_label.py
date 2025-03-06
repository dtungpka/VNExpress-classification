import json
import os
import glob

# Load the article_labels data
with open('/home/21010294/NLP/VNExpress/article_labels.json', 'r',encoding='utf-8') as f:
    article_labels = json.load(f)

# Get all article files
article_files = glob.glob('/home/21010294/NLP/VNExpress/articles/*.json')

# Process each article file
for article_file in article_files:
    # Extract the filename
    filename = os.path.basename(article_file)
    
    # Open and load the article data
    with open(article_file, 'r',encoding='utf-8') as f:
        article_data = json.load(f)
    
    # Check if the filename exists in article_labels
    if filename in article_labels:
        # Add the labeled_date to the article data
        article_data['labeled_date'] = article_labels[filename].get('labeled_date')
        article_data['category_code'] = article_labels[filename].get('label')
        article_data['category_label'] = article_labels[filename].get('category')
        
        # Write back the updated article data
        with open(article_file, 'w',encoding='utf-8') as f:
            json.dump(article_data, f, indent=4, ensure_ascii=False)
        
        print(f"Updated {filename}")
    else:
        print(f"Skipped {filename} - not found in article_labels")

print("Process completed!")