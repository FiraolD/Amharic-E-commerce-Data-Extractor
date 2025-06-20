#This script cleans, tokenizes and normalizes Amharic text data using a pre-trained tokenizer.

import pandas as pd
from transformers import AutoTokenizer
import re

# Load cleaned data
df = pd.read_csv('D:/PYTHON PROJECTS/KIAM PROJECTS/Amharic-E-commerce-Data-Extractor/Data/telegram_scraped_data.csv')


def clean_amharic_text(text):
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'[\U00010000-\U0010ffff]', '', text)  # Remove emojis
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'(09|\+2519)\d{8}', '', text)
    text = re.sub(r'\s+', ' ', text).strip()  # Normalize spaces
    return text
# Use a pre-trained Amharic-friendly tokenizer
tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")

def tokenize_and_normalize(text):
    if not isinstance(text, str):
        return []
    # Normalize case (optional for Amharic)
    text = text.strip()
    # Tokenize using XLM-R tokenizer
    tokens = tokenizer.tokenize(text)
    return tokens


# Clean the text data
df['cleaned_text'] = df['text'].apply(clean_amharic_text)
# Check for empty strings and remove them
df = df[df['cleaned_text'].str.strip() != '']
# Reset index after cleaning
df.reset_index(drop=True, inplace=True)

# Apply tokenization
df['tokens'] = df['cleaned_text'].apply(tokenize_and_normalize)

df.to_csv('Data/pre-processed-data.csv', index=False)
print("✅ Tokenized and normalized data saved.")

# Show results
print(df[['cleaned_text', 'tokens']].head(10))
