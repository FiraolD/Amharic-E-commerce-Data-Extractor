import pandas as pd
from transformers import AutoTokenizer

# Load cleaned data
df = pd.read_csv('D:/PYTHON PROJECTS/KIAM PROJECTS/Amharic-E-commerce-Data-Extractor/Data/telegram_scraped_data.csv')

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

# Apply tokenization
df['tokens'] = df['cleaned_text'].apply(tokenize_and_normalize)

df.to_csv('Data/tokenized_telegram_data.csv', index=False)
print("✅ Tokenized and normalized data saved.")

# Show results
print(df[['cleaned_text', 'tokens']].head())
