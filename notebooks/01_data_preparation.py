from datasets import load_dataset
import pandas as pd
import json

data = load_dataset("Abirate/english_quotes")
df = pd.DataFrame(data['train'])
df.dropna(subset=['quote', 'author'], inplace=True)
df['tags'] = df['tags'].apply(lambda x: x if x else [])
df['full_text'] = df['quote'] + ' - ' + df['author'] + ' | ' + df['tags'].apply(lambda x: ', '.join(x))
cleaned_data = df[['quote', 'author', 'tags', 'full_text']].to_dict(orient='records')
with open("data/cleaned_quotes.json", "w") as f:
    json.dump(cleaned_data, f, indent=2)