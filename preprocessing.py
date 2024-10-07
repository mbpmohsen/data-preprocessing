from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
import json
import pandas as pd

# Read JSONL file
logs = []
with open('./test_records.jsonl', 'r') as f:
    for line in f:
        try:
            log = json.loads(line)  # Use the standard json module, not pandas
            logs.append(log)
        except json.JSONDecodeError:
            # Skip invalid JSON lines
            continue


# Convert the list of JSON objects into a DataFrame
df = pd.json_normalize(logs)

# Print the DataFrame to check the results
print(df.head())

# Step 1: Handle missing values by filling them with a default value 'Unknown'
df.fillna('Unknown', inplace=True)

# Step 2: Handle 'date.issued' and 'date.available' fields, flattening if needed
if 'date.issued' in df.columns:
    df['date.issued'] = df['date.issued'].apply(lambda x: x[0] if isinstance(x, list) else x)
    df['date.issued'] = pd.to_datetime(df['date.issued'], errors='coerce')
    df['year_issued'] = df['date.issued'].dt.year
    df['month_issued'] = df['date.issued'].dt.month

if 'date.available' in df.columns:
    df['date.available'] = df['date.available'].apply(lambda x: x[0] if isinstance(x, list) else x)
    df['date.available'] = pd.to_datetime(df['date.available'], errors='coerce')

# Step 3: Calculate the length of the 'description' field
if 'description' in df.columns:
    df['description'] = df['description'].apply(lambda x: ' '.join(x) if isinstance(x, list) else x)
    df['description_length'] = df['description'].apply(len)

# Step 4: Vectorize the 'description' field using TF-IDF
if 'description' in df.columns:
    vectorizer = TfidfVectorizer(max_features=100)
    tfidf_matrix = vectorizer.fit_transform(df['description'])
    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=vectorizer.get_feature_names_out())
    df = pd.concat([df, tfidf_df], axis=1)

# Step 6: Scale numerical features like year, month, and description length
scaler = MinMaxScaler()
features_to_scale = ['year_issued', 'month_issued', 'description_length']
for feature in features_to_scale:
    if feature in df.columns:
        df[feature] = scaler.fit_transform(df[[feature]])

# Save the processed DataFrame to a CSV file
df.to_csv('preprocessed_nasa_logs.csv', index=False)

# Print a success message
print("Preprocessing completed and saved as 'preprocessed_nasa_logs.csv'.")
