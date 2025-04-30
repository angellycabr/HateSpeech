from wordcloud import WordCloud
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizer, TFDistilBertModel
from transfer_learning_hate_speech import build_model, encode_texts

# NOTE: This code is to generate a world cloud of common words used in positive samples. Code modifications are necessary

# Load the model weights
model = build_model()
model.load_weights("ethos_model.weights.h5")

# Load DistilBERT model and tokenizer
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
distilbert = TFDistilBertModel.from_pretrained('distilbert-base-uncased')

# Split the data to extract the text 
df = pd.read_csv('File name goes here')
comments = df['comment']
labels = df['label']

X_train, X_test, y_train, y_test = train_test_split(df['comment'], df['label'], test_size=0.6, random_state=42) 

train_comments, test_comments, train_labels, test_labels = train_test_split(
    comments, labels, test_size=0.6, random_state=42)

train_comment_ids, train_comment_masks = encode_texts(train_comments, tokenizer)
test_comment_ids, test_comment_masks = encode_texts(test_comments, tokenizer)

# Reset indices
X_test = X_test.reset_index(drop=True)
y_test_indices = y_test.reset_index(drop=True)

# Generate predictions
predictions = (model.predict([test_comment_ids, test_comment_masks]) >= 0.5).astype(int)
predicted_labels_indices = predictions.flatten()

# Identify correct and incorrect predictions
correct_indices = np.where(predicted_labels_indices == y_test_indices.to_numpy())[0]
incorrect_indices = np.where(predicted_labels_indices != y_test_indices.to_numpy())[0]
correct_sarcasm_samples = [X_test[i] for i in correct_indices ]

# Generate a word cloud
def generate_wordcloud(text_samples, title):
    if text_samples:  # Ensure there are valid samples
        text = " ".join(text_samples)
        wordcloud = WordCloud(width=800, height=400, background_color="white", colormap="coolwarm").generate(text)

        # Display word cloud
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        plt.title(title, fontsize=14)
        plt.show()
    else:
        print(f"No correctly classified samples found for {title}.")

generate_wordcloud(correct_sarcasm_samples, "Word Cloud of Correctly Classified Samples")

# Print first 5 correct and incorrect predictions
print("\nFirst 5 Correct Predictions:")
for i in correct_indices[:5]:
    print(f"Sample {i}:")
    print(f"True Label: {y_test_indices.iloc[i]}, Predicted Label: {predicted_labels_indices[i]}")
    print(f"Text: {X_test[i]}")
    print("-" * 50)

print("\nFirst 5 Incorrect Predictions:")
for i in incorrect_indices[:5]:
    print(f"Sample {i}:")
    print(f"True Label: {y_test_indices.iloc[i]}, Predicted Label: {predicted_labels_indices[i]}")
    print(f"Text: {X_test[i]}")
    print("-" * 50)