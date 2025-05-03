import pandas as pd
import numpy as np
import random
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
import seaborn as sns

#from google.colab import drive
#drive.mount('/content/drive')

# NOTE: This code can be run locally. However, you will need to adjust the dataframe processing in lines 77-88 based on your chosen datasets

def calculate_jaccard(sarcasm_texts, hate_texts, top_n):
    # Computes the n-gram frquencies
    vectorizer = CountVectorizer(ngram_range=(1, 2), stop_words='english')
    X_sarcasm = vectorizer.fit_transform(sarcasm_texts)
    X_hate = vectorizer.transform(hate_texts)

    vocab = vectorizer.get_feature_names_out()
    sarcasm_freqs = pd.DataFrame(X_sarcasm.toarray(), columns=vocab).sum().sort_values(ascending=False)
    hate_freqs = pd.DataFrame(X_hate.toarray(), columns=vocab).sum().sort_values(ascending=False)

    # Normalize
    sarcasm_norm = sarcasm_freqs / sarcasm_freqs.sum()
    hate_norm = hate_freqs / hate_freqs.sum()

    # Compute Jacard similarity
    top_sarcasm = set(sarcasm_norm.head(top_n).index)
    top_hate = set(hate_norm.head(top_n).index)
    jaccard = len(top_sarcasm & top_hate) / len(top_sarcasm | top_hate)
    return jaccard, sarcasm_norm, hate_norm

def plot_top_ngrams(sarcasm_freqs_norm, hate_freqs_norm, top_n, title='Top N-gram Comparison'):
    top_sarcasm = sarcasm_freqs_norm.head(top_n)
    top_hate = hate_freqs_norm.head(top_n)
    df = pd.DataFrame({
        'n-gram': list(top_sarcasm.index) + list(top_hate.index),
        'frequency': list(top_sarcasm.values) + list(top_hate.values),
        'source': ['Sarcasm'] * top_n + ['Hate'] * top_n
    })

    df['n-gram'] = df['n-gram'].astype(str)
    df = df.sort_values(by='frequency', ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x='frequency', y='n-gram', hue='source')
    plt.title(title)
    plt.xlabel("Normalized Frequency")
    plt.ylabel("N-gram")
    plt.tight_layout()
    plt.show()

def bootstrap_jaccard(sarcasm_samples, hate_samples, top_n=100, n_iterations=200, sample_size=100):
  # Find the average similarity score from n iterations with 100 samples each
    jaccard_scores = []
    for _ in range(n_iterations):
        sarcasm_sample = random.sample(sarcasm_samples, sample_size)
        hate_sample = random.sample(hate_samples, sample_size)
        jaccard, _, _ = calculate_jaccard(sarcasm_sample, hate_sample, top_n=top_n)
        jaccard_scores.append(jaccard)

    print(f"Bootstrapped Jaccard (top_n={top_n}, n={n_iterations}):")
    print(f"Mean: {np.mean(jaccard_scores):.3f}, Std: {np.std(jaccard_scores):.3f}, Min: {np.min(jaccard_scores):.3f}, Max: {np.max(jaccard_scores):.3f}")

    plt.hist(jaccard_scores, bins=10, color='skyblue', edgecolor='black')
    plt.title("Bootstrapped Jaccard Similarity")
    plt.xlabel("Jaccard Score")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()

# Imports the data and only saves positive sarcas and hate samples
sarcasm_data = pd.read_csv('Insert file path')
corpus_data = pd.read_csv('Insert file path')
ethos_data = pd.read_csv('Insert file path')

sarcasm_texts = sarcasm_data[sarcasm_data['label'] == 1]
sarcasm_samples = sarcasm_texts["comment"].to_list()

# We only care about implicit here (because we want to compare implicit vs. sarcasm and sarcasm vs. overall hate)
corpus_data['implicit'] = corpus_data['class'].apply(lambda x: 1 if x in ['implicit_hate'] else 0)
corpus_comments = corpus_data['post']
corpus_labels_implicit = corpus_data['implicit']
corpus_samples = corpus_data[corpus_data['implicit'] == 1]['post'].to_list()

ethos_data['label'] = ethos_data['isHate'].apply(lambda x: 1 if x >= 0.33 else 0)
ethos_texts = ethos_data[ethos_data['label'] == 1]
ethos_samples = ethos_texts["comment"].to_list()

jaccard, sarcasm_norm, hate_norm = calculate_jaccard(random.sample(sarcasm_samples, 100), random.sample(ethos_samples, 100), top_n=200)
print(f"Jaccard Similarity (Top 200): {jaccard:.3f}")

plot_top_ngrams(sarcasm_norm, hate_norm, top_n=10)
bootstrap_jaccard(sarcasm_samples, corpus_samples, top_n=200, n_iterations=100, sample_size=100)