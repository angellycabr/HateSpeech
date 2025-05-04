import random
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from scipy.spatial.distance import jensenshannon
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from sentence_transformers import SentenceTransformer

# NOTE: These are additional similarity tests separate from Jacard 

#from google.colab import drive
#drive.mount('/content/drive')

# Cosine is better for large text comparisons which are present in the dataset
def calculate_cosine(sarcasm_texts, hate_texts):
  distilbert = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')

  sarcasm_embeddings = distilbert.encode(sarcasm_texts, convert_to_tensor=True).cpu()
  hate_embeddings = distilbert.encode(hate_texts, convert_to_tensor=True).cpu()

  cosine_matrix = cosine_similarity(sarcasm_embeddings, hate_embeddings)
  mean_cosine = np.mean(cosine_matrix)

  return mean_cosine, cosine_matrix

def bootstrap_cosine_similarity(sarcasm_texts, hate_texts, model_name='distilbert-base-nli-stsb-mean-tokens', n_iter=100, sample_size=100):
    model = SentenceTransformer(model_name)
    scores = []

    for _ in range(n_iter):
        sarcasm_sample = random.sample(sarcasm_texts, sample_size)
        hate_sample = random.sample(hate_texts, sample_size)

        s_embeddings = model.encode(sarcasm_sample, convert_to_tensor=True).cpu()
        h_embeddings = model.encode(hate_sample, convert_to_tensor=True).cpu()

        cosine_matrix = cosine_similarity(s_embeddings, h_embeddings)
        mean_score = np.mean(cosine_matrix)
        scores.append(mean_score)

    print(f"Bootstrapped Semantic Cosine Similarity ({model_name}):")
    print(f"Mean: {np.mean(scores):.3f}")
    print(f"Std Dev: {np.std(scores):.3f}")
    print(f"Min: {np.min(scores):.3f}")
    print(f"Max: {np.max(scores):.3f}")
    
    return scores

def calculate_chi(sarcasm_samples, hate_samples):
  texts = sarcasm_samples + hate_samples
  labels = ['sarcasm'] * len(sarcasm_samples) + ['hate'] * len(hate_samples)

  vectorizer = CountVectorizer(ngram_range=(1, 2), stop_words='english', max_features=1000)
  X = vectorizer.fit_transform(texts)
  chi_df = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())
  chi_df['label'] = labels

  chi_scores = []
  for term in chi_df.columns[:-1]:
      contingency = pd.crosstab(chi_df[term], chi_df['label'])
      if contingency.shape == (2, 2): 
          chi2, p, _, _ = chi2_contingency(contingency)
          chi_scores.append((term, chi2, p))

  # Sort terms by significance
  significant_terms = sorted(chi_scores, key=lambda x: x[1], reverse=True)[:20]
  return significant_terms

def calculate_JS(sarcasm_samples, hate_samples):
  vectorizer = CountVectorizer(ngram_range=(1, 2), stop_words='english', max_features=1000)

  # Fit vectorizer on both datasets combined
  vectorizer.fit(sarcasm_samples + hate_samples)

  sarcasm_vector = vectorizer.transform(sarcasm_samples).toarray().sum(axis=0)
  hate_vector = vectorizer.transform(hate_samples).toarray().sum(axis=0)

  # Normalize
  sarcasm_dist = sarcasm_vector / sarcasm_vector.sum()
  hate_dist = hate_vector / hate_vector.sum()

  js_distance = jensenshannon(sarcasm_dist, hate_dist, base=2)
  return js_distance

def bootstrap_JS_distance(sarcasm_texts, hate_texts, n_iter=100, sample_size=100):
    distances = []

    for _ in range(n_iter):
        s_sample = random.sample(sarcasm_texts, sample_size)
        h_sample = random.sample(hate_texts, sample_size)

        distance = calculate_JS(s_sample, h_sample)
        distances.append(distance)

    print(f"Bootstrapped Jensen-Shannon Divergence:")
    print(f"Mean: {np.mean(distances):.3f}")
    print(f"Std Dev: {np.std(distances):.3f}")
    print(f"Min: {np.min(distances):.3f}")
    print(f"Max: {np.max(distances):.3f}")
    
    return distances

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

# Run tests
scores = bootstrap_cosine_similarity(sarcasm_samples, corpus_samples, model_name='distilbert-base-nli-stsb-mean-tokens', n_iter=100, sample_size=100)
chi_df = pd.DataFrame(calculate_chi(random.sample(sarcasm_samples, 100), random.sample(ethos_samples, 100)), columns=["Term", "Chi2", "p-value"])
js_distances = bootstrap_JS_distance(sarcasm_samples, corpus_samples, n_iter=100, sample_size=100)