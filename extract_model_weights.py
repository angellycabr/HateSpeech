import torch
import numpy as np
import pandas as pd
import tensorflow as tf
from transformers import AutoModel
from transformers import DistilBertTokenizer, TFDistilBertModel
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input, LSTM, Bidirectional, Dense, Dropout, Lambda
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, roc_auc_score

# Code was written in Google Colab, hence file paths must be updated and code modifications must be made to run locally
from google.colab import drive
drive.mount('/content/drive')

# Load DistilBERT model and tokenizer
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
distilbert = TFDistilBertModel.from_pretrained('distilbert-base-uncased')

def encode_texts(texts, tokenizer, max_length=128):
    # Tokenizes and encodes text for BERT
    input_ids = []
    attention_masks = []
    for text in texts:
        encoded = tokenizer.encode_plus(text, max_length=max_length, padding='max_length', truncation=True, return_attention_mask=True)
        input_ids.append(encoded['input_ids'])
        attention_masks.append(encoded['attention_mask'])
    return np.array(input_ids), np.array(attention_masks)

def bert_embedding_layer(inputs):
    # Lambda function to use DistilBERT inside Keras model
    input_ids, attention_mask = inputs
    return distilbert(input_ids, attention_mask=attention_mask).last_hidden_state

def build_model():
    # Builds the BERT + LSTM model
    input_ids = Input(shape=(128,), dtype=tf.int32, name='input_ids')
    attention_mask = Input(shape=(128,), dtype=tf.int32, name='attention_mask')

    # Use a Lambda layer with explicit output shape
    sequence_output = Lambda(bert_embedding_layer, output_shape=(128, 768))([input_ids, attention_mask])

    # LSTM layers
    x = Bidirectional(LSTM(64, return_sequences=True))(sequence_output)
    x = Bidirectional(LSTM(32, return_sequences=False))(x)
    x = Dropout(0.2)(x)
    dense = Dense(32, activation='relu')(x)
    output = Dense(1, activation='sigmoid')(dense)

    model = Model(inputs=[input_ids, attention_mask], outputs=output)
    model.compile(optimizer=Adam(learning_rate=5e-5), loss='binary_crossentropy', metrics=['accuracy'])

    return model

# Load dataset
df = pd.read_csv('File name goes here') # Must be updated with filepath

# Prepare inputs, modifications vary based on dataset
comments = df['comment']
parents = df['parent_comment']
labels = df['label']

train_comments, test_comments, train_parents, test_parents, train_labels, test_labels = train_test_split(
    comments, parents, labels, test_size=0.2, random_state=42)

# Encode text
train_comment_ids, train_comment_masks = encode_texts(train_comments, tokenizer)
test_comment_ids, test_comment_masks = encode_texts(test_comments, tokenizer)
train_parent_ids, _ = encode_texts(train_parents, tokenizer)
test_parent_ids, _ = encode_texts(test_parents, tokenizer)

# Build model
model = build_model()
model.load_weights("weights go here")
model.summary()

# Extract embeddings
intermediate_layer_model = Model(
    inputs=model.input,
    outputs=model.get_layer('dense_4').output # Replace with the layer succeeding the embedding layer
)

embeddings = intermediate_layer_model.predict([train_comment_ids, train_parent_ids])
pd.DataFrame(embeddings).to_csv('embeddings.csv', index=False)