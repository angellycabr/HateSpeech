import pandas as pd
import numpy as np
import tensorflow as tf
from transformers import DistilBertTokenizer, TFDistilBertModel
from tensorflow.keras.layers import Input, LSTM, Bidirectional, Dense, Dropout, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split

# NOTE: This code is better as a notebook to run cell blocks independently (also saves you time)
from google.colab import drive
drive.mount('/content/drive')

# Load DistilBERT model & tokenizer
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

"""Define Model Architecture"""

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

"""Track Performance"""

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, roc_auc_score

def evaluate_model(model, X_test, y_test, masks):
    # Evaluate model performance
    predictions = (model.predict([X_test, masks]) >= 0.5).astype(int)

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)
    mcc = matthews_corrcoef(y_test, predictions)
    auc = roc_auc_score(y_test, predictions)

    return {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1,
        "MCC": mcc,
        "AUC": auc
    }

results = {}

"""Train Sarcasm Model (Multi-Input)"""

# Load sarcasm dataset
sarcasm_df = pd.read_csv('Path to dataset')

# Prepare inputs
sarcasm_comments = sarcasm_df['comment']
sarcasm_parents = sarcasm_df['parent_comment']
sarcasm_labels = sarcasm_df['label']

train_comments, test_comments, train_parents, test_parents, train_labels, test_labels = train_test_split(
    sarcasm_comments, sarcasm_parents, sarcasm_labels, test_size=0.2, random_state=42)

# Encode text
train_comment_ids, train_comment_masks = encode_texts(train_comments, tokenizer)
test_comment_ids, test_comment_masks = encode_texts(test_comments, tokenizer)
train_parent_ids, _ = encode_texts(train_parents, tokenizer)
test_parent_ids, _ = encode_texts(test_parents, tokenizer)

# Build sarcasm model
sarcasm_model = build_model()

# Train
sarcasm_model.fit([train_comment_ids, train_comment_masks, train_parent_ids], train_labels, epochs=5, batch_size=64)

# Evaluate
results["Sarcasm"] = evaluate_model(sarcasm_model, test_comment_ids, test_labels, test_comment_masks)

# Save weights
sarcasm_model.save_weights("sarcasm_model_weights.h5")

"""Load Implicit Hate Corpus & Fine-Tune Model"""

from sklearn.utils.class_weight import compute_class_weight

# Load Implicit Corpus
corpus_df = pd.read_csv('Path to dataset')
corpus_df['hate'] = corpus_df['class'].apply(lambda x: 1 if x in ['implicit_hate', 'explicit_hate'] else 0)
corpus_df['implicit'] = corpus_df['class'].apply(lambda x: 1 if x in ['implicit_hate'] else 0)

# Prepare inputs
corpus_comments = corpus_df['post']
corpus_labels_implicit = corpus_df['implicit']
corpus_labels_hate = corpus_df['hate']

train_comments, test_comments, train_labels_hate, test_labels_hate = train_test_split(
    corpus_comments, corpus_labels_implicit, test_size=0.2, random_state=42)

train_comment_ids, train_comment_masks = encode_texts(train_comments, tokenizer)
test_comment_ids, test_comment_masks = encode_texts(test_comments, tokenizer)

# Build Corpus model
corpus_model = build_model()

# Load sarcasm-trained weights into Corpus model
corpus_model.load_weights("sarcasm_model_weights.h5", by_name=True, skip_mismatch=True)

# Freeze BERT layers (optional since we are using a transfer learning apporach)
#for layer in distilbert.layers:
#    layer.trainable = False

# Given the dataset is unbalanced, we'll weigh the classes
implicit_hate_labels = np.array(corpus_labels_implicit)

class_weights = compute_class_weight(class_weight="balanced", classes=np.array([0, 1]), y=implicit_hate_labels)
class_weight_dict = {0: class_weights[0], 1: class_weights[1]}
sample_weights = np.array([class_weight_dict[label] for label in train_labels_hate])

# Fine-tune on Corpus
corpus_model.fit([train_comment_ids, train_comment_masks], train_labels_hate, epochs=15, batch_size=64, sample_weight=sample_weights, validation_data=([test_comment_ids, test_comment_masks], test_labels_hate))

# Evaluate
results["Implicit Hate"] = evaluate_model(corpus_model, test_comment_ids, test_labels_hate, test_comment_masks)

# Save weights
corpus_model.save_weights("corpus_model.weights.h5")
#files.download('corpus_model_weights.h5')

"""Load ETHOS Dataset & Fine-Tune Model"""

# Load ETHOS dataset
ethos_df = pd.read_csv('Path to dataset')
ethos_df['label'] = ethos_df['isHate'].apply(lambda x: 1 if x >= 0.33 else 0) # Threshold can be adjusted, we just opted for 0.33 based on our evaluation of the dataset

# Prepare inputs
ethos_comments = ethos_df['comment']
ethos_labels = ethos_df['label']

train_comments, test_comments, train_labels, test_labels = train_test_split(
    ethos_comments, ethos_labels, test_size=0.6, random_state=42)

train_comment_ids, train_comment_masks = encode_texts(train_comments, tokenizer)
test_comment_ids, test_comment_masks = encode_texts(test_comments, tokenizer)

# Build ETHOS model (without parent comment)
ethos_model = build_model()

# Load sarcasm-trained weights into ETHOS model
ethos_model.load_weights("corpus_model.weights.h5")

# Freeze BERT layers initially
#for layer in distilbert.layers:
#    layer.trainable = False

# Fine-tune on ETHOS
ethos_model.fit([train_comment_ids, train_comment_masks], train_labels, epochs=20, batch_size=64, validation_data=([test_comment_ids, test_comment_masks], test_labels))

# Evaluate
results["ETHOS"] = evaluate_model(ethos_model, test_comment_ids, test_labels, test_comment_masks)

# Save weights
ethos_model.save_weights("ethos_model.weights.h5")

"""Display Metrics"""

# Convert results to df
import pandas as pd
results_df = pd.DataFrame(results).T

# Display final results
print("\nPerformance Metrics After Each Stage:")
print(results_df)

"""Evaluate the Model"""

from sklearn.metrics import precision_score, recall_score, f1_score, roc_curve, auc

# Make predictions
predicted_probs = ethos_model.predict([test_comment_ids, test_comment_masks])
predicted_labels = (predicted_probs >= 0.5).astype(int) # Confidence interval can also be adjusted

# Calculate metrics
precision = precision_score(test_labels, predicted_labels)
recall = recall_score(test_labels, predicted_labels)
f1 = f1_score(test_labels, predicted_labels)

print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1-Score: {f1:.2f}")