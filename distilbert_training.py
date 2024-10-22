#!/usr/bin/env python3

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizer, TFDistilBertForSequenceClassification
from sklearn.preprocessing import MultiLabelBinarizer
import tensorflow as tf
import pickle

from tensorflow.keras.mixed_precision import set_global_policy

# Enable mixed precision
set_global_policy('mixed_float16')


# Step 1: Load the new CSV file
df = pd.read_csv('final_filtered!(onlyHard).csv')

# Step 2: Extract job descriptions and labels
texts = df['Combined_Description'].tolist()
labels = df['Labels'].apply(lambda x: x.strip("[]").replace("'", "").split(", ")).tolist()

# Step 3: Split data into training, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(texts, labels, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

print(f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}, Test samples: {len(X_test)}")

# Step 4: Tokenization and Encoding with Conditional Loading
# Path for the encoded data folder
encoded_data_dir = 'encoded_data/'

# Create the encoded_data folder if it doesn't exist
if not os.path.exists(encoded_data_dir):
    os.makedirs(encoded_data_dir)
    print(f"Created directory: {encoded_data_dir}")

# Paths to save/load tokenized data
X_train_enc_path = os.path.join(encoded_data_dir, 'X_train_enc.pkl')
X_val_enc_path = os.path.join(encoded_data_dir, 'X_val_enc.pkl')
X_test_enc_path = os.path.join(encoded_data_dir, 'X_test_enc.pkl')

if os.path.exists(X_train_enc_path) and os.path.exists(X_val_enc_path) and os.path.exists(X_test_enc_path):
    print("Loading encoded data from saved files.")
    with open(X_train_enc_path, 'rb') as f:
        X_train_enc = pickle.load(f)
    with open(X_val_enc_path, 'rb') as f:
        X_val_enc = pickle.load(f)
    with open(X_test_enc_path, 'rb') as f:
        X_test_enc = pickle.load(f)
else:
    print("Tokenizing and encoding data.")
    tokenizer_save_path = 'Bandito/tokenizer'

    # Check if the tokenizer exists at the specified path
    if os.path.exists(tokenizer_save_path):
        print("Loading tokenizer from saved location.")
        tokenizer = DistilBertTokenizer.from_pretrained(tokenizer_save_path)
    else:
        print("Saved tokenizer not found. Creating and saving a new tokenizer.")
        tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        tokenizer.save_pretrained(tokenizer_save_path)
        print(f"Tokenizer saved to {tokenizer_save_path}")

    # Tokenize and encode the data
    X_train_enc = tokenizer(X_train, truncation=True, padding=True, max_length=512, return_tensors='tf')
    X_val_enc = tokenizer(X_val, truncation=True, padding=True, max_length=512, return_tensors='tf')
    X_test_enc = tokenizer(X_test, truncation=True, padding=True, max_length=512, return_tensors='tf')

    # Save the encoded data for future use
    with open(X_train_enc_path, 'wb') as f:
        pickle.dump(X_train_enc, f)
    with open(X_val_enc_path, 'wb') as f:
        pickle.dump(X_val_enc, f)
    with open(X_test_enc_path, 'wb') as f:
        pickle.dump(X_test_enc, f)

# Step 6: Convert labels to a binary matrix
mlb = MultiLabelBinarizer()
y_train_bin = mlb.fit_transform(y_train)
y_val_bin = mlb.transform(y_val)
y_test_bin = mlb.transform(y_test)

# Step 7: Build the model using DistilBERT
model = TFDistilBertForSequenceClassification.from_pretrained(
    'distilbert-base-uncased',
    num_labels=len(mlb.classes_)
)

# Step 8: Compile the model
model.compile(optimizer=tf.optimizers.Adam(learning_rate=1e-3),
              loss=tf.keras.losses.BinaryCrossentropy(),
              metrics=['accuracy'])

# Step 9: Set up checkpointing and early stopping
checkpoint_dir = 'model_checkpoints'
os.makedirs(checkpoint_dir, exist_ok=True)

checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=os.path.join(checkpoint_dir, 'model_epoch_{epoch:02d}.weights.h5'),
    save_weights_only=True,
    save_freq='epoch'
)

early_stopping_callback = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', patience=2, restore_best_weights=True
)

# Step 10: Train the model with checkpointing
history = model.fit(
    X_train_enc['input_ids'],
    y_train_bin,
    validation_data=(X_val_enc['input_ids'], y_val_bin),
    epochs=1,
    batch_size=32,
)

# Step 11: Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(X_test_enc['input_ids'], y_test_bin)
print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")

# Save the final model to Google Drive
model_save_path = 'distilbandito_finalformaru'
model.save_pretrained(model_save_path)
tokenizer.save_pretrained(model_save_path)
