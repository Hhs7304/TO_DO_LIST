import numpy as np
from transformers import BertTokenizer, TFBertForSequenceClassification
from sklearn.metrics import accuracy_score, classification_report
import os

# Optional: Suppress TensorFlow logs for cleaner output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Load the tokenizer and trained model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = TFBertForSequenceClassification.from_pretrained('./saved_model')  # Path to your saved model

# Define your test dataset
test_texts = [
    "Hello, how are you?",
    "What is your return policy?",
    "I want to file a complaint.",
    "I have problems with canceling an order."
]
test_labels = [0, 1, 2, 3]  # Replace with the correct labels for your dataset

# Preprocess the test data
test_encodings = tokenizer(test_texts, padding=True, truncation=True, return_tensors="tf")

# Make predictions
predictions = model.predict(test_encodings['input_ids'])

# Convert logits to predicted class labels
predicted_labels = np.argmax(predictions.logits, axis=1)

# Calculate and print evaluation metrics
accuracy = accuracy_score(test_labels, predicted_labels)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:")
print(classification_report(test_labels, predicted_labels))
