from transformers import BertTokenizer, BertForSequenceClassification

# Load the tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# Test the tokenizer
sample_text = "Hello, how can I help you?"
tokens = tokenizer(sample_text, return_tensors='pt')

print("Tokens:", tokens)

# Test the model (optional)
outputs = model(**tokens)
print("Model Outputs:", outputs)
