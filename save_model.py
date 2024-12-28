from transformers import TFBertForSequenceClassification, BertTokenizer

# Load pre-trained model and tokenizer
model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Save locally
model.save_pretrained('./saved_model')
tokenizer.save_pretrained('./saved_model')
