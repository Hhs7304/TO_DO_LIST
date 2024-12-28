from transformers import BertTokenizer, BertForSequenceClassification

# Download and load the tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Download and load the pre-trained BERT model
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)