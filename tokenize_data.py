# tokenize_data.py
from transformers import BertTokenizer
import torch

# Initialize tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def tokenize_data(data):
    texts = [item['text'] for item in data]
    labels = [item['label'] for item in data]
    
    tokens = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
    input_ids = tokens['input_ids']
    attention_mask = tokens['attention_mask']
    labels = torch.tensor(labels)
    
    return input_ids, attention_mask, labels

if __name__ == "__main__":
    from split_dataset import split_data
    from dataset import get_dataset
    
    dataset = get_dataset()
    train_data, val_data = split_data(dataset)
    
    train_input_ids, train_attention_mask, train_labels = tokenize_data(train_data)
    val_input_ids, val_attention_mask, val_labels = tokenize_data(val_data)
    
    print("Tokenized Training Data:", train_input_ids.shape, train_labels.shape)
    print("Tokenized Validation Data:", val_input_ids.shape, val_labels.shape)
