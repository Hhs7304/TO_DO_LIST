# train_model.py
from transformers import BertForSequenceClassification, AdamW
from torch.utils.data import DataLoader, TensorDataset
from tokenize_data import tokenize_data
from split_dataset import split_data
from dataset import get_dataset

def train_model(train_loader, model, optimizer, epochs=3):
    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for batch in train_loader:
            b_input_ids, b_attention_mask, b_labels = batch

            optimizer.zero_grad()
            outputs = model(input_ids=b_input_ids, attention_mask=b_attention_mask, labels=b_labels)
            loss = outputs.loss
            total_loss += loss.item()

            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch + 1}, Training Loss: {total_loss / len(train_loader)}")

if __name__ == "__main__":
    dataset = get_dataset()
    train_data, val_data = split_data(dataset)

    train_input_ids, train_attention_mask, train_labels = tokenize_data(train_data)
    
    train_dataset = TensorDataset(train_input_ids, train_attention_mask, train_labels)
    train_loader = DataLoader(train_dataset, batch_size=8)

    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=8)
    optimizer = AdamW(model.parameters(), lr=5e-5)

    train_model(train_loader, model, optimizer)
