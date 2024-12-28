# split_dataset.py
import random
from dataset import get_dataset

def split_data(dataset, split_ratio=0.8):
    random.shuffle(dataset)
    split_index = int(len(dataset) * split_ratio)
    train_data = dataset[:split_index]
    val_data = dataset[split_index:]
    return train_data, val_data

if __name__ == "__main__":
    dataset = get_dataset()
    train_data, val_data = split_data(dataset)
    print("Training Data:", train_data)
    print("Validation Data:", val_data)
