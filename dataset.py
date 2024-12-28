# dataset.py

# Define the dataset
dataset = [
    {"text": "Hello, how are you?", "label": 0},  # Greeting
    {"text": "What is your return policy?", "label": 1},  # Question
    {"text": "I want to file a complaint.", "label": 2},  # Complaint
    {"text": "I have problems with canceling an order", "label": 3},
    {"text": "How can I find information about canceling orders?", "label": 4},
    {"text": "I need help with canceling the last order", "label": 5},
    {"text": "Could you help me cancel the last order I made?", "label": 6},
    {"text": "Problem with canceling an order I made", "label": 7},
]

def get_dataset():
    return dataset
