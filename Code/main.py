# dataset
# https://data.world/crowdflower/sentiment-analysis-in-text

import os 
import csv
import torch
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score


train_dataset = "./simpleDataset/train.txt"

distinct_words = set()
texts = []
labels = []

# Cleaning the Dataset
with open(train_dataset, 'r', encoding='utf-8') as file:
    # Let's read the csv, shall we
    csv_reader = csv.reader(file, delimiter=';')
    for row in csv_reader:
        # Get the last value after the last semicolon
        last_value = row[-1].strip()
        
        # Split the last value into words
        words = last_value.split()

        # Add distinct words to the set
        distinct_words.update(words)

# Print the distinct words after the last semicolon
print("Distinct words after the last semicolon:", distinct_words)

print("Assigning labels")  
# 0: joy, 1: sadness, 2: anger, 3: surprise, 4: fear, 5: love

with open(train_dataset, 'r', encoding='utf-8') as file:
    for entry in file:
        parts = entry.split(';')
        if len(parts) == 2: 
            text, label = parts
            texts.append(text.strip())
            

            sentiment_mapping = {'joy': 0, 'sadness': 1, 'anger': 2, 'surprise': 3, 'fear': 4, 'love': 5}
            labels.append(sentiment_mapping.get(label.strip().lower(), -1))

print("Texts:", texts)
print("Labels:", labels)

# 3 Preprocessing

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
tokenized_texts = [tokenizer.encode(text, add_special_tokens=True) for text in texts]





# Step 4: Create PyTorch Dataset
class SentimentDataset(Dataset):
    def __init__(self, tokenized_texts, labels):
        self.tokenized_texts = tokenized_texts
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'input_ids': torch.tensor(self.tokenized_texts[idx]),
            'label': torch.tensor(self.labels[idx])
        }

# Step 5: Define Model
# model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=6) 
# Step 6: Fine-tuning
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=6)  # Adjust num_labels based on your sentiment classes

# Set up training parameters
epochs = 3
batch_size = 8
learning_rate = 2e-5

# Define optimizer and loss function
optimizer = AdamW(model.parameters(), lr=learning_rate)
loss_function = torch.nn.CrossEntropyLoss()

# Step 7: Training Loop
train_dataset = SentimentDataset(tokenized_texts, labels)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

for epoch in range(epochs):
    model.train()
    total_loss = 0.0
    
    for batch in train_dataloader:
        input_ids = batch['input_ids'].to(device)
        labels = batch['label'].to(device)

        # Forward pass
        outputs = model(input_ids)
        logits = outputs.logits
        
        # Calculate loss
        loss = loss_function(logits, labels)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    average_loss = total_loss / len(train_dataloader)
    print(f"Epoch {epoch + 1}/{epochs}, Average Loss: {average_loss}")

# Step 8: Evaluation
# Assuming you have a validation dataset (val_texts, val_labels)

# val
# val_labels = 
tokenized_texts = [tokenizer.encode(text, add_special_tokens=True) for text in val_texts]



model.eval()
val_dataset = SentimentDataset(tokenized_val_texts, val_labels)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size)

all_predictions = []
all_labels = []

with torch.no_grad():
    for val_batch in val_dataloader:
        input_ids_val = val_batch['input_ids'].to(device)
        labels_val = val_batch['label'].to(device)

        val_outputs = model(input_ids_val)
        val_logits = val_outputs.logits
        
        predictions = torch.argmax(val_logits, dim=-1).cpu().numpy()
        all_predictions.extend(predictions)
        all_labels.extend(labels_val.cpu().numpy())

# Calculate accuracy
accuracy = accuracy_score(all_labels, all_predictions)
print(f"Validation Accuracy: {accuracy * 100:.2f}%")




# do the same thing, with test dataset 


# calcul accuracy

# Step 9: Save Model
model.save_pretrained('modelResult/')