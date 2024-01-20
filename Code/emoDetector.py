
import torch
from transformers import BertTokenizer, BertForSequenceClassification

# Load the trained model
model = BertForSequenceClassification.from_pretrained('modelResultat/config.json')

# Load the tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Example text for prediction
new_text = "I really enjoyed the movie. It was so entertaining!"


# pasii de prelucrare


# Tokenize and format the input text
tokenized_text = tokenizer.encode(new_text, add_special_tokens=True)
input_ids = torch.tensor(tokenized_text).unsqueeze(0)  # Add a batch dimension

# Make a prediction
with torch.no_grad():
    outputs = model(input_ids)

# Get the predicted class probabilities
probs = torch.nn.functional.softmax(outputs.logits, dim=-1)

# Get the predicted class index (argmax)
predicted_class = torch.argmax(probs, dim=-1).item()

# Map the predicted class index back to the sentiment label
sentiment_mapping_reverse = {0: 'joy', 1: 'sadness', 2: 'anger', 3: 'surprise', 4: 'fear', 5: 'love'}
predicted_sentiment = sentiment_mapping_reverse.get(predicted_class, 'Unknown')

# Print the results
print("Predicted Sentiment:", predicted_sentiment)
print("Predicted Class Probabilities:", probs.squeeze().tolist())
