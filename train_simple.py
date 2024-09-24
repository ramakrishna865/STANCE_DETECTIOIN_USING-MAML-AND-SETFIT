import os
import torch
from transformers import BertTokenizer, BertForSequenceClassification, AdamW

def train_and_save_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load tokenizer and model
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
    model.to(device)

    # Dummy optimizer and training step
    optimizer = AdamW(model.parameters(), lr=2e-5)
    optimizer.zero_grad()
    dummy_input = torch.tensor([tokenizer.encode("Dummy input", add_special_tokens=True)]).to(device)
    dummy_labels = torch.tensor([1]).to(device)
    outputs = model(dummy_input, labels=dummy_labels)
    loss = outputs.loss
    loss.backward()
    optimizer.step()

    # Ensure the models directory exists
    model_dir = 'models'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        print(f"Directory {model_dir} created")

    # Save the model
    model_path = os.path.join(model_dir, 'simple_model.pt')
    print(f"Saving model to {model_path}")
    torch.save(model.state_dict(), model_path)

    # Check if the model was saved
    if os.path.exists(model_path):
        print(f"Model successfully saved to {model_path}")
    else:
        print(f"Failed to save model to {model_path}")

if __name__ == "__main__":
    train_and_save_model()
