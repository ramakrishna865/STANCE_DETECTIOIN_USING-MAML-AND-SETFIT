# import torch
# from torch.utils.data import DataLoader
# from datasets import load_dataset
# from transformers import BertTokenizer, BertForSequenceClassification, AdamW

# def train_maml(epochs=5, batch_size=32, learning_rate=2e-5):
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#     tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
#     model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
#     model.to(device)

#     optimizer = AdamW(model.parameters(), lr=learning_rate)
#     dataset = load_dataset('csv', data_files={'train': 'data/MergedData_train.csv', 'val': 'data/MergedData_val.csv'})

#     def tokenize_function(examples):
#         return tokenizer(examples['Tweet'], padding='max_length', truncation=True, return_tensors='pt')

#     tokenized_datasets = dataset.map(tokenize_function, batched=True)
#     tokenized_datasets.set_format(type='torch', columns=['input_ids', 'attention_mask', 'Stance'])

#     train_loader = DataLoader(tokenized_datasets['train'], batch_size=batch_size, shuffle=True)
#     val_loader = DataLoader(tokenized_datasets['val'], batch_size=batch_size)

#     for epoch in range(epochs):
#         model.train()
#         for batch in train_loader:
#             optimizer.zero_grad()
#             input_ids = batch['input_ids'].to(device)
#             attention_mask = batch['attention_mask'].to(device)
#             labels = batch['Stance'].to(device)

#             outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
#             loss = outputs.loss
#             loss.backward()
#             optimizer.step()

#         model.eval()

#         print(f"Epoch {epoch + 1}/{epochs} - Loss: {loss.item()}")

#     torch.save(model.state_dict(), 'models/MergedData_fine_tuned_model.pt')

# if __name__ == "__main__":
#     train_maml()



import os
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
# from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from transformers import DebertaTokenizer,DebertaForSequenceClassification, AdamW
from transformers import RobertaTokenizer, RobertaForSequenceClassification,AdamW
import pandas as pd

def train_maml(epochs=5, batch_size=32, learning_rate=2e-5, experiment_id=1):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    tokenizer = DebertaTokenizer.from_pretrained('roberta-base')
    model = DebertaForSequenceClassification.from_pretrained('roberta-base', num_labels=2)
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=learning_rate)
    dataset = load_dataset('csv', data_files={'train': 'data/MergedData_train.csv', 'val': 'data/MergedData_val.csv'})

    def tokenize_function(examples):
        return tokenizer(examples['Tweet'], padding='max_length', truncation=True)

    print("Tokenizing datasets...")
    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    tokenized_datasets.set_format(type='torch', columns=['input_ids', 'attention_mask', 'Stance'])

    train_loader = DataLoader(tokenized_datasets['train'], batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(tokenized_datasets['val'], batch_size=batch_size)

    results_table = []

    print("Starting training...")
    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['Stance'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_train_loss += loss.item()
            loss.backward()
            optimizer.step()

        avg_train_loss = total_train_loss / len(train_loader)

        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['Stance'].to(device)

                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_loader)

        print(f"Epoch {epoch + 1}/{epochs} - Training Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}")

        results_table.append({
            'Experiment ID': experiment_id,
            'Epoch': epoch + 1,
            'Training Loss': avg_train_loss,
            'Validation Loss': avg_val_loss
        })

    model_dir = 'models'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        print(f"Directory {model_dir} created")

    model_path = os.path.join(model_dir, f'MergedData_Roberta_fine_tuned_model.pt')
    print(f"Saving model to {model_path}")

    torch.save(model.state_dict(), model_path)

    if os.path.exists(model_path):
        print(f"Model successfully saved to {model_path}")
    else:
        print(f"Failed to save model to {model_path}")

    tokenizer.save_pretrained(model_dir)

    results_df = pd.DataFrame(results_table)
    results_dir = 'results'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
        print(f"Directory {results_dir} created")

    results_df.to_csv(f'{results_dir}/training_results_bernie_Roberta.csv', index=False)
    print(f"Training results saved to {results_dir}/training_results_{experiment_id}.csv")

if __name__ == "__main__":
    train_maml()
