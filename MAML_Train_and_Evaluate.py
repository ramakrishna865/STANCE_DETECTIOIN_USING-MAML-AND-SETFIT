import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_dataset
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from transformers import BertTokenizer, BertForSequenceClassification, RobertaTokenizer, RobertaForSequenceClassification, DebertaTokenizer, DebertaForSequenceClassification, AdamW

class MAMLStancePrediction(nn.Module):
    def __init__(self, model_name, model_class, tokenizer_class, num_labels=2, learning_rate_inner=0.01):
        super(MAMLStancePrediction, self).__init__()
        self.tokenizer = tokenizer_class.from_pretrained(model_name)
        self.model = model_class.from_pretrained(model_name, num_labels=num_labels)
        self.learning_rate_inner = learning_rate_inner

    def forward(self, input_ids, attention_mask, labels=None):
        return self.model(input_ids, attention_mask, labels=labels)

    def inner_update(self, input_ids, attention_mask, labels):
        logits = self(input_ids, attention_mask)
        loss = nn.CrossEntropyLoss()(logits.logits, labels)
        grads = torch.autograd.grad(loss, self.model.parameters(), create_graph=True)
        updated_weights = [p - self.learning_rate_inner * g for p, g in zip(self.model.parameters(), grads)]
        return updated_weights

    def outer_loss(self, input_ids, attention_mask, labels, updated_weights):
        original_weights = [param.clone() for param in self.model.parameters()]
        for param, updated_param in zip(self.model.parameters(), updated_weights):
            param.data = updated_param.data

        logits = self(input_ids, attention_mask)
        loss = nn.CrossEntropyLoss()(logits.logits, labels)

        for param, original_param in zip(self.model.parameters(), original_weights):
            param.data = original_param.data

        return loss

def train_maml(model_name, tokenizer_class, model_class, optimizer_class, epochs=5, batch_size=32, learning_rate=2e-5, learning_rate_inner=0.01):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    maml_model = MAMLStancePrediction(model_name, model_class, tokenizer_class, learning_rate_inner=learning_rate_inner)
    maml_model.to(device)

    optimizer = optimizer_class(maml_model.parameters(), lr=learning_rate)
    dataset = load_dataset('csv', data_files={'train': 'data/MergedData_train.csv', 'val': 'data/MergedData_val.csv', 'test': 'data/MergedData_test.csv'})

    def tokenize_function(examples):
        return maml_model.tokenizer(examples['Tweet'], padding='max_length', truncation=True, return_tensors='pt')

    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    tokenized_datasets.set_format(type='torch', columns=['input_ids', 'attention_mask', 'Stance'])

    train_loader = DataLoader(tokenized_datasets['train'], batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(tokenized_datasets['val'], batch_size=batch_size)
    test_loader = DataLoader(tokenized_datasets['test'], batch_size=batch_size)

    results_table = []

    for epoch in range(epochs):
        maml_model.train()
        total_train_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['Stance'].to(device)

            # Inner Update
            updated_weights = maml_model.inner_update(input_ids, attention_mask, labels)

            # Outer Update
            outer_loss = maml_model.outer_loss(input_ids, attention_mask, labels, updated_weights)
            outer_loss.backward()
            optimizer.step()
            total_train_loss += outer_loss.item()

        avg_train_loss = total_train_loss / len(train_loader)

        # Validation
        maml_model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['Stance'].to(device)

                # Calculate Validation Loss with Updated Weights
                updated_weights = maml_model.inner_update(input_ids, attention_mask, labels)
                val_loss = maml_model.outer_loss(input_ids, attention_mask, labels, updated_weights)
                total_val_loss += val_loss.item()

        avg_val_loss = total_val_loss / len(val_loader)

        print(f"Epoch {epoch + 1}/{epochs} - Training Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}")

        results_table.append({
            'Epoch': epoch + 1,
            'Training Loss': avg_train_loss,
            'Validation Loss': avg_val_loss
        })

    torch.save(maml_model.state_dict(), f'models/{model_name}_maml_fine_tuned_model.pt')

    results_df = pd.DataFrame(results_table)
    results_df.to_csv(f'results/{model_name}_maml_training_results.csv', index=False)

    # Evaluation
    maml_model.load_state_dict(torch.load(f'models/{model_name}_maml_fine_tuned_model.pt'))
    maml_model.eval()

    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['Stance'].to(device)

            outputs = maml_model(input_ids, attention_mask)
            _, predicted = torch.max(outputs.logits, 1)
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions, average='weighted')
    recall = recall_score(all_labels, all_predictions, average='weighted')
    f1 = f1_score(all_labels, all_predictions, average='weighted')

    print(f'Accuracy: {accuracy * 100:.2f}%')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1-Score: {f1:.4f}')

    evaluation_results = {
        'Model': model_name,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1
    }

    return evaluation_results

if __name__ == "__main__":
    from torch.optim import Adam, SGD, RMSprop

    experiments = [
        {
            'model_name': 'bert-base-uncased',
            'tokenizer_class': BertTokenizer,
            'model_class': BertForSequenceClassification,
            'optimizer_class': Adam,
            'epochs': 3,
            'batch_size': 16,
            'learning_rate': 3e-5,
            'learning_rate_inner': 0.01
        },
        {
            'model_name': 'roberta-base',
            'tokenizer_class': RobertaTokenizer,
            'model_class': RobertaForSequenceClassification,
            'optimizer_class': SGD,
            'epochs': 5,
            'batch_size': 32,
            'learning_rate': 1e-3,
            'learning_rate_inner': 0.01
        },
        {
            'model_name': 'microsoft/deberta-base',
            'tokenizer_class': DebertaTokenizer,
            'model_class': DebertaForSequenceClassification,
            'optimizer_class': RMSprop,
            'epochs': 4,
            'batch_size': 64,
            'learning_rate': 2e-4,
            'learning_rate_inner': 0.01
        }
    ]

    results = []
    for experiment in experiments:
        result = train_maml(**experiment)
        results.append(result)

    results_df = pd.DataFrame(results)
    results_df.to_csv('results/maml_experiment_results.csv', index=False)

