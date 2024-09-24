import torch
from transformers import BertTokenizer, BertForSequenceClassification
from datasets import load_dataset
from torch.utils.data import DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import pandas as pd
import os

def evaluate(experiment_id=1):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model_path = f'models/MergedData_fine_tuned_model_{experiment_id}.pt'
    if not os.path.exists(model_path):
        print(f"Model file {model_path} does not exist.")
        return

    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    dataset = load_dataset('csv', data_files={'test': 'data/bernie_test.csv'})
    def tokenize_function(examples):
        return tokenizer(examples['Tweet'], padding='max_length', truncation=True)

    print("Tokenizing datasets...")
    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    tokenized_datasets.set_format(type='torch', columns=['input_ids', 'attention_mask', 'Stance'])

    test_loader = DataLoader(tokenized_datasets['test'], batch_size=32, shuffle=False)

    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['Stance'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
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

    results = {
        'Experiment ID': experiment_id,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1
    }

    results_df = pd.DataFrame([results])
    results_dir = 'results'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
        print(f"Directory {results_dir} created")

    results_df.to_csv(f'{results_dir}/evaluation_results_bernie_Deberta.csv', index=False)
    print(f"Evaluation results saved to {results_dir}/evaluation_results_{experiment_id}.csv")

if __name__ == "__main__":
    evaluate(experiment_id=1)
    evaluate(experiment_id=2)
