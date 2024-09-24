import pandas as pd
from datasets import load_dataset
from setfit import SetFitModel, SetFitTrainer, sample_dataset

# Load SST-2 dataset
dataset = load_dataset("sst2")

# Sample the dataset with a smaller number of samples
train_sampled_dataset = sample_dataset(dataset["train"], label_column="label", num_samples=100)  # Adjust this value as needed
eval_sampled_dataset = dataset["validation"].select(range(100))
test_sampled_dataset = dataset["validation"].select(range(100, len(dataset["validation"])))

# Initialize the model
model = SetFitModel.from_pretrained("sentence-transformers/paraphrase-albert-small-v2")

# Create the trainer with reduced batch size
trainer = SetFitTrainer(
    model=model,
    train_dataset=train_sampled_dataset,
    eval_dataset=eval_sampled_dataset,
    metric="accuracy",
    batch_size=16,  # Adjust the batch size as needed to reduce memory usage
    column_mapping={"sentence": "text", "label": "label"},
)

# Train the model
trainer.train()

# Evaluate the model
metrics = trainer.evaluate()
print(metrics)

# Push the model to Hugging Face Hub
trainer.push_to_hub("tomaarsen/setfit-paraphrase-mpnet-base-v2-sst2")

# Load the model from the Hub
model = SetFitModel.from_pretrained("tomaarsen/setfit-paraphrase-mpnet-base-v2-sst2")

# Make predictions with the model
preds = model.predict(["i loved the spiderman movie!", "pineapple on pizza is the worst 🤮"])
print(preds)
