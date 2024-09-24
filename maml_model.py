import torch
import torch.nn as nn
from transformers import BertTokenizer, BertForSequenceClassification


#Meta
class MAMLStancePrediction(nn.Module):
    def __init__(self, learning_rate_inner=0.01):
        super(MAMLStancePrediction, self).__init__()
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)
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
        self.model.load_state_dict({name: w for name, w in zip(self.model.state_dict().keys(), updated_weights)})
        logits = self(input_ids, attention_mask)
        loss = nn.CrossEntropyLoss()(logits.logits, labels)
        return loss
