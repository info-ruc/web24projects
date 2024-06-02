from transformers import DebertaConfig, DebertaForSequenceClassification
from transformers import DebertaTokenizer
import torch.nn as nn
from transformers import AdamW
from config import num_labels, hidden_dropout_prob, weight_decay, learning_rate, device


tokenizer = DebertaTokenizer.from_pretrained('microsoft/deberta-base')
# 加载模型
config = DebertaConfig.from_pretrained("microsoft/deberta-base", num_labels=num_labels, hidden_dropout_prob=hidden_dropout_prob)
model = DebertaForSequenceClassification.from_pretrained("microsoft/deberta-base", config=config)
model = model.to(device)

no_decay = ['bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]

optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)
criterion = nn.CrossEntropyLoss()

