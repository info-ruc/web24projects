# 定义训练的函数
import torch
from icecream import ic
def train(model, dataloader, tokenizer, optimizer, criterion, device):
    model.train()
    epoch_loss = 0
    epoch_acc = 0
    for i, batch in enumerate(dataloader):
        # 标签形状为 (batch_size, 1)
        label = torch.tensor(batch["label"]).to(device)
        text = batch["text"]

        # tokenized_text 包括 input_ids， token_type_ids， attention_mask
        tokenized_text = tokenizer(text, max_length=200, add_special_tokens=True, truncation=True, padding=True, return_tensors="pt")
        tokenized_text = tokenized_text.to(device)
        # 梯度清零
        optimizer.zero_grad()
        # ic(label)
        # ic(text)
        #output: (loss), logits, (hidden_states), (attentions)
        output = model(**tokenized_text, labels=label)
        # ic(output)
        # y_pred_prob = logits : [batch_size, num_labels]
        y_pred_prob = output[1]
        # ic(y_pred_prob)
        y_pred_label = y_pred_prob.argmax(dim=1)
        # ic(y_pred_label)
        # 计算loss
        # 这个 loss 和 output[0] 是一样的
        # ic(y_pred_prob.view(-1, 4))
        # ic(label.view(-1))
        loss = criterion(y_pred_prob.view(-1, 4), label.view(-1))
        
        # 计算acc
        acc = ((y_pred_label == label.view(-1)).sum()).item()

        # 反向传播
        loss.backward()
        optimizer.step()

        # epoch 中的 loss 和 acc 累加
        # loss 每次是一个 batch 的平均 loss
        epoch_loss += loss.item()
        # acc 是一个 batch 的 acc 总和
        epoch_acc += acc
        if i % 200 == 0:
            print("current loss:", epoch_loss / (i+1), "\t", "current acc:", epoch_acc / ((i+1)*len(label)))
        print("total batchs: ", len(dataloader)," now batch number: ", i)
    # len(dataloader) 表示有多少个 batch，len(dataloader.dataset.dataset) 表示样本数量
    return epoch_loss / len(dataloader), epoch_acc / len(dataloader.dataset.dataset)

def evaluate(model, iterator, tokenizer, device):
    model.eval()
    epoch_loss = 0
    epoch_acc = 0
    with torch.no_grad():
        for _, batch in enumerate(iterator):
            label = torch.tensor(batch["label"]).to(device)
            text = batch["text"]
            tokenized_text = tokenizer(text, max_length=200, add_special_tokens=True, truncation=True, padding=True,
                                       return_tensors="pt")
            tokenized_text = tokenized_text.to(device)

            output = model(**tokenized_text, labels=label)
            y_pred_label = output[1].argmax(dim=1)
            loss = output[0]
            acc = ((y_pred_label == label.view(-1)).sum()).item()
            # epoch 中的 loss 和 acc 累加
            # loss 每次是一个 batch 的平均 loss
            epoch_loss += loss.item()
            # acc 是一个 batch 的 acc 总和
            epoch_acc += acc
            print("total batchs: ", len(iterator)," now batch number: ", _)
    # len(dataloader) 表示有多少个 batch，len(dataloader.dataset.dataset) 表示样本数量
    return epoch_loss / len(iterator), epoch_acc / len(iterator.dataset.dataset)