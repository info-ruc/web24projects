import pandas as pd
import torch
from dataset import text_train_loader, text_valid_loader
from train import train, evaluate
from model import model, tokenizer, optimizer, criterion, device
from config import epochs
from utils import check_file_in_directory



# 开始训练和验证
for i in range(epochs):
    train_loss, train_acc = train(model, text_train_loader, tokenizer, optimizer, criterion, device)
    print("train loss: ", train_loss, "\t", "train acc:", train_acc)
    # torch.save(model, 'deberta.pth')
    # model = torch.load('model.pth')
    # train_loss, train_acc = train(model, text_train_loader2, tokenizer, optimizer, criterion, device)
    # print("train loss: ", train_loss, "\t", "train acc:", train_acc)
    torch.save(model, 'deberta.pth')
    print("Model saved.")
    valid_loss, valid_acc = evaluate(model, text_valid_loader, tokenizer, device)
    print("valid loss: ", valid_loss, "\t", "valid acc:", valid_acc)