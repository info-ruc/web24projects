from flask import Flask, render_template, request
from icecream import ic
from transformers import DebertaTokenizer
from api import check_text_writen_by_LLM
import json
import torch

app = Flask(__name__)
tokenizer = DebertaTokenizer.from_pretrained('microsoft/deberta-base')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = torch.load('./src/deberta.pth')
model = model.to(device) 
model.eval()

@app.route('/', methods=["GET"])
def index():
    return render_template('index.html')

@app.route('/check', methods=["POST"])
def check():
    data = request.get_json()
    text = data['text']
    ic(text)
    result = check_text_writen_by_LLM(text, tokenizer, model, device)
    ic(result)
    result = json.dumps(result)
    return json.dumps(result)
    
if __name__ == '__main__':
    app.run()