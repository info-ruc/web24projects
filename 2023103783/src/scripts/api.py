import torch
from icecream import ic

def check_text_writen_by_LLM(text, tokenizer, model, device):
    
    tokenized_text = tokenizer(text, max_length=300, add_special_tokens=True, truncation=True, padding=True,
                            return_tensors="pt")
    tokenized_text = tokenized_text.to(device)
    output = model(**tokenized_text)
    probs = torch.nn.functional.softmax(output.logits, dim=-1)
    probs_list = probs.tolist()[0]
    
    largest_val = max(probs_list)
    largest_id = probs_list.index(largest_val)
    return {'world': "文字主题为全球事件的概率为：" + str(probs_list[0]), \
            'sports': '文字主题为运动赛事概率为：' + str(probs_list[1]), \
            'business': '文字主题为商业资讯的概率为：' + str(probs_list[2]), \
            'sci_tec': '文字主题为科学技术的概率为：' + str(probs_list[3]), \
            'largest': largest_id}