import re
import datasets
import requests
from datasets import Dataset, Image
import evaluate
import numpy as np
import pandas as pd
import torch
import json
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer, \
    DataCollatorWithPadding
from transformers import AutoModelForTokenClassification
from transformers import AutoImageProcessor, AutoModelForImageClassification
from transformers import YolosImageProcessor, YolosForObjectDetection, AutoModelForObjectDetection
from transformers import MarianMTModel, MarianTokenizer
from transformers import VisionEncoderDecoderModel, ViTImageProcessor
from sentence_transformers import SentenceTransformer, util, models
from PIL import Image as _Image
from PIL import ImageDraw

MODELS_ROOT = 'E:/RUC/olml-2024-01-10/models'
DATASET_ROOT = 'E:/RUC/olml-2024-01-10/dataset'
METRIC_PATH = 'E:/RUC/olml-2024-01-10/metric/accuracy.py'


class DNNModel:
    def __init__(self, model_record):
        self.name = model_record[0]
        self.dataset_name = model_record[1]
        self.model_path = model_record[2]
        """
        add load mark to show where the model currently be
        1 for GPU
        0 for CPU
        """
        self.load = 0
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
        self.model.to('cuda')
        self.load = 1

    def load_model(self):
        self.model.to('cuda')
        self.load = 1

    def unload_model(self):
        self.model.to('cpu')
        self.load = 0

    def compute(self, op_input, attr_values):
        """
        add transfer_to_gpu for NOT pre_load case
        """
        if self.load != 1:
            self.load_model()
        # TODO: add op_input to attr_values
        if op_input is None:
            results = []
            for attr_value in attr_values:
                inputs = self.tokenizer(attr_value, return_tensors='pt', truncation=True, padding=True, max_length=128)
                inputs.to('cuda')
                with torch.no_grad():
                    logits = self.model(**inputs).logits
                result = logits.softmax(dim=1).squeeze().tolist()
                results.append(result[0])
            return results
        else:
            # dataset_lines = ['text_a\ttext_b']
            # here op_input is added to every attr_value for calculation
            results = []
            for attr_value in attr_values:
                inputs = self.tokenizer(op_input + '\t' + attr_value, return_tensors='pt', truncation=True,
                                        padding=True, max_length=128)
                inputs.to('cuda')
                with torch.no_grad():
                    logits = self.model(**inputs).logits
                result = logits.softmax(dim=1).squeeze().tolist()
                results.append(result[0])
            return results


class NERModelCH(DNNModel):
    def __init__(self, model_record):
        self.name = model_record[0]
        self.dataset_name = model_record[1]
        self.model_path = model_record[2]
        self.load = 0
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        with open("E:/RUC/olml-2024-01-10/dataset/msra_ner/label2id.json", 'r') as f:
            self.label2id = json.load(f)
            self.id2label = {v: k for k, v in self.label2id.items()}
        self.model = AutoModelForTokenClassification.from_pretrained(self.model_path, num_labels=len(self.label2id),
                                                                     id2label=self.id2label, label2id=self.label2id)
        self.load_model()

    def extract_ner(self, attrs_for_ner, results):
        ners = []
        for j in range(len(results)):
            result = results[j]
            i, start_idx, end_idx = 0, -1, len(result)
            while i < len(result):
                if result[i] != 'O' and result[i] != '':
                    start_idx = i
                    break
                i = i + 1
            while i < len(result):
                if result[i] == 'O' and result[i] != '':
                    end_idx = i
                    break
                i = i + 1
            if start_idx != -1:
                ners.append(''.join(attrs_for_ner[j].split(' ')[start_idx:end_idx]))
            else:
                ners.append('N/A')
        return ners

    def compute(self, op_input, attr_values):
        if self.load != 1:
            self.load_model()
        results = []
        attrs_for_ner = []
        for attr_value in attr_values:
            inputs = self.tokenizer(attr_value, return_tensors='pt', truncation=True, padding=True, max_length=128)
            inputs.to('cuda')
            with torch.no_grad():
                logits = self.model(**inputs).logits
            predictions = torch.argmax(logits, dim=2)
            predicted_token_class = [self.model.config.id2label[t.item()] for t in predictions[0]]
            results.append(predicted_token_class)

            attr_for_ner = []
            for char in attr_value:
                if char != ' ':
                    attr_for_ner.append(char)
            attrs_for_ner.append(' '.join(attr_for_ner))
        return self.extract_ner(attrs_for_ner, results)


class NERModelEN(NERModelCH):
    def __init__(self, model_record):
        self.name = model_record[0]
        self.dataset_name = model_record[1]
        self.model_path = model_record[2]
        self.load = 0
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.label2id = {'O': 0, 'B-PER': 1, 'I-PER': 2, 'B-ORG': 3, 'I-ORG': 4, 'B-LOC': 5, 'I-LOC': 6, 'B-MISC': 7,
                         'I-MISC': 8}
        self.id2label = {v: k for k, v in self.label2id.items()}
        self.model = AutoModelForTokenClassification.from_pretrained(self.model_path, num_labels=len(self.label2id),
                                                                     id2label=self.id2label, label2id=self.label2id)
        self.load_model()

    def compute(self, op_input, attr_values):
        if self.load != 1:
            self.load_model()
        results = []
        attrs_for_ner = []
        for attr_value in attr_values:
            inputs = self.tokenizer(attr_value, return_tensors='pt', truncation=True, padding=True, max_length=128)
            inputs.to('cuda')
            with torch.no_grad():
                logits = self.model(**inputs).logits
            predictions = torch.argmax(logits, dim=2)
            predicted_token_class = [self.model.config.id2label[t.item()] for t in predictions[0]]
            results.append(predicted_token_class)

            attr_for_ner = []
            for char in attr_value:
                if char != ' ':
                    attr_for_ner.append(char)
            attrs_for_ner.append(' '.join(attr_for_ner))
        return self.extract_ner(attrs_for_ner, results)


class SentenceSimModel(DNNModel):
    def __init__(self, model_record):
        self.name = model_record[0]
        self.dataset_name = model_record[1]
        self.model_path = model_record[2]
        self.load = 0
        self.model = SentenceTransformer(self.model_path)
        self.load_model()

    def compute(self, op_input, attr_values):
        if self.load != 1:
            self.load_model()
        results = []
        #query_embedding = self.model.encode(op_input, convert_to_tensor=True)
        #query_embedding.to('cuda')
        #embeddings = self.model.encode(attr_values)
        #embeddings.to('cuda')
        #results = util.dot_score(query_embedding, embeddings)
        #print("Similarity:", results)

        for attr_value in attr_values:
            sentences = [op_input, attr_value]
            embeddings = self.model.encode(sentences, convert_to_tensor=True)
            embeddings.to('cuda')
            distance = util.dot_score(embeddings[0], embeddings[1]).tolist()
            results.append(distance[0][0])

        return results


class TransCH2ENModel(DNNModel):
    def __init__(self, model_record):
        self.name = model_record[0]
        self.dataset_name = model_record[1]
        self.model_path = model_record[2]
        """
        add load mark to show where the model currently be
        1 for GPU
        0 for CPU
        """
        self.load = 0
        self.tokenizer = MarianTokenizer.from_pretrained(self.model_path)
        self.model = MarianMTModel.from_pretrained(self.model_path)
        self.model.to('cuda')
        self.load = 1

    def compute(self, op_input, attr_values):
        """
        add transfer_to_gpu for NOT pre_load case
        """
        if self.load != 1:
            self.load_model()
        inputs = self.tokenizer(attr_values, return_tensors="pt", padding=True, max_length=512, truncation=True)
        inputs.to('cuda')
        translated = self.model.generate(**inputs, max_new_tokens=40)
        results = [self.tokenizer.decode(t, skip_special_tokens=True) for t in translated]
        return results


class TransEN2CHModel(TransCH2ENModel):
    def __init__(self, model_record):
        super().__init__(model_record)


class TransEN2FRModel(TransCH2ENModel):
    def __init__(self, model_record):
        super().__init__(model_record)


class TransFR2ENModel(TransCH2ENModel):
    def __init__(self, model_record):
        super().__init__(model_record)


class ImageClass2Model(DNNModel):
    def __init__(self, model_record):
        self.name = model_record[0]
        self.dataset_name = model_record[1]
        self.model_path = model_record[2]
        self.load = 0
        self.image_processor = AutoImageProcessor.from_pretrained(self.model_path)
        with open("E:/RUC/olml-2024-01-10/dataset/beans/label2id.json", 'r') as f:
            self.label2id = json.load(f)
            self.id2label = {v: k for k, v in self.label2id.items()}
        self.model = AutoModelForImageClassification.from_pretrained(self.model_path, num_labels=len(self.label2id),
                                                                     id2label=self.id2label, label2id=self.label2id)
        self.load_model()

    def compute(self, op_input, attr_values):
        if self.load != 1:
            self.load_model()
        results = []
        for attr_value in attr_values:
            # attr_value contains the path of an image
            dataset = Dataset.from_dict({"image": [attr_value]}).cast_column("image", Image())
            inputs = self.image_processor(dataset[0]["image"].convert("RGB"), return_tensors="pt").to("cuda")
            with torch.no_grad():
                logits = self.model(**inputs).logits
            predicted_label = logits.argmax(-1).item()
            results.append(self.model.config.id2label[predicted_label])
        return results


class ImageClass3Model(ImageClass2Model):
    def __init__(self, model_record):
        self.name = model_record[0]
        self.dataset_name = model_record[1]
        self.model_path = model_record[2]
        self.load = 0
        self.image_processor = AutoImageProcessor.from_pretrained(self.model_path)
        with open("E:/RUC/olml-2024-01-10/dataset/beans/label2id.json", 'r') as f:
            self.label2id = json.load(f)
            self.id2label = {v: k for k, v in self.label2id.items()}
        self.model = AutoModelForImageClassification.from_pretrained(self.model_path, num_labels=len(self.label2id),
                                                                     id2label=self.id2label, label2id=self.label2id)
        self.load_model()


class Image2TextModel(DNNModel):
    def __init__(self, model_record):
        self.name = model_record[0]
        self.dataset_name = model_record[1]
        self.model_path = model_record[2]
        self.load = 0

        self.model = VisionEncoderDecoderModel.from_pretrained(self.model_path)
        self.feature_extractor = ViTImageProcessor.from_pretrained(self.model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.load_model()

    def compute(self, op_input, attr_values):
        # attr_values should be the image path
        if self.load != 1:
            self.load_model()
        results = []
        gen_kwargs = {"max_length": 32, "num_beams": 4}
        for image_path in attr_values:
            _image = _Image.open(image_path)
            if _image.mode != "RGB":
                _image = _image.convert(mode="RGB")
            pixel_values = self.feature_extractor(images=[_image], return_tensors="pt").pixel_values
            pixel_values = pixel_values.to('cuda')

            output_ids = self.model.generate(pixel_values, **gen_kwargs)
            result = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
            results.append(result)
        return results


class DigitModel(ImageClass2Model):
    def __init__(self, model_record):
        self.name = model_record[0]
        self.dataset_name = model_record[1]
        self.model_path = model_record[2]
        self.load = 0
        self.image_processor = AutoImageProcessor.from_pretrained(self.model_path)
        self.label2id = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9}
        self.id2label = {v: k for k, v in self.label2id.items()}
        self.model = AutoModelForImageClassification.from_pretrained(self.model_path, num_labels=len(self.label2id),
                                                                     id2label=self.id2label, label2id=self.label2id)
        self.load_model()


class ObjectDetectionModel(DNNModel):
    def __init__(self, model_record):
        self.name = model_record[0]
        self.dataset_name = model_record[1]
        self.model_path = model_record[2]
        self.load = 0
        # self.image_processor = AutoImageProcessor.from_pretrained(self.model_path)
        # self.model = AutoModelForObjectDetection.from_pretrained(self.model_path)
        self.image_processor = YolosImageProcessor.from_pretrained(self.model_path)
        self.model = YolosForObjectDetection.from_pretrained(self.model_path)

    def compute(self, op_input, attr_values):
        self.unload_model()
        return_results = []

        for image_path in attr_values:
            image = _Image.open(image_path)

            inputs = self.image_processor(images=image, return_tensors="pt")
            outputs = self.model(**inputs)

            target_sizes = torch.tensor([image.size[::-1]])
            results = self.image_processor.post_process_object_detection(outputs, threshold=0.9, target_sizes=target_sizes)[0]

            draw = ImageDraw.Draw(image)

            for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
                box = [round(i, 2) for i in box.tolist()]
                draw.rectangle((box[0], box[1], box[2], box[3]), outline="red", width=1)
                draw.text((box[0], box[1]), self.model.config.id2label[label.item()], fill="white")
                print(
                    f"Detected {self.model.config.id2label[label.item()]} with confidence "
                    f"{round(score.item(), 3)} at location {box}"
                )

            image.save(image_path)
            return_results.append(image_path)

        return return_results


def get_model_bin_name(dataset_name, op_name):
    return MODELS_ROOT + '/' + dataset_name + '_' + op_name + '.bin'


def train_model(dataset_record, base_model_path):
    train_path = DATASET_ROOT + '/' + dataset_record[0] + '/' + dataset_record[1]
    dev_path = DATASET_ROOT + '/' + dataset_record[0] + '/' + dataset_record[2]
    test_path = DATASET_ROOT + '/' + dataset_record[0] + '/' + dataset_record[3]
    output_model_path = get_model_bin_name(dataset_record[0], dataset_record[4])

    _train = []
    with open(train_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        col_name = re.sub(r'\n', '', lines[0]).split('\t')
        for i in range(1, len(lines)):
            _line = lines[i].split('\t')
            _data = {}
            _data[col_name[0]] = int(_line[0])
            _data[col_name[1]] = _line[1]
            _train.append(_data)
    dataset_train = datasets.Dataset.from_pandas(pd.DataFrame(data=_train))

    _dev = []
    with open(dev_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        col_name = re.sub(r'\n', '', lines[0]).split('\t')
        for i in range(1, len(lines)):
            _line = lines[i].split('\t')
            _data = {}
            _data[col_name[0]] = int(_line[0])
            _data[col_name[1]] = _line[1]
            _dev.append(_data)
    dataset_dev = datasets.Dataset.from_pandas(pd.DataFrame(data=_dev))

    if dataset_record[-1] == 'SentimentClsCH' or 'SentimentClsEN':
        model = AutoModelForSequenceClassification.from_pretrained(base_model_path)
    elif dataset_record[-1] == 'NerCH' or 'NerEN':
        model = AutoModelForTokenClassification.from_pretrained(base_model_path)
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, use_fast=False)
    model.to('cuda')

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding=True, truncation=True)

    tokenized_train_sets = dataset_train.map(tokenize_function, batched=True)
    tokenized_dev_sets = dataset_dev.map(tokenize_function, batched=True)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    metric = evaluate.load(METRIC_PATH)

    # metric = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)

    training_args = TrainingArguments(
        output_dir=output_model_path,
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=2,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_sets,
        eval_dataset=tokenized_dev_sets,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        data_collator=data_collator,
    )
    trainer.train()
    trainer.save_model(output_model_path)

    return output_model_path
