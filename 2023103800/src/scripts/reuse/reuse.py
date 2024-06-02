import sys

sys.path.append('E:/RUC/olml-2024-03-24')
import math
import jieba
from op.model import *
import random


class Reuse:
    def __init__(self, op_name, datasets, models):
        self.op_name, self.datasets, self.models = op_name, datasets, models

    def auto_labeling(self, unlabeled_data):
        if self.op_name != 'SentimentCls' and self.op_name != 'SentenceSim':
            raise Exception('auto labeling is not supported for op: %s' % op_name)
        infer_results = []
        for model in self.models:
            model_res = model.compute(None, unlabeled_data)
            infer_results.append(model_res)
            print('  finish infer from model: %s' % model.name)
        data_with_scores = []
        for i in range(len(unlabeled_data)):
            sum_value = 0
            for result in infer_results:
                sum_value = sum_value + float(result[i])
            data_with_scores.append((unlabeled_data[i], sum_value / len(infer_results)))
        # data_with_scores.sort(key = lambda x : x[-1], reverse=True)
        return data_with_scores

    def get_word_freq(self, texts):
        word_freq = {}
        for text in texts:
            tokens = jieba.cut(text)
            for token in tokens:
                if token not in word_freq.keys():
                    word_freq[token] = 1
                else:
                    word_freq[token] = word_freq[token] + 1
        return word_freq

    def compute_word_entropy(self, pos_word_freq, neg_word_freq):
        pos_word_entropy = []
        neg_word_entropy = []
        for word in pos_word_freq.keys():
            if word in neg_word_freq.keys():
                pos_freq, neg_freq = pos_word_freq[word], neg_word_freq[word]
                total = pos_freq + neg_freq
                p_pos, p_neg = pos_freq / total, neg_freq / total
                entropy = -(p_pos * math.log(p_pos, 2) + p_neg * math.log(p_neg, 2))

                if pos_freq == neg_freq:
                    pos_word_entropy.append((word, entropy))
                    neg_word_entropy.append((word, entropy))
                else:
                    word_entropy = pos_word_entropy if pos_freq > neg_freq else neg_word_entropy
                    word_entropy.append((word, entropy))
            else:
                pos_word_entropy.append((word, 0.0))

        for word in neg_word_freq.keys():
            if word not in pos_word_freq.keys():
                neg_word_entropy.append((word, 0.0))
        pos_word_entropy.sort(key=lambda x: x[-1])
        neg_word_entropy.sort(key=lambda x: x[-1])
        return pos_word_entropy, neg_word_entropy

    def compute_word_stats(self, positives, negatives):
        min_len = min(len(positives), len(negatives))
        positives.sort(key=lambda x: x[-1], reverse=True)
        negatives.sort(key=lambda x: x[-1], reverse=False)
        positives = positives[:min_len]
        negatives = negatives[:min_len]
        pos_word_freq = self.get_word_freq([x[0] for x in positives])
        neg_word_freq = self.get_word_freq([x[0] for x in negatives])
        pos_word_entropy, neg_word_entropy = self.compute_word_entropy(pos_word_freq, neg_word_freq)
        pos_word_entropy = [x for x in pos_word_entropy if x[-1] < 0.3]
        neg_word_entropy = [x for x in neg_word_entropy if x[-1] < 0.3]
        return pos_word_entropy, neg_word_entropy

    def get_pos_and_neg_by_threshold(self, data_with_scores):
        positives = [x for x in data_with_scores if x[-1] >= 0.90]
        negatives = [x for x in data_with_scores if x[-1] <= 0.10]
        return positives, negatives

    def select_equal_samples(self, positives, negatives, select_num):
        positives.sort(key=lambda x: x[-1], reverse=True)
        negatives.sort(key=lambda x: x[-1], reverse=False)
        min_length = min([len(positives), len(negatives), select_num])
        res = positives[:min_length] + negatives[:min_length]
        random.shuffle(res)
        return res

    def equal_select_for_each_cls(self, data_with_scores, select_num):
        positives, negatives = self.get_pos_and_neg_by_threshold(data_with_scores)
        return self.select_equal_samples(positives, negatives, select_num)

    def recompute_score(self, data_with_scores, pos_word_entropy, neg_word_entropy):
        result = []
        pos_words = [x[0] for x in pos_word_entropy]
        neg_words = [x[0] for x in neg_word_entropy]
        for item in data_with_scores:
            pos_cnt, neg_cnt = 0, 0
            words = jieba.cut(item[0])
            for word in words:
                # if word in pos_words and word in neg_words:
                if word in pos_words:
                    pos_cnt = pos_cnt + 1
                if word in neg_words:
                    neg_cnt = neg_cnt + 1
            if pos_cnt > 0 and neg_cnt > 0:
                if item[1] >= 0.5:
                    result.append((item[0], 1.0))
                else:
                    result.append((item[0], 0.0))
            else:
                result.append(item)
        return result

    def effective_select(self, data_with_scores, select_num):
        positives, negatives = self.get_pos_and_neg_by_threshold(data_with_scores)
        positives = [x for x in positives if len(x[0]) > 5]
        negatives = [x for x in negatives if len(x[0]) > 5]
        pos_word_entropy, neg_word_entropy = self.compute_word_stats(positives, negatives)
        positives = self.recompute_score(positives, pos_word_entropy, neg_word_entropy)
        negatives = self.recompute_score(negatives, pos_word_entropy, neg_word_entropy)
        # print("negatives:###########################")
        # print(negatives)
        return self.select_equal_samples(positives, negatives, select_num)


if __name__ == "__main__":
    hotel_record = tuple(
        'hotel_review	hotel	E:/RUC/olml-2022-11-02/olml/models/hotel_SentimentCls.bin	SentimentCls	train.tsv'.split(
            '\t'))
    movie_record = tuple(
        'movie_review	movie	E:/RUC/olml-2022-11-02/olml/models/movie_SentimentCls.bin	SentimentCls	train.tsv'.split(
            '\t'))
    hotel_model = DNNModel(hotel_record)
    hotel_model.load()
    movie_model = DNNModel(movie_record)
    movie_model.load()
    reuse_algo = Reuse('SentimentCls', None, [hotel_model, movie_model])

    unlabeled_data = [line.strip() for line in
                      open('E:/RUC/olml-2022-11-02/olml/table/table_data/book_review.tsv').readlines()]
    data_with_scores = reuse_algo.auto_labeling(unlabeled_data)
    # positives, negatives = reuse_algo.get_pos_and_neg_by_threshold(data_with_scores)
    # pos_entropy, neg_entropy = reuse_algo.compute_word_stats(positives, negatives)
    # reuse_algo.recompute_score(positives, pos_entropy, neg_entropy)
    # reuse_algo.recompute_score(negatives, pos_entropy, neg_entropy)
    selected = reuse_algo.effective_select(data_with_scores, 144)
    print(len(selected))
    print(selected)
