import os
import random

import torch

from word_vector import *

categories = ['娱乐', '财经', '房地产', '旅游', '科技', '体育', '健康', '教育', '汽车', '其他', '文化', '女人']

# 类别名对应的id
labels = {'娱乐': 0, '财经': 1, '房地产': 2, '旅游': 3, '科技': 4, '体育': 5,
          '健康': 6, '教育': 7, '汽车': 8, '新闻': 9, '文化': 10, '女人': 11}

# sampple-?.pth的最大序号
MAX_TRAIN_INDEX = 235
MAX_TEST_INDEX = 119

model_path = 'model/word2vec/w2v_model.model'

def shuffle_txt_data(pre_txt, path_to_save):
    # 用于将原有排序的数据打乱，方便后面训练
    encoding = 'utf-8'
    with open(pre_txt, 'r', encoding=encoding) as lines:
        # 也可以 for line in lines: line_list.append(line)
        line_list = list(lines)
        # 打乱列表顺序
        random.shuffle(line_list)
        with open(path_to_save, 'w', encoding=encoding) as file:
            for each_line in line_list:
                file.write(each_line)


def split_words(news_txt, split_txt):
    with open(news_txt, 'r', encoding='utf-8') as lines:
        with open(split_txt, 'a', encoding='utf-8') as file:
            for line in lines:
                split = line.split('\t')
                # label = split[0], 分类名
                content = split[1]
                # len(content)获取最大词汇,我想让词向量模型更全面些
                keys = get_key_words_all_step(content, len(content))
                for key in keys:
                    file.write(key + '\n')


def package_news_data(news_txt, directory):
    sample_size = 100  # 每100个新闻为一个sample
    key_size = 50  # 关键词数量
    model = get_vector_model(model_path)  # 词向量模型
    with open(news_txt, 'r', encoding='utf-8') as lines:
        sample_index = 0  # sample_序号
        news_count = 0  # sample内的新闻数量
        sample = []  # 保存sample_size个新闻
        for line in lines:
            sp = line.split('\t')
            label = labels[sp[0]]
            content = sp[1]
            # 分词
            content_key = get_key_words_all_step(content, key_size)
            key_index = [get_index(model, key) for key in content_key]
            key_index.extend([0] * (key_size - len(key_index)))  # 补0
            # 加入样本集
            sample.append((label, key_index))
            news_count += 1
            print('[sample %d ][ news %d ]' % (sample_index, news_count))
            if news_count == sample_size:
                torch.save(sample, directory + 'sample-' + str(sample_index) + '.pth')
                print('[sample %d package done]' % sample_index)
                news_count = 0
                sample.clear()
                sample_index += 1
        if news_count > 0:  # 未满sample_size
            torch.save(sample, directory + 'sample-' + str(sample_index) + '.pth')
            print('[sample %d package done]' % sample_index)


def get_lw_sample(file):
    sample = torch.load(file)
    labels = [l for l, w in sample]  # 提取label和words
    words = [w for l, w in sample]
    return labels, words


# 根据索引加载打包好的训练数据
def get_lw_of_train_sample(index):
    file = 'data/train_data/sample-' + str(index) + '.pth'
    return get_lw_sample(file)


# 根据索引加载打包好的测试数据
def get_lw_of_test_sample(index):
    file = 'data/test_data/sample-' + str(index) + '.pth'
    return get_lw_sample(file)
