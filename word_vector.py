import multiprocessing

import jieba
import jieba.analyse
import numpy
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence

stop_word_path = 'word_vec_data/cn_stopwords.txt'
model_path = 'model/word2vec/w2v_model.model'

stopwords = {}.fromkeys([line.rstrip() for line in open(stop_word_path,
                                                        encoding='utf-8')])

pos = ('n', 'nz', 'v', 'ns', 'vn', 'i', 'a', 'nt', 'b', 'vd', 't', 'ad', 'an', 'c', 'nr')


# jieba.analyse.set_stop_words(stop_words)

# 分词，返回分词列表
def word_split(sentence):
    cut = jieba.cut(sentence)
    return list(cut)


# 去除停用词
def remove_stop_words(words):
    result = []
    for word in words:
        if word not in stopwords:
            result.append(word)
    return result


# 获取前num个关键字，传进来word列表
def get_key_words(words, num):
    sentence = ''
    for word in words:
        sentence = sentence + ' ' + word
    key_words = jieba.analyse.extract_tags(sentence, topK=num, withWeight=False, allowPOS=pos)
    return list(key_words)


# 获取前num个关键字，直接传进来句子
def get_key_words_all_step(sentence, num):
    return get_key_words(remove_stop_words(word_split(sentence)), num)


# 读w2v模型
def get_vector_model(model_path):
    return Word2Vec.load(model_path)


# 向w2v中添加句子
def add_sentences(model, sentences):
    model.build_vocab(sentences, update=True)
    model.train(sentences, total_words=model.corpus_count, epochs=model.epochs)
    model_save(model)
    return model


# 新建一个w2v模型
def build_model(sentences):
    model = Word2Vec(sentences=sentences, window=1, min_count=1,
                     workers=multiprocessing.cpu_count(), sg=1)
    model_save(model)
    return model


def model_save(model):
    model.save(model_path)


def file_to_sentences(file):
    return LineSentence(file)


def words_to_sentences(words):
    return [words]


# 获取词汇的词向量, 返回类型是 ndArray, 可以.tolist()转化为list
def get_vector_of_text(model, text):
    if has_text(model, text):
        return model.wv.get_vector(text)
    return get_empty_vector()


def has_text(model, text):
    return model.wv.has_index_for(text)


def get_vector_weight(model):
    return model.wv.vectors


def get_text_of_index(model, index):
    # index_to_key 是list,key_to_index是dict
    return model.wv.index_to_key[index]


def get_index(model, text):
    if has_text(model, text):
        return model.wv.get_index(text)
    return 0


def get_empty_vector():
    v = numpy.zeros(100, dtype=float, order='C')
    return v


def train_vector(model, split_txt):
    # build_model(file_to_sentences(split_txt)) 当model没有时
    return add_sentences(model, file_to_sentences(split_txt))


def train_new_vector(split_txt):
    model = build_model(file_to_sentences(split_txt))
    return add_sentences(model, file_to_sentences(split_txt))
