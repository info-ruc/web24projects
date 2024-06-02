from data_process import *
from word_vector import *
import numpy as np

# shuffle_txt_data('data/raw_data/sohu_test.txt', 'data/raw_data/shuffled_sohu_test.txt')
# print('done')
#
# split_words('data/shuffled_sohu_train.txt', 'data/split_train_data.txt')
#
# print('done')
# split_data = open('data/split_train_data.txt', 'r', encoding='utf-8')
#
# model = build_model(file_to_sentences(split_data))
# print('done')

# model = get_vector_model(model_path)
#
# vector = get_vector_of_text(model, '帅哥')
#
# print(vector)

package_news_data('data/raw_data/shuffled_sohu_test.txt', 'data/test_data/')
print('done')

# labels, words = get_lw_sample('data/sample-0.pth')
#
# lable_ = np.array(labels)
# word_ = np.array(words)
#
# print(lable_.shape)
# print(word_.shape)
#
# print(lable_)
# print(word_)

# m = get_vector_model('model/word2vec/w2v_model.model')
#
# vw = get_vector_weight(m)
#
# print(np.array(vw).shape)