import jieba
from gensim import corpora,models,similarities


def get_similar_model_index(documents, document_test):
    documents_list = []
    for document in documents:
        document_list = [word for word in jieba.cut(document)]
        documents_list.append(document_list)
    document_test_list = [word for word in jieba.cut(document_test)]

    dictionary = corpora.Dictionary(documents_list)
    corpus = [dictionary.doc2bow(item) for item in documents_list]
    document_test_vec = dictionary.doc2bow(document_test_list)
    tfidf = models.TfidfModel(corpus)
    index = similarities.SparseMatrixSimilarity(tfidf[corpus],num_features=len(dictionary.keys()))
    similarity = index[tfidf[document_test_vec]]

    result = sorted(enumerate(similarity), key= lambda item: -item[1])
    return result[0][0]
