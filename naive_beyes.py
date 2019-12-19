import numpy as np
from functools import reduce


def load_dataset():
    words_list = []
    class_vec = []

    with open('all_email.txt', 'r', encoding='utf-8') as all_email:
        line = all_email.readline()
        while line:
            words = line.split(' ')
            words_list.append(words)
            line = all_email.readline()

    with open('label.txt', 'r') as label:
        line = label.readline()
        while line:
            class_vec.append(int(line))
            line = label.readline()

    print('load complete')
    return words_list, class_vec


def doc2VecList(doc_list):
    a = list(reduce(lambda x, y: set(x) | set(y), doc_list))
    return a


def words2Vec(vec_list, input_words):
    result_vec = [0] * len(vec_list)
    for word in input_words:
        if word in vec_list:
            result_vec[vec_list.index(word)] += 1
    return np.array(result_vec)


def train(train_matrix, train_labels):
    # m-estimate
    # nc = 0
    # l = n = sum
    # p = 1 / l

    num_labels = len(train_labels)
    num_words = len(train_matrix[0])
    p0_num = np.ones(num_words)
    p1_num = np.ones(num_words)
    p0_words = 2.0
    p1_words = 2.0

    # p0_words = 0.0
    # p1_words = 0.0

    for i in range(num_labels):
        if train_labels[i] == 1:
            p1_num += train_matrix[i]
            p1_words += sum(train_matrix[i])
        else:
            p0_num += train_matrix[i]
            p0_words += sum(train_matrix[i])
    p0_vec = np.log(p0_num / p0_words)
    p1_vec = np.log(p1_num / p1_words)

    # p0_vec = np.log(p0_num / p0_words * 2)
    # p1_vec = np.log(p1_num / p1_words * 2)

    p_class1 = sum(train_labels) / float(num_labels)
    return p0_vec, p1_vec, p_class1


def classify(test_vec, p0_vec, p1_vec, p_class1):
    p1 = sum(test_vec * p1_vec) + np.log(p_class1)
    p0 = sum(test_vec * p0_vec) + np.log(1 - p_class1)
    if p0 > p1:
        return 0
    return 1


def evaluate():
    # import dataset
    doc_list, class_vec = load_dataset()

    # extract 1% of the dataset
    train_list = doc_list[:int(len(doc_list)*0.01)]
    test_list = doc_list[int(len(doc_list)*0.01)+1:int(len(doc_list)*0.02)+1]
    train_label = class_vec[:int(len(doc_list)*0.01)]
    test_label = class_vec[int(len(doc_list)*0.01)+1:int(len(doc_list)*0.02)+1]

    # convert to vector list
    train_words_vec = doc2VecList(train_list)
    print('convert finished')

    # convert to matrix
    train_mat = list(map(lambda x: words2Vec(train_words_vec, x), train_list))
    print('matrix finished')

    # train
    p0_v, p1_v, p_class1 = train(train_mat, train_label)
    print('train finished')

    # calculate accuracy
    accuracy = 0
    for i in range(len(test_list)):
        test_vec = words2Vec(train_words_vec, test_list[i])

        # 0 for normal, 1 for dump
        test_classify = classify(test_vec, p0_v, p1_v, p_class1)
        if test_classify == test_label[i]:
            accuracy += 1
    print(accuracy / len(test_list))


if __name__ == '__main__':
    evaluate()
