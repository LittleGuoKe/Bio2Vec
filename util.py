from gensim.models import Word2Vec
import numpy as np


def get_sentence_vector(sen_words: list, model: Word2Vec):
    '''
    1:Representing sentence vector.
    :param sen_words:
    :return: sentence vector
    '''
    sen_vec = 0
    counter = 0
    for word in sen_words:
        try:
            w2v = model.wv.word_vec(word)
            counter = counter + 1
        except KeyError:
            w2v = 0
        sen_vec = sen_vec + w2v
    if counter == 0:
        sen_vec = np.zeros([1024, ])  # 极少数蛋白质序列中的氨基酸没有词向量，因此设置为零
        print('分母为零：', '===============')
    else:
        sen_vec = sen_vec / counter
    return sen_vec


def make_data():
    '''
    step:
    2.Generating train data set
    :return: x,y. x shape:[num,dim],y shape:[num,1]
    '''

    # model = Word2Vec.load('new_data/w2v_human2_20000unigram')
    model = Word2Vec.load('new_data/w2v_human2_windows_size_3')
    temp_x = []
    # with open('new_data/tokenize_Protein_Pa_20000unigram.txt', encoding='utf-8', mode='r') as ta:
    #     line_num = len(ta.readlines())
    # with open('new_data/tokenize_Protein_Pa_20000unigram.txt', encoding='utf-8', mode='r') as ta, \
    #         open('new_data/tokenize_Protein_Na_20000unigram.txt', encoding='utf-8', mode='r') as ta1, \
    #         open('new_data/tokenize_Protein_Pb_20000unigram.txt', encoding='utf-8', mode='r') as tb, \
    #         open('new_data/tokenize_Protein_Nb_20000unigram.txt', encoding='utf-8', mode='r') as tb1:

    with open('new_data/tokenize_Protein_Pa_windows_size_3.txt', encoding='utf-8', mode='r') as ta:
        line_num = len(ta.readlines())
    with open('new_data/tokenize_Protein_Pa_windows_size_3.txt', encoding='utf-8', mode='r') as ta, \
            open('new_data/tokenize_Protein_Na_windows_size_3.txt', encoding='utf-8', mode='r') as ta1, \
            open('new_data/tokenize_Protein_Pb_windows_size_3.txt', encoding='utf-8', mode='r') as tb, \
            open('new_data/tokenize_Protein_Nb_windows_size_3.txt', encoding='utf-8', mode='r') as tb1:
        for _ in range(line_num):
            ta_p = ta.readline()
            ta_p_vector = get_sentence_vector(ta_p.strip().split(), model)
            tb_p = tb.readline()
            tb_p_vector = get_sentence_vector(tb_p.strip().split(), model)
            ab_vector = np.concatenate((ta_p_vector, tb_p_vector))
            temp_x.append(ab_vector)
        positive_num = len(temp_x)

        for k in range(line_num):
            ta1_p = ta1.readline()
            ta1_p_vector = get_sentence_vector(ta1_p.strip().split(), model)
            tb1_p = tb1.readline()
            # print(k)
            tb1_p_vector = get_sentence_vector(tb1_p.strip().split(), model)
            a1b1_vector = np.concatenate((ta1_p_vector, tb1_p_vector))
            temp_x.append(a1b1_vector)
        negative_num = len(temp_x) - positive_num

        positive_y = np.ones([positive_num, 1])
        negative_y = np.zeros([negative_num, 1])
        y = np.concatenate((positive_y, negative_y))
        x = np.array(temp_x)

    return x, y


def make_test_data():
    # model = Word2Vec.load('new_data/w2v_human2_20000unigram')
    model = Word2Vec.load('data/w2v_skip_gram_1024_dims_5_windows_protein_200000unigram')
    temp_x = []
    with open('predict_test_data/tokenize_Celeg_protein_A_20000unigram.txt', encoding='utf-8', mode='r') as ta:
        line_num = len(ta.readlines())
    with open('predict_test_data/tokenize_Celeg_protein_A_20000unigram.txt', encoding='utf-8', mode='r') as ta, \
            open('predict_test_data/tokenize_Celeg_protein_B_20000unigram.txt', encoding='utf-8', mode='r') as ta1:
        for _ in range(line_num):
            ta_p = ta.readline()
            ta_p_vector = get_sentence_vector(ta_p.strip().split(), model)
            tb_p = ta1.readline()
            tb_p_vector = get_sentence_vector(tb_p.strip().split(), model)
            ab_vector = np.concatenate((ta_p_vector, tb_p_vector))
            temp_x.append(ab_vector)
        positive_num = len(temp_x)
        positive_y = np.ones([positive_num, 1])
        y = positive_y
        x = np.array(temp_x)

    return x, y


if __name__ == '__main__':
    x, y = make_data()
    # print(x.shape)
    # print(y.shape)
