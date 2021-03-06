from keras.models import load_model
from gensim.models import Word2Vec
from util import make_data
from sklearn.model_selection import train_test_split
from util import make_test_data
from sklearn.metrics.scorer import accuracy_score


def get_protein_vector(sen_words: list, model: Word2Vec):
    '''
    1:Representing sentence vector.
    :param sen_words:
    :return: sentence vector
    '''
    protein_vec = 0
    counter = 0
    for aac in sen_words:
        try:
            w2v = model.wv.word_vec(aac)
            counter = counter + 1
        except KeyError:
            w2v = 0
        protein_vec = protein_vec + w2v
    protein_vec = protein_vec / counter

    return protein_vec


'''Predict lot of sample'''


def predict_lot_of_sample():
    '''Loading corpus'''
    max_feature = 2048
    x, y = make_data()

    # expected input data shape: (batch_size, timesteps, data_dim)
    x = x.reshape(-1, 1, max_feature)
    # split data
    _, x_test, _, y_test = train_test_split(x, y, test_size=0.1, random_state=64)
    model = load_model('model/patience_30_human2_model_windows_size_3')
    pre = model.predict(x_test, verbose=1)
    predict = []
    with open('new_data/protein_predict_probability_windows_size_3', mode='w', encoding='utf-8') as f1, \
            open('new_data/protein_predict_class_windows_size_3', mode='w', encoding="utf-8") as f2, \
            open('new_data/protein_label_windows_size_3', mode='w', encoding='utf-8') as f3:
        for p in pre:
            f1.write(str(p[0]) + '\n')  # The model predict probability.
            if p[0] >= 0.5:  # The model predict class.
                f2.write('1' + '\n')
                predict.append(1)
            else:
                f2.write('0' + '\n')
                predict.append(0)
        for label in y_test:  #
            f3.write(str(label[0]) + '\n')
        acc = accuracy_score(y_test, predict)
        print(acc)


def predict_test_dataset():
    max_feature = 2048
    dataset = 'Celeg'
    x, y = make_test_data()
    x = x.reshape(-1, 1, max_feature)
    # model = load_model('model/patience_30_human2_model_20000unigram')
    model = load_model('model/patience_70_protein_model')
    pre = model.predict(x, verbose=1)
    predict = []
    with open('predict_test_data/' + dataset + '_predict_probability', mode='w', encoding='utf-8') as f1, \
            open('predict_test_data/' + dataset + '_predict_class', mode='w', encoding="utf-8") as f2, \
            open('predict_test_data/' + dataset + '_label', mode='w', encoding='utf-8') as f3:
        for p in pre:
            f1.write(str(p[0]) + '\n')  # The model predict probability.
            if p[0] >= 0.5:  # The model predict class.
                f2.write('1' + '\n')
                predict.append(1)
            else:
                f2.write('0' + '\n')
                predict.append(0)
        for label in y:  #
            f3.write(str(label[0]) + '\n')
        acc = accuracy_score(y, predict)
        print(acc)


if __name__ == '__main__':
    # predict_test_dataset()
    predict_lot_of_sample()
