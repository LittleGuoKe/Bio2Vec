from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
import datetime


def get_w2v(corpus_path):
    print("start", datetime.datetime.now())
    print('Loading data...')
    sentences = LineSentence(corpus_path)
    print('Training w2v...')
    # model = Word2Vec(sentences, window=5, size=512, sg=1, workers=multiprocessing.cpu_count(), iter=50, min_count=1)
    model = Word2Vec(sentences, window=5, size=2048, sg=1, workers=10, iter=50, min_count=1)
    print('Saving w2v model...')
    model.save('data/test_w2v_20000unigram_2048_dim')
    print("end", datetime.datetime.now())


if __name__ == '__main__':
    # get_w2v('new_data/tokenize_merge_human2_20000unigram.txt')
    get_w2v('data/tokenize_merge_protein_20000unigram.txt')
