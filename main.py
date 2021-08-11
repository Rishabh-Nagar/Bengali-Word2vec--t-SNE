import nltk
import numpy as np
import matplotlib.pyplot as plt
from gensim.models import Word2Vec
from sklearn.manifold import TSNE
import sys
import pandas as pd
import re
from nltk.tokenize import word_tokenize
from indicnlp import common
from nltk.corpus import stopwords
from indicnlp.tokenize import indic_tokenize
from indicnlp.normalize.indic_normalize import IndicNormalizerFactory

nltk.download('stopwords')
stopwords.words('english')
nltk.download('punkt')
# The path to the local git repo for Indic NLP library
INDIC_NLP_LIB_HOME = r"indic_nlp_library"
# The path to the local git repo for Indic NLP Resources
INDIC_NLP_RESOURCES = r"indic_nlp_resources"
# Add library to Python path
sys.path.append(r'{}\src'.format(INDIC_NLP_LIB_HOME))
# Set environment variable for resources folder
common.set_resources_path(INDIC_NLP_RESOURCES)


class scanta:

    def sw(self, stop_df):
        stop_words = []
        stop_words = list(stop_df[0])
        # print(stop_words)
        stop_words.append('!')
        stop_words.append('-')
        stpwrd = nltk.corpus.stopwords.words('english')
        stpwrd.extend(stop_words)
        return stpwrd

    def cleaning(self, text, stpwrd):
        text_tokens = word_tokenize(text)
        removing_custom_words = [words for words in text_tokens if not words in stpwrd]
        new_sent = " ".join(removing_custom_words)
        return new_sent

    def punctuation(self, t1):
        whitespace = re.compile(u"[\s\u0020\u00a0\u1680\u180e\u202f\u205f\u3000\u2000-\u200a]+", re.UNICODE)
        bangla_fullstop = u"\u0964"
        punctSeq = u"['\"“”‘’]+|[.?!,…]+|[:;]+"
        punc = u"[(),$%^&*+={}\[\]:\"|\'\~`<>/,¦!?½£¶¼©⅐⅑⅒⅓⅔⅕⅖⅗⅘⅙⅚⅛⅜⅝⅞⅟↉¤¿º;-]+"
        t1 = whitespace.sub(" ", t1).strip()
        t1 = re.sub(punctSeq, " ", t1)
        t1 = re.sub(bangla_fullstop, " ", t1)
        t1 = re.sub(punc, " ", t1)
        return t1

    def replace_num(self, one):
        one = one.replace("০", "0")
        one = one.replace("১", "1")
        one = one.replace("২", "2")
        one = one.replace("৩", "3")
        one = one.replace("৪", "4")
        one = one.replace("৫", "5")
        one = one.replace("৬", "6")
        one = one.replace("৭", "7")
        one = one.replace("৮", "8")
        one = one.replace("৯", "9")
        return one

    def non_bengali(self, a):
        a = "".join(i for i in a if i in [".", "।"] or 2432 <= ord(i) <= 2559 or ord(i) == 32)
        a = re.sub(' +', ' ', a)
        return a

    def basic_clean(self, text):
        remove_nuktas = False
        factory = IndicNormalizerFactory()
        normalizer = factory.get_normalizer("bn")
        output_text = normalizer.normalize(text)
        words = indic_tokenize.trivial_tokenize(text, lang='bn')
        return words

    def make_model(self, values):
        return Word2Vec(values, vector_size=100, window=5, min_count=5, workers=6, sg=0)

    def display_closestwords_tsnescatterplot(self, model, word, size):

        arr = np.empty((0, size), dtype='f')
        word_labels = [word]
        close_words = model.wv.similar_by_word(word)
        arr = np.append(arr, np.array([model.wv[word]]), axis=0)
        for wrd_score in close_words:
            wrd_vector = model.wv[wrd_score[0]]
            word_labels.append(wrd_score[0])
            arr = np.append(arr, np.array([wrd_vector]), axis=0)

        tsne = TSNE(n_components=2, random_state=0)
        np.set_printoptions(suppress=True)
        Y = tsne.fit_transform(arr)
        x_coords = Y[:, 0]
        y_coords = Y[:, 1]
        plt.scatter(x_coords, y_coords)
        for label, x, y in zip(word_labels, x_coords, y_coords):
            plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points')
        plt.xlim(x_coords.min() + 0.00005, x_coords.max() + 0.00005)
        plt.ylim(y_coords.min() + 0.00005, y_coords.max() + 0.00005)
        plt.title('T-SNE')
        plt.savefig('tsne.jpg')
        plt.show()


scanta = scanta()


def main():
    data = pd.read_csv('bengali_hate_v2.0.csv')
    stop_df = pd.read_csv('stop.csv', sep='\n', header=None)
    # stop_words.head()

    stpwrd = scanta.sw(stop_df)

    for i in range(len(data['text'])):
        data['text'][0] = scanta.cleaning(data['text'][0], stpwrd)

    data['text'] = data['text'].apply(scanta.punctuation)

    data['text'] = data['text'].apply(scanta.replace_num)

    data['text'] = data['text'].apply(scanta.non_bengali)

    words = scanta.basic_clean(''.join(str(data['text'].tolist())))

    values = []

    size = data.shape[0]
    for i in range(0, size):
        worddd = scanta.basic_clean(data['text'][i])
        values.append(worddd)

    model = scanta.make_model(values)

    # model.wv.most_similar('খেলা', topn=5)

    filename = "embedding_word2vec.txt"

    model.wv.save_word2vec_format(filename, binary=False)

    # embeddings_index = {}
    # f = open(os.path.join('', 'embedding_word2vec.txt'))
    # for line in f:
    #     value = line.split()
    #     word = value[0]
    #     coefs = np.asarray(value[1:])
    #     embeddings_index[word] = coefs
    # f.close()

    scanta.display_closestwords_tsnescatterplot(model, 'খেলা', 100)


if __name__ == '__main__':
    main()
