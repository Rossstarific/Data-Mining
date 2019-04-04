import pandas as pd
import numpy as np
import nltk
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
from gensim.corpora import Dictionary
from gensim.models.word2vec import Word2Vec
import re
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from IPython.display import display
import base64
import io
import seaborn as sns
from collections import Counter
import pickle

train_data = pd.read_csv(("TrainingSet.csv"), encoding="ISO-8859-1")
'''
def remove_ascii_words(df):
    non_ascii_words = []
    for i in range(len(df)):
        for word in df.loc[i, 'Sentence'].split(' '):
            for character in word:
                if any([ord(character) >= 128]):
                    non_ascii_words.append(character)
                    df.loc[i, 'Sentence'] = df.loc[i, 'Sentence'].replace(character, ' ')
    return non_ascii_words

def remove_some_punctuation(df):
    punctuation = []
    for i in range(len(df)):
        for word in df.loc[i,'Sentence'].split(' '):
            for character in word:
                if character == '"' or character == '(' or character == ')' or character == "'" or character == ',' or character == '/' or character == ':' or character == ';' or character == '-':
                    punctuation.append(character)
                    df.loc[i, 'Sentence'] = df.loc[i, 'Sentence'].replace(character, '')
    return punctuation


                
remove_some_punctuation(train_data)
remove_ascii_words(train_data)



def w2v_preprocessing(df):
    df['Sentence'] = df.Sentence.str.lower()
    df['document_sentences'] = df['Sentence'].apply(nltk.sent_tokenize)#nltk.sent_tokenize(df.Sentence)
    df['tokenized_sentences'] = list(map(lambda sentences: list(map(nltk.word_tokenize, sentences)), df.document_sentences))

w2v_preprocessing(train_data)'''
tokenize_sent_file = open('TokenizedSentences.txt', 'rb')
#pickle.dump(train_data.tokenized_sentences, tokenize_sent_file)
tokenized_sentences = pickle.load(tokenize_sent_file)

'''
sentences = []
for sentence_group in train_data.tokenized_sentences:
    sentences.extend(sentence_group)
'''
sent_file = open('Sentences.txt', 'rb')
#pickle.dump(sentences, sent_file)
sentences = pickle.load(sent_file)
'''
document_lengths = np.array(list(map(len, train_data.Sentence.str.split(' '))))
print("average number of words in a sentence:", np.mean(document_lengths))
print("min number of words in a sentence:", min(document_lengths))
print("max number of words in a sentence:", max(document_lengths))
'''
num_features = 200
min_word_count = 3
num_workers = 4
context = 6
downsampling = 1e-3

W2Vmodel = Word2Vec(sentences=sentences,
                    sg=1,
                    hs=0,
                    workers=num_workers,
                    size=num_features,
                    min_count=min_word_count,
                    window=context,
                    sample=downsampling,
                    negative=5,
                    iter=6)

def get_w2v_features(w2v_model, sentence_group):
    words = np.concatenate(sentence_group)
    index2word_set = set(w2v_model.wv.vocab.keys())
    featureVec = np.zeros(w2v_model.vector_size, dtype="float32")
    nwords = 0
    for word in words:
        if word in index2word_set:
            featureVec = np.add(featureVec, w2v_model[word])
            nwords += 1
        if nwords > 0:
            featureVec = np.divide(featureVec, nwords)
        return featureVec

train_data['w2v_features'] = list(map(lambda sen_group: get_w2v_features(W2Vmodel, sen_group), tokenized_sentences))

label_encoder = LabelEncoder()

label_encoder.fit(train_data.Playwright)
train_data['Playwright_id'] = label_encoder.transform(train_data.Playwright)

def get_cross_validated_model(model, X, y,  param_grid, newfile, score, nr_folds=10):
    grid_cv = GridSearchCV(model, param_grid=param_grid, n_jobs=-1, scoring=score, cv=nr_folds)
    best_model = grid_cv.fit(X, y)
    result_df = pd.DataFrame(best_model.cv_results_)
    show_columns = ['mean_test_score', 'mean_train_score', 'rank_test_score']
    for col in result_df.columns:
        if col.startswith('param_'):
            show_columns.append(col)
    newfile.write("Metric: " + score + "\n" +
                  str((result_df[show_columns].sort_values(by='rank_test_score').head())) + "\n")
    return best_model

pd.options.display.max_columns = 6
X_train_w2v = np.array(list(map(np.array, train_data.w2v_features)))
param_grid = {'penalty': ['l1', 'l2']}

LRFile = open('LogisticRegression.txt', 'a+')

lr = LogisticRegression()

get_cross_validated_model(lr, param_grid, X_train_w2v, train_data.Playwright_id, LRFile, 'accuracy')
get_cross_validated_model(lr, param_grid, X_train_w2v, train_data.Playwright_id, LRFile, 'f1_weighted')
get_cross_validated_model(lr, param_grid, X_train_w2v, train_data.Playwright_id, LRFile, 'precision_weighted')
get_cross_validated_model(lr, param_grid, X_train_w2v, train_data.Playwright_id, LRFile, 'recall_weighted')

LRFile.close()


SVMFile = open('SVM.txt', 'a+')

svm = LinearSVC()

get_cross_validated_model(svm, param_grid, X_train_w2v, train_data.Playwright_id, SVMFile, 'accuracy')
get_cross_validated_model(svm, param_grid, X_train_w2v, train_data.Playwright_id, SVMFile, 'f1_weighted')
get_cross_validated_model(svm, param_grid, X_train_w2v, train_data.Playwright_id, SVMFile, 'precision_weighted')
get_cross_validated_model(svm, param_grid, X_train_w2v, train_data.Playwright_id, SVMFile, 'recall_weighted')

SVMFile.close()



NBFile = open('NaiveBayes.txt', 'a+')

nb = MultinomialNB()

get_cross_validated_model(nb, params, X_train_w2v, train_data.Playwright_id, NBFile, 'accuracy')
get_cross_validated_model(nb, params, X_train_w2v, train_data.Playwright_id, NBFile, 'f1_weighted')
get_cross_validated_model(nb, params, X_train_w2v, train_data.Playwright_id, NBFile, 'precision_weighted')
get_cross_validated_model(nb, params, X_train_w2v, train_data.Playwright_id, NBFile, 'recall_weighted')

NBFile.close()



