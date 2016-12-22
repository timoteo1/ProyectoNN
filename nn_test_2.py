import numpy as np
import gc 
from keras.layers import Dense, Dropout
from keras.layers.recurrent import LSTM
from keras.layers.embeddings import Embedding
from keras.models import Sequential
from dataset import readFile, vocabulary, filter_vocabulary, mapping
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import text_to_word_sequence
from keras.optimizers import SGD

if __name__ == '__main__':
    X, y = readFile('C:\\Tesis\\DataSet\\magazines\\all.review')
    print 'read'
    print X[0]
    print y[0]
    X = map(text_to_word_sequence, X)
    print 'words'
    print X[0]
    print y[0]
    v, f = vocabulary(X)
    print 'voc'
    print len(v)
    v = filter_vocabulary(v, f, total=len(X), min_value=0.01, max_value=0.8)
    print 'filter'
    print len(v)
    to, fr = mapping(v)
    gc.collect()
    X = [[to[w] for w in s] for s in X]
    gc.collect()
    X = [filter(lambda x: x > 0, s) for s in X]
    gc.collect()
    print 'to'
    print X[0]
    print y[0]
    X = pad_sequences(X, maxlen=40)
    gc.collect()
    print 'pading'
    print X[0]
    print y[0]
    print [fr[w] for w in X[0]]

    model = Sequential()
    model.add(Embedding(len(v)+1, 100))
    model.add(LSTM(50, return_sequences=True))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(50, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(15, activation='relu'))
    model.add(Dense(5, activation='softmax'))
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

    s = int(len(X)*.7)
    X_train = np.asarray(X[:s])
    y_train = np.asarray(y[:s])
    X_test = np.asarray(X[s:])
    y_test = np.asarray(y[s:])
    for i in xrange(0, 6):
        print 'Epoch {}'.format(i)
        model.fit(X_train, y_train, nb_epoch=1, batch_size=500)
        print model.evaluate(X_test, y_test, batch_size=10000)
        
    