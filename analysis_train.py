import string
import re
from numpy import array
from os import listdir
from nltk.corpus import stopwords
from collections import Counter
from keras.preprocessing.text import Tokenizer
from keras.utils.vis_utils import plot_model
from keras.models import Sequential
from keras.layers import Dense
from keras.models import load_model

# turn a doc into clean tokens
def clean_doc(doc):
    # split into tokens by white space
    tokens = doc.split()
    # prepare regex for char filtering
    re_punc = re.compile('[%s]' % re.escape(string.punctuation))
    # remove punctuation from each word
    tokens = [re_punc.sub('', w) for w in tokens]
    # remove remaining tokens that are not alphabetic
    tokens = [word for word in tokens if word.isalpha()]
    # filter out stop words
    stop_words = set(stopwords.words('english'))
    tokens = [w for w in tokens if not w in stop_words]
    # filter out short tokens
    tokens = [word for word in tokens if len(word) > 1]
    return tokens

# load doc, clean and return line of tokens
def doc_to_line(doc, vocab):    
    tokens = clean_doc(doc)
    # filter by vocab
    tokens = [w for w in tokens if w in vocab]
    return ' '.join(tokens)

def load_docs(start, end, file_name, lines, vocab):    
    file = open(file_name, "r")
    count = 0
    for line in file:
        if count >= start and count <= end:
            lines.append(doc_to_line(line,vocab))
            #print(line)
        count= count+1
    file.close()
    return lines        

def create_tokenizer(lines): 
    tokenizer = Tokenizer() 
    tokenizer.fit_on_texts(lines) 
    return tokenizer

def define_model(n_words): 
    model = Sequential() 
    model.add(Dense(50, input_shape=(n_words,), activation='relu')) 
    model.add(Dense(1, activation='sigmoid')) 
    # compile network 
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy']) 
    # summarize defined model 
    model.summary() 
    #plot_model(model, to_file='model.png', show_shapes=True) 
    return model

def predict_sentiment(comment, vocab, tokenizer, model): 
    tokens = clean_doc(comment)
    tokens = [w for w in tokens if w in vocab] 
    line = ' '.join(tokens)
    encoded = tokenizer.texts_to_matrix([line], mode='binary')
    yhat = model.predict(encoded, verbose=0) # retrieve predicted percentage and label 
    percent_pos = yhat[0,0] 
    if round(percent_pos) == 0: 
        return (1-percent_pos), 'NEGATIVE' 
    return percent_pos, 'POSITIVE'

vocab_file = open("C:\\Projects\\ml\\sentiment\\vocab.txt", "r")
vocab = vocab_file.read()
vocab = set(vocab.split())
    
train_lines = list()
test_lines = list()
train_count = 500
test_count = 100
train_lines = load_docs(0,train_count-1,"C:\\Projects\\ml\\sentiment\\promoter.txt",train_lines,vocab)
train_lines = load_docs(0,train_count-1,"C:\\Projects\\ml\\sentiment\\detractor.txt",train_lines,vocab)
y_train = [0 for _ in range(train_count)] + [1 for _ in range(train_count)] 

test_lines = load_docs(train_count,1000,"C:\\Projects\\ml\\sentiment\\promoter.txt",test_lines,vocab)
test_lines = load_docs(train_count,1000,"C:\\Projects\\ml\\sentiment\\detractor.txt",test_lines,vocab)
y_test = [0 for _ in range(test_count)] + [1 for _ in range(test_count)] 

print(len(test_lines))
print(len(train_lines))

tokenizer = create_tokenizer(train_lines)
Xtrain = tokenizer.texts_to_matrix(train_lines, mode='freq')
Xtest = tokenizer.texts_to_matrix(test_lines, mode='freq')

n_words = Xtest.shape[1] 
model = define_model(n_words) 
model.fit(Xtrain, y_train, epochs=10, verbose=2) 
loss, acc = model.evaluate(Xtest, y_test, verbose=0) 
print('Test Accuracy: %f' % (acc*100))

model.save('151_sentiment.h5')

#text = 'excellent' 
#percent, sentiment = predict_sentiment(text, vocab, tokenizer, model) 
#print('Review: [%s]\nSentiment: %s (%.3f%%)' % (text, sentiment, percent*100)) 





