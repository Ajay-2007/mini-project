
import pandas as pd
import numpy as np
import seaborn as sns
from nltk import FreqDist
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer

import warnings

warnings.filterwarnings('ignore')

pd.set_option('max_colwidth', 100)

data = pd.read_csv('physics.csv')
data.columns
# print(data)

data.Year.dtype
data.Year

data.head()

"""# Data Preprocessing

### Data Preprocessing Pipelines
![Data Preprocessing Pipeline](https://miro.medium.com/max/976/1*vXpUaBPZRZtAsciMXllmkQ.png)

---



---
"""

# from sklearn.utils import shuffle
# from sklearn.utils import shuffle
# data = shuffle(data)
# data = data.sample(frac=1, replace = True, random_state = 1)
# data.head()

import nltk

nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('punkt')

from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
import re

stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()


def preprocess(document):
    document = str(document)
    document = document.lower()
    # document = re.sub('\W\D',' ', document)
    words = word_tokenize(document)
    # r = compile('\W\D')
    # words = [word for word in words if not r.match(word)]
    words = [lemmatizer.lemmatize(stemmer.stem(word)) for word in words if word not in stopwords.words('english')]

    document = str(" ".join(words))

    return document


# print(data.Topic.unique())
# print(data.Year.unique())

questions = [question for question in data.Questions]
# print(questions[1:3])

# preprocess questions using the preprocess function
questions = [preprocess(question) for question in questions]
# print(questions[1:3])
# len(questions)

# type(data.Questions)
questions = pd.DataFrame(questions)
# questions.head()
data['Questions'] = questions
# data.head()

"""# Feature Engineeing"""

# Tfidf vectorization
vectorizer = TfidfVectorizer()
tfidf_model = vectorizer.fit_transform(data['Questions'])

questions_tfidf = pd.DataFrame(tfidf_model.toarray(), columns=vectorizer.get_feature_names())

# questions_tfidf.head()

column_name = questions_tfidf.columns
# column_name[:70]
questions_tfidf.drop(list(column_name[:85]), inplace=True, axis=1)
# questions_tfidf.head()

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

train_x, valid_x, train_y, valid_y = train_test_split(data['Questions'], data['Year'])

# label encode the target variable here it is topic
encoder = LabelEncoder()
train_y = encoder.fit_transform(train_y)
valid_y = encoder.fit_transform(valid_y)
np.unique(train_y)

"""# Count Vectors as features"""

# create a CountVectorizer object
from sklearn.feature_extraction.text import CountVectorizer

count_vect = CountVectorizer()
count_vect.fit(data['Questions'])
# print("printing valid x value", valid_x)
# transform the training and validation data using count vectorizer object
xtrain_count = count_vect.transform(train_x)
xvalid_count = count_vect.transform(valid_x)
print(xvalid_count)
#
# print(type(xtrain_count))
# print(xtrain_count.shape)

"""# Tf-Idf Vectors as features
![tf_idf](https://skymind.ai/images/wiki/tfidf.png)


---



---
"""

# word level tf-idf

tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern='w{1,}', max_features=1000)
tfidf_vect.fit(data['Questions'])

xtrain_tfidf = tfidf_vect.transform(train_x)
xvalid_tfidf = tfidf_vect.transform(valid_x)

# ngram level tf-idf
tfidf_vect_ngram = TfidfVectorizer(analyzer='word', token_pattern='\w{1,}', ngram_range=(2, 3), max_features=1000)

tfidf_vect_ngram.fit(data['Questions'])
xtrain_tfidf_ngram = tfidf_vect_ngram.transform(train_x)
xvalid_tfidf_ngram = tfidf_vect_ngram.transform(valid_x)

# characters level tf-idf
tfidf_vect_ngram_chars = TfidfVectorizer(analyzer='char', token_pattern='\w{1,}', ngram_range=(2, 3), max_features=1000)

tfidf_vect_ngram_chars.fit(data['Questions'])

xtrain_tfidf_ngram_chars = tfidf_vect_ngram_chars.transform(train_x)
xvalid_tfidf_ngram_chars = tfidf_vect_ngram_chars.transform(valid_x)

"""# Model Building"""

from sklearn import linear_model, naive_bayes, metrics, svm


def train_model(classifier, feature_vector_train, label, feature_vector_valid):
    # fit the training dataset on the classifier
    classifier.fit(feature_vector_train, label)
    # print(feature_vector_valid)
    # predict the labels on validation dataset
    predictions = classifier.predict(feature_vector_valid)
    # print(predictions)

    return metrics.accuracy_score(predictions, valid_y), classifier

# def test_model(classifier, feature_vector_test, label, feature_vector_valid):

"""# Naive Bayes"""

# Naive Bayes on Count Vectors
print("printing xvalid count", xvalid_count)
accuracy_NB_CV, NB_Classifier = train_model(naive_bayes.MultinomialNB(), xtrain_count, train_y, xvalid_count)
print("NB, Count Vectors: ", accuracy_NB_CV)

# Naive Bayes on Word Level Tf-Idf Vectors
accuracy_NB_Tf_IDF, NB_Classifier = train_model(naive_bayes.MultinomialNB(), xtrain_tfidf, train_y, xvalid_tfidf)
print("NB, WordLevel Tf-Idf: ", accuracy_NB_Tf_IDF)

# Naive Bayes on Ngram Level Tf-Idf Vectors
accuracy = train_model(naive_bayes.MultinomialNB(), xtrain_tfidf_ngram, train_y, xvalid_tfidf_ngram)
print("NB, N-Gram Vectors: ", accuracy)

# Naive Bayes on Character level Tf-Idf Vectors
accuracy = train_model(naive_bayes.MultinomialNB(), xtrain_tfidf_ngram_chars, train_y, xvalid_tfidf_ngram_chars)
print("NB, Character Level Vectors: ", accuracy)

"""# Logistic Regression"""

# Logistic Regression on Count Vectors
accuracy_LR_CV, LR_Classifier = train_model(linear_model.LogisticRegression(), xtrain_count, train_y, xvalid_count)
print("LR, Count Vectors: ", accuracy_LR_CV)

# Logistic Regression on Word Level Tf-Idf Vectors
accuracy_LR_TF_IDF, LR_Classifier = train_model(linear_model.LogisticRegression(), xtrain_tfidf, train_y, xvalid_tfidf)
print("LR, WordLevel Tf-Idf: ", accuracy_LR_TF_IDF)

# Logistic Regression on Ngram Level Tf-Idf Vectors
accuracy = train_model(linear_model.LogisticRegression(), xtrain_tfidf_ngram, train_y, xvalid_tfidf_ngram)
print("LR, N-Gram Vectors: ", accuracy)

# Logistic Regression on Character level Tf-Idf Vectors
accuracy = train_model(linear_model.LogisticRegression(), xtrain_tfidf_ngram_chars, train_y, xvalid_tfidf_ngram_chars)
print("LR, Character Level Vectors: ", accuracy)



# Support Vector Machine on Count Vectors
from sklearn import svm
accuracy_SVM_CV, SVM_Classifier = train_model( svm.SVC(), xtrain_count, train_y, xvalid_count)
print("LR, Count Vectors: ", accuracy_SVM_CV)

# Support Vector Machine on Word Level Tf-Idf Vectors
accuracy_SVM_TF_IDF, SVM_Classifier = train_model( svm.SVC(), xtrain_tfidf, train_y, xvalid_tfidf)
print("LR, WordLevel Tf-Idf: ", accuracy_SVM_TF_IDF)

# Support Vector Machine on Ngram Level Tf-Idf Vectors
accuracy = train_model( svm.SVC(), xtrain_tfidf_ngram, train_y, xvalid_tfidf_ngram)
print("LR, N-Gram Vectors: ", accuracy)

# Support Vector Machine on Character level Tf-Idf Vectors
accuracy = train_model( svm.SVC(), xtrain_tfidf_ngram_chars, train_y, xvalid_tfidf_ngram_chars)
print("LR, Character Level Vectors: ", accuracy)
import datetime

from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
import sys
import pymysql as MySQLdb
from PyQt5.uic import loadUiType


predict, _ = loadUiType('predict.ui')
prediction, _ = loadUiType('prediction.ui')
welcome, _ = loadUiType('welcome.ui')
question_classification, _ = loadUiType(('question_classification.ui'))
login, _ = loadUiType('login.ui')
front_page, _ = loadUiType('front_page.ui')

class FrontPage(QMainWindow, front_page):
    def __init__(self):
        QMainWindow.__init__(self)
        self.setupUi(self)
        self.pushButton.clicked.connect(self.call_login)

    def call_login(self):
        self.login = Login()
        self.close()
        self.login.show()

class Login(QWidget, login):
    def __init__(self):
        QWidget.__init__(self)
        self.setupUi(self)
        self.pushButton.clicked.connect(self.Handle_Login)
    def Handle_Login(self):
        username = self.lineEdit.text()
        password = self.lineEdit_2.text()

        self.db = MySQLdb.connect(host='localhost' , user='root', password='d33ps3curity', db='library')
        self.cur = self.db.cursor()

        sql: str = ''' SELECT * FROM users '''
        self.cur.execute(sql)
        data = self.cur.fetchall()
        for row in data:
            if username == row[1] and password == row[3]:
                print('user match')
                self.window_2 = Welcome()
                self.close()
                self.window_2.show()
            else:
                self.label.setText('Make Sure You Entered Your Username and Password Correctly')

class Welcome(QMainWindow, welcome):
    def __init__(self):
        QMainWindow.__init__(self)
        self.setupUi(self)
        self.pushButton.clicked.connect(self.call_practice)
        self.pushButton_2.clicked.connect(self.call_other)
        self.pushButton_3.clicked.connect(self.call_predict)
    def call_practice(self):
        self.practice_window = PracticeApp()
        self.close()
        self.practice_window.show()

    def call_other(self):
        self.other_info = MainApp()
        self.close()
        self.other_info.show()

    def call_predict(self):
        self.predict = Predict()
        self.close()
        self.predict.show()



class PracticeApp(QMainWindow, question_classification):
    def __init__(self):
        QMainWindow.__init__(self)
        self.setupUi(self)

class Predict(QMainWindow, predict):
    def __init__(self):
        QMainWindow.__init__(self)
        self.setupUi(self)

    #

        # text_data = self.textEdit.toPlainText()
        # text_data = preprocess(text_data)
        #
        # test_chem = pd.read_csv('test_chemistry.csv')
        # # text_data = str(text_data)
        # # test_chem.head()
        #
        # questions = [question for question in test_chem.Questions]
        # # print(questions[1:3])
        #
        # # preprocess questions using the preprocess function
        # questions = [preprocess(question) for question in questions]
        # # print(questions[1:3])
        # # len(questions)
        #
        # # type(data.Questions)
        # questions = pd.DataFrame(questions)
        # # questions.head()
        # test_chem['Questions'] = questions
        # # data.head()
        # test_x = test_chem['Questions']
        # count_vect_test = CountVectorizer()
        # # count_vect_test.fit(test_x)
        #
        # xtest_count = count_vect_test.transform(test_x)
        # print("xvalue_test", xtest_count)
        # # print(test_chem['Questions'])
        # # X = naive_bayes.check_array(text_data, accept_sparse='csr')
        # prediction_value = NB_Classifier.predict(xtest_count)
        # print("printing prediction value", prediction_value)
        self.pushButton_2.clicked.connect(self.call_prediction)

    def call_prediction(self):
        self.window = MainApp()
        self.close()
        self.window.show()



class MainApp(QMainWindow, prediction):
    def __init__(self):
        QMainWindow.__init__(self)
        self.setupUi(self)
        # self.getQuestion(self)
        self.pushButton.clicked.connect(self.call_back)
        self.Show_Prediction()

    def call_back(self):
        self.back = Welcome()
        self.close()
        self.back.show()
    def Show_Prediction(self):
        self.textBrowser.setText(str(round(accuracy_NB_CV, 2)))
        self.textBrowser_2.setText(str(round(accuracy_NB_Tf_IDF, 2)))


        self.textBrowser_4.setText(str(round(accuracy_LR_CV, 2)))
        self.textBrowser_3.setText(str(round(accuracy_LR_TF_IDF, 2)))


        self.textBrowser_6.setText(str(round(accuracy_SVM_CV, 2)))
        self.textBrowser_5.setText(str(round(accuracy_SVM_TF_IDF, 2)))


def main():
        app = QApplication(sys.argv)
        window = FrontPage()
        window.show()
        app.exec_()
if __name__ == '__main__':
        main()
