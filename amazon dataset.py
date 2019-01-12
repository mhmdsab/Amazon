import pandas as pd
import numpy as np
import pickle
from sklearn.utils import shuffle
from sklearn.metrics import roc_auc_score
from string import punctuation
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer,WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.cross_validation import train_test_split
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.metrics import confusion_matrix

#import data
data = pd.read_pickle('Amazon_Unlocked_Mobile.pickle')
data = data[['Reviews','Rating']]
#data = shuffle(data)
data = data.head(4000)

#drop nan values
data.dropna(inplace = True)

#labelizing ratings
data = data[data['Rating'] != 3]
data['Rating'] = np.where(data['Rating'] > 3, 1, 0)

#cleaning the data
for c in punctuation:
    data['Reviews'] = data['Reviews'].str.replace(c,'').str.lower()


#removing stop words & stemming words
##review_array = np.array(data['Reviews'])
##corpus = []
##stop_words = set(stopwords.words('english'))
##ps = PorterStemmer()
##
##for i in range(len(data)):
##    review = review_array[i]
##    review = word_tokenize(review)
##    review = [w for w in review if not w in stop_words]
##    review = [ps.stem(w) for w in review]
##    review = ' '.join(review)
##    corpus.append(review)
   

#creating feature matrices
cv = CountVectorizer(min_df = 5,ngram_range = (1,2))
X = cv.fit_transform(data['Reviews']).toarray()
y = data['Rating'].values


#cross_validation
X_train,X_cv,y_train,y_cv = train_test_split(X,y,test_size = 0.2)

#fit the model
clf = MultinomialNB(alpha = 2)
clf.fit(X_train,y_train)

#prediction
y_pred = clf.predict(X_cv)

#score
print(roc_auc_score(y_cv,y_pred))
print(clf.score(X_cv,y_cv))


#confusion matrix
cm = confusion_matrix(y_cv,y_pred)
print(cm)

#visualize feature names
feature_names = np.array(cv.get_feature_names())
print(len(feature_names))
feature_value_index = clf.coef_[0].argsort()

#top ten highest coeff features
print('bottom',feature_names[feature_value_index[:10]])
print('top',feature_names[feature_value_index[-10:]])

a = 'not an issue phone is working'
b = 'an issue phone is not working'
listt = [a,b]
print(clf.predict(cv.transform(listt)))
