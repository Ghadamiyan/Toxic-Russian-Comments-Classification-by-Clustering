

import pandas as pd
import numpy as np
import string
import matplotlib.pyplot as plt

import nltk
from nltk.stem import WordNetLemmatizer, SnowballStemmer
nltk.download('wordnet')
nltk.download('punkt')
stemmer = SnowballStemmer('russian')

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import MinMaxScaler

from sklearn.naive_bayes import MultinomialNB
from sklearn.cluster import KMeans, DBSCAN

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics.cluster import fowlkes_mallows_score

import seaborn as sns

from yellowbrick.text import FreqDistVisualizer, TSNEVisualizer

from sklearn.pipeline import Pipeline 
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV, RandomizedSearchCV

"""## Data Preprocessing"""

data = pd.read_csv('labeled.csv')

# Plotting the distribution of the labels
toxicc = 0
okc = 0
for label in data['toxic']:
    if label == 1:
        toxicc += 1
    if label == 0:
        okc += 1

plt.bar('0 - Non toxic comments', okc)
plt.bar('1 - Toxic comments', toxicc, color = 'orange')
plt.show()

data2 = []
for i in range(0, len(data.index)):

    # Punctuation removal
    table = str.maketrans(dict.fromkeys(string.punctuation))                   
    sentences = (data.comment[i].translate(table))

    # " '\n " removal
    words = sentences[:-3] 

    # Tokenization
    words = nltk.word_tokenize(words)

    # shrt words removal & lemmatization & stemming
    words_ = []
    for word in words:
        if len(word) > 2:   
            if not word.isnumeric():                                                  
                word1 = stemmer.stem(WordNetLemmatizer().lemmatize(word, pos='v'))          
                words_.append(word1)
    data2.append(words_)

data_frame = pd.DataFrame({'comment':data2, 'toxic':data['toxic']})

label1 = data_frame.toxic[data_frame.toxic.eq(1)].index
label0 = data_frame.toxic[data_frame.toxic.eq(0)].sample(toxicc).index

df = data_frame.loc[label0.union(label1)]

df.head()

# Plotting the distribution of the labels
toxicc = 0
okc = 0
for label in df['toxic']:
    if label == 1:
        toxicc += 1
    if label == 0:
        okc += 1

plt.bar('0 - Non toxic comments', okc)
plt.bar('1 - Toxic comments', toxicc, color = 'orange')
plt.show()

comparison = pd.DataFrame({'comments': data['comment'], 'preprocessed comments': data2, 'labels': data['toxic']})
comparison.head()

train_data__, test_data__, train_labels, test_labels = train_test_split(df['comment'], df['toxic'], test_size = 0.2, random_state = 25)

# CountVectorizer & TermFrequencies
cvect = CountVectorizer(ngram_range=(1, 1), lowercase='true')   
tfidf_transformer = TfidfTransformer(norm= 'l2', use_idf= False)

# Transforming the processed data to a list (for tfidf)
data4 = train_data__.astype(str).values.tolist()

train_data1 = cvect.fit_transform(data4)
train_data = tfidf_transformer.fit_transform(train_data1)

# Same procedure for the test data
data5 = test_data__.astype(str).values.tolist()

test_data1 = cvect.transform(data5)
test_data = tfidf_transformer.transform(test_data1)

# Same procedure for the entire data set
train_data6 = df['comment'].astype(str).values.tolist()

data1_ = cvect.fit_transform(train_data6)
data_ = tfidf_transformer.fit_transform(data1_)

"""### Data visualization"""

#https://www.scikit-yb.org/en/latest/api/text/freqdist.html
features = cvect.get_feature_names()

visualizer = FreqDistVisualizer(features=features)
visualizer.fit(data1_)
visualizer.poof()

#https://www.scikit-yb.org/en/latest/api/text/freqdist.html

tsne = TSNEVisualizer()
tsne.fit_transform(data_, df['toxic'])
tsne.poof()

"""## Supervised learning method - Naive Bayes"""

# Model fitting
model = MultinomialNB(alpha = 0.1)
model.fit(train_data, train_labels)

# Prediction
prediction = model.predict(test_data)

"""We are using two scores to compare the results

Accuracy score = $\frac{TP + TN}{Total}$

Fowlkes Mallwows Score = $\frac{TP}{\sqrt(TP+FP)(TP+FN}$ 
"""

accuracy_score(test_labels, prediction)

print(fowlkes_mallows_score(prediction, test_labels))

models = Pipeline([('CountVect', CountVectorizer()), 
                     ('TermFreq', TfidfTransformer()), 
                     ('NB', MultinomialNB())]) 

parameters = { 'CountVect__ngram_range': [(1, 1), (1, 2), (2, 2),(4,5)], 
              'TermFreq__use_idf': (True, False), 
              'TermFreq__norm': ('l1', 'l2'), 
              'NB__alpha': [1, 1e-1, 1e-2, 1e-3] } 

CrossValFolds = 5
grid_search= GridSearchCV(models, parameters, cv = CrossValFolds, n_jobs = -1) 
grid_search.fit(data4, train_labels)

print(grid_search.best_score_) 
print(grid_search.best_params_)

print(classification_report(test_labels, prediction))

confusion_matrix(test_labels, prediction, )

ax = sns.heatmap(confusion_matrix(test_labels, prediction), annot = np.array([['187', '78'],['350', '598']]), cmap=plt.cm.Blues, fmt = '')

"""# Unsupervised methods

## K-means
"""

kmeans = KMeans(n_clusters=2, init='k-means++', random_state=0).fit(data_)

kmeans_pred = []
for label in kmeans.predict(data_):
    if label == 0:
        kmeans_pred.append(1)
    else:
        kmeans_pred.append(0)

fowlkes_mallows_score(kmeans_pred, df['toxic'])

def accuracy_score_(labels__, labels___):
    score = 0
    for idx, label in enumerate(labels__):
        if label == labels___[idx]:
            score = score + 1

    return score/len(labels___)

print(classification_report(df['toxic'], kmeans.predict(data_)))

confusion_matrix(df['toxic'], kmeans_pred)

ax = sns.heatmap(confusion_matrix(test_labels, prediction), annot = np.array([['6878', '2708'],['3733', '1093']]), cmap=plt.cm.Blues, fmt = '')

"""### Elbow method"""

Y = []
for k in range(1,10):
    kmean_ = KMeans(n_clusters=k).fit(data_)
    Y.append(kmean_.inertia_)

X = range(1,10)

plt.figure(figsize=(12,6))
plt.plot(X, Y)
plt.plot(2, 7554, 'gD')

plt.text(2.2, 7603, 'Elbow point', bbox=dict(color='green', alpha=0.8))
plt.text(2.38, 7545, 'k = 2', fontsize = 12)

plt.ylabel('Squared distances sum')
plt.xlabel('No of clusters')
plt.title('Elbow Method')

plt.show()

"""## DBSCAN"""

# Dimensionality reduction
SVD = TruncatedSVD(100)
Pca = SVD.fit_transform(data_)

# Makeing the data positive
scaler = MinMaxScaler().fit(Pca)
data_ = scaler.transform(Pca)

clustering = DBSCAN(eps=0.6, min_samples=4).fit(data_)

fowlkes_mallows_score(clustering.labels_, data['toxic'])

accuracy_score_(clustering.labels_, data['toxic'])

"""###Parameter Tunning"""

# Computing a tabel for parameter comparison

Acc = []
Param = []
X_ = []
print('Clusters \t Acc \t eps \t min_samples')
for eps_ in [ 0.6, 0.8, 0.9]:
    for min_samples_ in range(1, 10):
        clustering = DBSCAN(eps=eps_, min_samples=min_samples_).fit(data_)
        n_clusters = len(set(clustering.labels_)) - (1 if -1 in clustering.labels_ else 0)
        print( n_clusters,' \t       ', round(accuracy_score_(clustering.labels_, data['toxic']), 3),'\t ', eps_, '  \t ', min_samples_)
        
        X_.append(accuracy_score_(clustering.labels_, data['toxic']))
        if n_clusters == 2 or n_clusters == 3:
            Acc.append(accuracy_score_(clustering.labels_, data['toxic']))
            Param.append([eps_, min_samples_])

print('Best score: ', max(Acc),'\nBest Parameters: eps_ = ', Param[np.argmax(Acc, axis = 0)][0], ', min_sample = ', Param[np.argmax(Acc, axis = 0)][1])

Y_ = range(0,27)

plt.figure(figsize=(12,6))
plt.plot(Y_, X_)
plt.plot(22, 0.6602830974188176, 'gD')

plt.text(20.2, 0.635, 'Maximum accuracy', bbox=dict(color='green', alpha=0.8))
plt.text(19.4, 0.615, 'eps_ =  0.9 , min_sample =  1', fontsize = 10)

plt.xlabel('Accuracy')
plt.ylabel('Parameters')
plt.title('Best Parameters DBSCAN')

plt.show()