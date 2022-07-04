#!/usr/bin/env python
# coding: utf-8

# In[1]:


# install library
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import string
import warnings
warnings.filterwarnings('ignore') # Hides warning
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore",category=UserWarning)


# In[2]:


df=pd.read_excel("C:/Users/hp/Downloads/B08XJCMGL7-IN-Reviews-220608.xlsx")
df=df.iloc[:,2:3]
df


# In[3]:


df.shape


# In[4]:


df.info()


# In[5]:


df.isnull().sum()


# In[6]:


df.dropna(inplace = True, axis = 0)


# In[7]:


df.isnull().sum()


# In[8]:


df.shape


# 

# # 1 . Data Preprocessing

# 1. Maketext in lower case
# 2. Remove numbers
# 3. Remove html tags
# 4. Remove punctuions
# 5. Remove Stopwords
# 6. Remove multiple spaces

# In[9]:


#pip install data-purifier

import datapurifier as dp
from datapurifier import Mleda
from datapurifier import Nlpeda
from datapurifier import Nlpurifier
from datapurifier import  MlReport


# In[10]:


from datapurifier import Nlpurifier, NLAutoPurifier

df2 = NLAutoPurifier(df, target = "Content")


# In[11]:


df2


# In[12]:


import nltk
from nltk.stem import WordNetLemmatizer 


# In[13]:


lemmatizer = WordNetLemmatizer()
df2['Content'] = df2['Content'].apply(lambda x: " ".join([lemmatizer.lemmatize(word) for word in x.split()]))


# In[14]:


df2


# 

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# # 2. Sentiment

# ## 2.1  [ Textblob ]

# In[15]:


#import TextBlob
from textblob import TextBlob


# In[16]:


#get sentiment score for each review
df2['polarity_TextBlob'] = df2['Content'].apply(lambda s: TextBlob(s).sentiment.polarity)
df2.head()


# In[17]:


def sentiment(label):
    if label <0:
        return "Negative"
    elif label>=0:
        return "Positive"


# In[18]:


df2['sentiment_TextBlob'] = df2['polarity_TextBlob'].apply(sentiment)


# In[19]:


df2['sentiment_TextBlob'].value_counts()


# In[20]:


df2.head()


# ##  

# In[21]:


fig = plt.figure(figsize=(10,7))
sns.countplot(x='sentiment_TextBlob', data = df2)


# In[22]:


df2['polarity_TextBlob'].hist(bins=30, figsize=(15,10))


# 1. polarity is almost normal and positive 
# 2. polarity is max between 0 to 0.75, shows reviews have positive sentiment

# ##  ( WordCloud ) 

# ###  Positive 

# In[23]:


pos_content = df2[df2.sentiment_TextBlob == 'Positive']
pos_content = pos_content.sort_values(['polarity_TextBlob'], ascending= False)


# In[24]:


from wordcloud import WordCloud

text = ' '.join([word for word in pos_content['Content']])
plt.figure(figsize=(20,15), facecolor='None')
wordcloud = WordCloud(max_words=500, width=1600, height=800).generate(text)
plt.imshow(wordcloud)


# ### Negative

# In[25]:


neg_content = df2[df2.sentiment_TextBlob == 'Negative']
neg_content = neg_content.sort_values(['polarity_TextBlob'], ascending= False)


# In[26]:


text = ' '.join([word for word in neg_content['Content']])
plt.figure(figsize=(20,15), facecolor='None')
wordcloud = WordCloud(max_words=500, width=1600, height=800).generate(text)
plt.imshow(wordcloud)


# In[ ]:





# ## 2.2 [ VANDER ]

# In[27]:


#import VADER sentiment
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


# In[28]:


vader_sentiment = SentimentIntensityAnalyzer()
df2['polarity_VADER'] = df2['Content'].apply(lambda s: vader_sentiment.polarity_scores(s)['compound'])


# In[29]:


def sentiment(label):
    if label <0:
        return "Negative"
    elif label>=0:
        return "Positive"


# In[30]:


df2['sentiment_VADER'] = df2['polarity_VADER'].apply(sentiment)


# In[31]:


print(df2['sentiment_TextBlob'].value_counts())
df2.head()


# 

# In[32]:


fig = plt.figure(figsize=(10,7))
sns.countplot(x='sentiment_VADER', data = df2)


# In[33]:


df2['polarity_VADER'].hist(bins=30, figsize=(15,10))


# In[ ]:





# ## (WordCloud)

# ### Positive

# In[34]:


pos_content = df2[df2.sentiment_VADER == 'Positive']
pos_content = pos_content.sort_values(['polarity_VADER'], ascending= False)


# In[35]:


from wordcloud import WordCloud

text = ' '.join([word for word in pos_content['Content']])
plt.figure(figsize=(20,15), facecolor='None')
wordcloud = WordCloud(max_words=500, width=1600, height=800).generate(text)
plt.imshow(wordcloud)


# ### Negative

# In[36]:


neg_content = df2[df2.sentiment_VADER == 'Negative']
neg_content = neg_content.sort_values(['polarity_VADER'], ascending= False)


# In[37]:


text = ' '.join([word for word in neg_content['Content']])
plt.figure(figsize=(20,15), facecolor='None')
wordcloud = WordCloud(max_words=500, width=1600, height=800).generate(text)
plt.imshow(wordcloud)


# In[38]:


df2.to_csv("FinalSentiment.csv")


# # 4. Model Building

# In[39]:


df2.head()


# ## 4.1.1  Applying NB on TextBlob

# In[40]:


# Split into training and testing data
from sklearn.model_selection import train_test_split
x = df2['Content']
y = df2['sentiment_TextBlob']
x, x_test, y, y_test = train_test_split(x,y, stratify=y, test_size=0.25, random_state=42)


# In[41]:


from sklearn.feature_extraction.text import CountVectorizer

vec = CountVectorizer(stop_words='english')
x = vec.fit_transform(x).toarray()
x_test = vec.transform(x_test).toarray()


# In[42]:


from sklearn.naive_bayes import MultinomialNB

modelT = MultinomialNB()
modelT.fit(x, y)


# In[43]:


modelT.score(x_test, y_test)


# In[44]:


from sklearn.naive_bayes import GaussianNB

modelT2 = GaussianNB()
modelT2.fit(x, y)

modelT2.score(x_test, y_test)


# In[45]:


from sklearn.naive_bayes import BernoulliNB

modelT3 = BernoulliNB()
modelT3.fit(x, y)
modelT3.score(x_test, y_test)


# In[46]:


from sklearn.naive_bayes import ComplementNB

modelT4 = ComplementNB()
modelT4.fit(x, y)
modelT4.score(x_test, y_test)


# In[ ]:





# In[ ]:





# ## 4.1.2  Applying NB on VADER

# In[47]:


# Split into training and testing data
from sklearn.model_selection import train_test_split
x1 = df2['Content']
y1 = df2['sentiment_VADER']
x1, x1_test, y1, y1_test = train_test_split(x1,y1, stratify=y1, test_size=0.25, random_state=42)


# In[48]:


from sklearn.feature_extraction.text import CountVectorizer

vec = CountVectorizer(stop_words='english')
x1 = vec.fit_transform(x1).toarray()
x1_test = vec.transform(x1_test).toarray()


# In[49]:


from sklearn.naive_bayes import MultinomialNB

modelV = MultinomialNB()
modelV.fit(x1, y1)


# In[50]:


modelV.score(x1_test, y1_test)


# In[51]:


from sklearn.naive_bayes import GaussianNB

modelV2 = GaussianNB()
modelV2.fit(x1, y1)

modelV2.score(x1_test, y1_test)


# In[52]:


from sklearn.naive_bayes import BernoulliNB

modelV3 = BernoulliNB()
modelV3.fit(x1, y1)
modelV3.score(x1_test, y1_test)


# In[53]:


from sklearn.naive_bayes import ComplementNB

modelV4 = ComplementNB()
modelV4.fit(x1, y1)
modelV4.score(x1_test, y1_test)


# In[ ]:





# ## 4.2.1  TextBlob

# In[54]:


df3 = pd.read_csv('FinalSentiment.csv')
df3.drop(['polarity_VADER','Unnamed: 0','sentiment_VADER'], axis = 1, inplace = True)
df3.head()


# In[55]:


df3['sentiment_TextBlob'] = df3['sentiment_TextBlob'].map({'Positive':1, 'Negative': 0})
df3.head()


# In[56]:


X = df3['Content'].values.astype(str)
y = df3['sentiment_TextBlob']


# In[57]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=20, stratify=y)

print('Number of reviews in the total set : {}'.format(len(X)))
print('Number of reviews in the training set : {}'.format(len(X_train)))
print('Number of reviews in the testing set : {}'.format(len(X_test)))


# # Using CountVectorizer

# In[58]:


from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
cvx_train = cv.fit_transform(X_train).toarray()
cvx_test = cv.transform(X_test).toarray()


# ### Logistic Regression 

# In[59]:


# Logistic regression model
from sklearn.linear_model import LogisticRegression

classifier=LogisticRegression()
classifier.fit(cvx_train,y_train)

print(classifier.fit(cvx_train,y_train))

# Predict for x dataset
y_pred=classifier.predict(cvx_train)


print("_______________________________________________")
    

# Confusion Matrix for the model accuracy
from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_train,y_pred)
print (f'confusion_matrix:\n{confusion_matrix}')


print("_______________________________________________")


# The model accuracy is calculated by (a+d)/(a+b+c+d)
auc = (2068+8845)/(2068+113+38+8845)
print(f'train accuracy:{auc}')


print("_______________________________________________")


#Classification report
from sklearn.metrics import classification_report
print(f'Classification report:\n{classification_report(y_train,y_pred)}')


print("_______________________________________________")


print(" test accuracy: ",classifier.score(cvx_test,y_test))


# ### KNN 

# In[60]:


from sklearn.neighbors import KNeighborsClassifier
 
knn = KNeighborsClassifier(n_neighbors = 1)
 
knn.fit(cvx_train, y_train)
pred = knn.predict(cvx_test)


# In[61]:


from sklearn.metrics import classification_report, confusion_matrix

knn_cm = confusion_matrix(y_test, pred)
print(f'KNN Confusion Matrix:\n{knn_cm}')
 
print("_______________________________________________")


knn_clr = classification_report(y_test, pred)
print(f'KNN Classification Report:\n{knn_clr}')


print("_______________________________________________")


accuracy = (484+2849)/(484+243+112+2849)
print(f' Random Forest Acc:{accuracy}')


# ### Random Forest 

# In[94]:


from sklearn.ensemble import RandomForestClassifier

ram = RandomForestClassifier(n_estimators = 150,criterion = 'entropy')
ram.fit(cvx_train,y_train)

y_pred = ram.predict(cvx_test)


# In[96]:


ram_cm = confusion_matrix(y_test, y_pred)
print(f'Random Forest CM:\n{ram_cm}')


print("_______________________________________________")


ram_cr = classification_report(y_test, y_pred)
print(f'Random Forest Classification Report:\n{ram_cr}')


print("_______________________________________________")


accuracy = (628+2854)/(628+141+65+2854)
print(f' Random Forest Acc:{accuracy}')


# In[ ]:





# In[ ]:





# # Using TfidfVectorizer

# In[64]:


from sklearn.feature_extraction.text import TfidfVectorizer

vectorization=TfidfVectorizer()
tfvx_train=vectorization.fit_transform(X_train)
tfvx_test=vectorization.transform(X_test)


# ### Logistic Regression

# In[65]:


# Logistic regression model
from sklearn.linear_model import LogisticRegression

classifier=LogisticRegression()
classifier.fit(tfvx_train,y_train)

print(classifier.fit(tfvx_train,y_train))

# Predict for x dataset
y_pred=classifier.predict(tfvx_train)


print("_______________________________________________")
    

# Confusion Matrix for the model accuracy
from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_train,y_pred)
print (f'confusion_matrix:\n{confusion_matrix}')


print("_______________________________________________")


# The model accuracy is calculated by (a+d)/(a+b+c+d)
auc = (1745+8842)/(1745+436+41+8842)
print(f'train accuracy:{auc}')


print("_______________________________________________")


#Classification report
from sklearn.metrics import classification_report
print(f'Classification report:\n{classification_report(y_train,y_pred)}')


print("_______________________________________________")


print(" test accuracy: ",classifier.score(tfvx_test,y_test))


# ### KNN Using TfidfVectorizer

# In[66]:


from sklearn.neighbors import KNeighborsClassifier
 
knn = KNeighborsClassifier(n_neighbors = 1)
 
knn.fit(tfvx_train, y_train)
pred = knn.predict(tfvx_test)


# In[67]:


from sklearn.metrics import classification_report, confusion_matrix

knn_cm = confusion_matrix(y_test, pred)
print(f'KNN Confusion Matrix:\n{knn_cm}')
 
print("_______________________________________________")


knn_clr = classification_report(y_test, pred)
print(f'KNN Classification Report:\n{knn_clr}')


print("_______________________________________________")


accuracy = (321+2864)/(321+406+97+2864)
print(f' KNN Acc:{accuracy}')


# ### Random Forest

# In[97]:


from sklearn.ensemble import RandomForestClassifier

ram = RandomForestClassifier(n_estimators = 150,criterion = 'entropy')
ram.fit(tfvx_train,y_train)


# In[98]:


y_pred = ram.predict(cvx_test)


# In[100]:


ram_cm = confusion_matrix(y_test, y_pred)
print(f'Random Forest CM:\n{ram_cm}')


print("_______________________________________________")


ram_cr = classification_report(y_test, y_pred)
print(f'Random Forest Classification Report:\n{ram_cr}')


print("_______________________________________________")


accuracy = (513+2888)/(513+256+31+2888)
print(f' Random Forest Acc:{accuracy}')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# ## 4.2.2  VADER

# In[71]:


df4 = pd.read_csv('FinalSentiment.csv')
df4.drop(['Unnamed: 0','polarity_TextBlob','sentiment_TextBlob'], axis = 1 , inplace = True)


# In[72]:


df4['sentiment_VADER'] = df4['sentiment_VADER'].map({'Positive':1, 'Negative': 0})
df4.head()


# In[73]:


X = df4['Content'].values.astype(str)
y = df4['sentiment_VADER']


# In[74]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=20, stratify=y)

print('Number of reviews in the total set : {}'.format(len(X)))
print('Number of reviews in the training set : {}'.format(len(X_train)))
print('Number of reviews in the testing set : {}'.format(len(X_test)))


# # Using CountVectorizer

# In[75]:


from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
cvx_train = cv.fit_transform(X_train).toarray()
cvx_test = cv.transform(X_test).toarray()


# ### Logistic Regression

# In[76]:


# Logistic regression model
from sklearn.linear_model import LogisticRegression

classifier=LogisticRegression()
print(classifier.fit(cvx_train,y_train))

# Predict for x dataset
y_pred=classifier.predict(cvx_train)


print("_______________________________________________")
    

# Confusion Matrix for the model accuracy
from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_train,y_pred)
print (f'confusion_matrix:\n{confusion_matrix}')


print("_______________________________________________")


# The model accuracy is calculated by (a+d)/(a+b+c+d)
auc=(2103+8694)/(2103+206+61+8694)
print(f'train accuracy:{auc}')


print("_______________________________________________")


#Classification report
from sklearn.metrics import classification_report
print(f'Classification report:\n{classification_report(y_train,y_pred)}')


print("_______________________________________________")


print(" test accuracy: ",classifier.score(cvx_test,y_test))


# ### KNN using CountVectorizer

# In[77]:


from sklearn.neighbors import KNeighborsClassifier
 
knn = KNeighborsClassifier(n_neighbors = 1)
 
knn.fit(cvx_train, y_train)
pred = knn.predict(cvx_test)


# In[78]:


from sklearn.metrics import classification_report, confusion_matrix

knn_cm = confusion_matrix(y_test, pred)
print(f'KNN Confusion Matrix:\n{knn_cm}')
 
print("_______________________________________________")


knn_clr = classification_report(y_test, pred)
print(f'KNN Classification Report:\n{knn_clr}')


print("_______________________________________________")


accuracy = (483+2800)/(483+286+119+2800)
print(f' KNN Accuracy:{accuracy}')


# ### Random Forest Using CountVectorizer

# In[101]:


from sklearn.ensemble import RandomForestClassifier

ram = RandomForestClassifier(n_estimators = 150,criterion = 'entropy')
ram.fit(cvx_train,y_train)

y_pred = ram.predict(cvx_test)


# In[103]:


ram_cm = confusion_matrix(y_test, y_pred)
print(f'Random Forest CM:\n{ram_cm}')


print("_______________________________________________")


ram_cr = classification_report(y_test, y_pred)
print(f'Random Forest Classification Report:\n{ram_cr}')


print("_______________________________________________")


accuracy = (622+2849)/(622+147+70+2849)
print(f' Random Forest Acc:{accuracy}')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# # Using TfidfVectorizer

# In[81]:


from sklearn.feature_extraction.text import TfidfVectorizer

vectorization=TfidfVectorizer()
tfvx_train=vectorization.fit_transform(X_train)
tfvx_test=vectorization.transform(X_test)


# ### Logistic Regression

# In[82]:


# Logistic regression model
from sklearn.linear_model import LogisticRegression

classifier=LogisticRegression()
classifier.fit(tfvx_train,y_train)

print(classifier.fit(tfvx_train,y_train))

# Predict for x dataset
y_pred=classifier.predict(tfvx_train)


print("_______________________________________________")
    

# Confusion Matrix for the model accuracy
from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_train,y_pred)
print (f'confusion_matrix:\n{confusion_matrix}')


print("_______________________________________________")


# The model accuracy is calculated by (a+d)/(a+b+c+d)
auc = (1795+8681)/(1795+514+74+8681)
print(f' train accuracy:{auc}')


print("_______________________________________________")


#Classification report
from sklearn.metrics import classification_report
print(f'Classification report:\n{classification_report(y_train,y_pred)}')


print("_______________________________________________")


print(" test accuracy: ",classifier.score(tfvx_test,y_test))


# ### KNN using TF-IDF

# In[83]:


from sklearn.neighbors import KNeighborsClassifier
 
knn = KNeighborsClassifier(n_neighbors = 1)
 
knn.fit(tfvx_train, y_train)
pred = knn.predict(tfvx_test)


# In[84]:


from sklearn.metrics import classification_report, confusion_matrix

knn_cm = confusion_matrix(y_test,pred)
print(f'KNN Confusion Matrix:\n{knn_cm}')

 
print("_______________________________________________")


knn_clr = classification_report(y_test,pred)
print(f'KNN Classification Report:\n{knn_clr}')


print("_______________________________________________")


accuracy = (348+2829)/(348+421+90+2829)
print(f' KNN Acc:{accuracy}')


# ### Random Forest Using TF-IDF

# In[104]:


from sklearn.ensemble import RandomForestClassifier

ram = RandomForestClassifier(n_estimators = 150,criterion = 'entropy')
ram.fit(tfvx_train,y_train)


# In[105]:


y_pred = ram.predict(tfvx_test)


# In[107]:


ram_cm = confusion_matrix(y_test, y_pred)
print(f'Random Forest CM:\n{ram_cm}')


print("_______________________________________________")


ram_cr = classification_report(y_test, y_pred)
print(f'Random Forest Classification Report:\n{ram_cr}')


print("_______________________________________________")


accuracy = (602+2874)/(602+167+48+2874)
print(f' Random Forest Acc:{accuracy}')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




