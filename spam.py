import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, fbeta_score, classification_report
import pickle
import joblib

df = pd.read_csv('spam.csv')
df['Category'] = df['Label']
df['Message'] = df['EmailText']
df = df.drop(['Label','EmailText'],axis=1)
df['spam'] = df['Category'].apply(lambda x: 1 if x=='spam' else 0)

df['Category'].value_counts().plot(kind = 'pie', explode=[0,0.1], figsize=(6,6), autopct='%1.2f%%')
plt.ylabel('Ham vs Spam')
plt.legend(["Ham", "Spam"])
plt.show()

X_train, X_test, y_train, y_test = train_test_split(df.Message, df.spam, test_size=0.25)

X_train, y_train = shuffle(X_train, y_train)
X_test, y_test = shuffle(X_test, y_test)

v = CountVectorizer()
X_train_count = v.fit_transform(X_train.values)
X_train_count.toarray()[:3]

model = MultinomialNB()
model.fit(X_train_count, y_train)
X_test_count = v.transform(X_test)
y_pred = model.predict(X_test_count)

X = v.fit_transform(df['Message']).toarray()

score = accuracy_score(y_test, y_pred, normalize=True)
# print(score*100)
# print(fbeta_score(y_test, y_pred, beta=0.5))

saved_model = pickle.dumps(model)
modelfrom_pickle = pickle.loads(saved_model)
y_pred = modelfrom_pickle.predict(X_test_count)
accuracy_score(y_test, y_pred)

joblib.dump(model, 'spam_detector')
joblib.dump(X, 'transform.pkl')