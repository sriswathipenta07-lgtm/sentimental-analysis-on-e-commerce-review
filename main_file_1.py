import pandas as pd 
from tabulate import tabulate
import textwrap
import seaborn as sns
import matplotlib.pyplot as plt
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer 
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib 
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix,accuracy_score
data=pd.read_csv('train_data.csv')
data.head()
data.columns
reviews_combined = data['reviews.title'] + " " + data['reviews.text']

df = pd.DataFrame({
    'reviews': reviews_combined,
    'sentiment': data['sentiment']
})
df.head()
def wrap_text(text, width=50):
    if pd.isna(text):  
        return ""  
    return "\n".join(textwrap.wrap(text, width))

df['reviews'] = df['reviews'].apply(lambda x: wrap_text(x, width=50))

print(tabulate(df.head(2), headers='keys', tablefmt='grid', showindex=False))
df['sentiment'].value_counts()
sns.set(style='whitegrid')

plt.figure(figsize=(6, 4))
sns.countplot(x='sentiment', data=df, palette='viridis')

plt.title('Count of Sentiment Categories', fontsize=16)
plt.xlabel('Sentiment', fontsize=14)
plt.ylabel('Count', fontsize=14)

plt.show()
df.describe()
df.isnull().sum() 
nltk.download('stopwords')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english')) 
def preprocess_text(text):
    if pd.isna(text):  
        return ""
    
    text = text.lower()
    
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    tokens = text.split()
    
    tokens = [word for word in tokens if word not in stop_words]
    
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    
    return ' '.join(tokens) 
df['cleaned_reviews'] = df['reviews'].apply(preprocess_text)
print(df[['reviews', 'cleaned_reviews']].head()) 
tfidf_vectorizer = TfidfVectorizer(max_features=5000)  
tfidf_matrix = tfidf_vectorizer.fit_transform(df['cleaned_reviews'])
tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_vectorizer.get_feature_names_out())
print(tfidf_df.head())
#joblib.dump(tfidf_matrix, 'tfidf_matrix.joblib')
#joblib.dump(tfidf_vectorizer, 'tfidf_vectorizer.joblib')

print("TF-IDF matrix and vectorizer saved successfully.") 

label_mapping = {'Negative': 0, 'Neutral': 1, 'Positive': 2}
 
tfidf_matrix = tfidf_vectorizer.transform(df['cleaned_reviews']) 
X = tfidf_matrix
y = df['sentiment']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) 
import xgboost as xgb
model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
model.fit(X_train, y_train)

y_pred = model.predict(X_test) 
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=label_mapping.keys())) 
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy Score: {accuracy:.2f}") 
joblib.dump(model,'Xgboost.joblib')  
conf_matrix = confusion_matrix(y_test, y_pred)


plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
            xticklabels=label_mapping.keys(), yticklabels=label_mapping.keys())
plt.title('Confusion Matrix Heatmap')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()