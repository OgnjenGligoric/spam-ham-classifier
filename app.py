import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

data = pd.read_csv('combined_data.csv')

X_train, X_test, y_train, y_test = train_test_split(data['text'], data['label'], test_size=0.2, random_state=42)

vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

def train_and_evaluate_model(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, pos_label='spam')
    recall = recall_score(y_test, y_pred, pos_label='spam')
    f1 = f1_score(y_test, y_pred, pos_label='spam')
    return accuracy, precision, recall, f1

nb_model = MultinomialNB()
svm_model = SVC()
nn_model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=300)

nb_metrics = train_and_evaluate_model(nb_model, X_train_tfidf, X_test_tfidf, y_train, y_test)
print(f"Naive Bayes - Accuracy: {nb_metrics[0]}, Precision: {nb_metrics[1]}, Recall: {nb_metrics[2]}, F1: {nb_metrics[3]}")

svm_metrics = train_and_evaluate_model(svm_model, X_train_tfidf, X_test_tfidf, y_train, y_test)
print(f"SVM - Accuracy: {svm_metrics[0]}, Precision: {svm_metrics[1]}, Recall: {svm_metrics[2]}, F1: {svm_metrics[3]}")

nn_metrics = train_and_evaluate_model(nn_model, X_train_tfidf, X_test_tfidf, y_train, y_test)
print(f"Neural Network - Accuracy: {nn_metrics[0]}, Precision: {nn_metrics[1]}, Recall: {nn_metrics[2]}, F1: {nn_metrics[3]}")