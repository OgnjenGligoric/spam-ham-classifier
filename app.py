import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import time


def main():
    start_time = time.time()

    # Load and Preprocess the Dataset
    load_start = time.time()
    data = pd.read_csv('combined_data.csv')
    load_end = time.time()
    print(f"Data loading time: {load_end - load_start:.2f} seconds")
    print(data.head())

    # Split the Data
    split_start = time.time()
    X_train, X_test, y_train, y_test = train_test_split(data['text'], data['label'], test_size=0.2, random_state=42)
    split_end = time.time()
    print(f"Data splitting time: {split_end - split_start:.2f} seconds")

    # Transform the Text Data
    transform_start = time.time()
    vectorizer = TfidfVectorizer()
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    transform_end = time.time()
    print(f"TF-IDF transformation time: {transform_end - transform_start:.2f} seconds")

    # Train and Evaluate Models
    def train_and_evaluate_model(model, X_train, X_test, y_train, y_test):
        model_fit_start = time.time()
        model.fit(X_train, y_train)
        model_fit_end = time.time()
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        return accuracy, precision, recall, f1, model_fit_end - model_fit_start

    # Initialize models
    nb_model = MultinomialNB()
    svm_model = SVC()
    nn_model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=300)

    # Train and evaluate Naive Bayes model
    nb_metrics = train_and_evaluate_model(nb_model, X_train_tfidf, X_test_tfidf, y_train, y_test)
    print(
        f"Naive Bayes - Accuracy: {nb_metrics[0]}, Precision: {nb_metrics[1]}, Recall: {nb_metrics[2]}, F1: {nb_metrics[3]}, Training Time: {nb_metrics[4]:.2f} seconds")

    # Train and evaluate SVM model
    svm_metrics = train_and_evaluate_model(svm_model, X_train_tfidf, X_test_tfidf, y_train, y_test)
    print(
        f"SVM - Accuracy: {svm_metrics[0]}, Precision: {svm_metrics[1]}, Recall: {svm_metrics[2]}, F1: {svm_metrics[3]}, Training Time: {svm_metrics[4]:.2f} seconds")

    # Train and evaluate Neural Network model
    nn_metrics = train_and_evaluate_model(nn_model, X_train_tfidf, X_test_tfidf, y_train, y_test)
    print(
        f"Neural Network - Accuracy: {nn_metrics[0]}, Precision: {nn_metrics[1]}, Recall: {nn_metrics[2]}, F1: {nn_metrics[3]}, Training Time: {nn_metrics[4]:.2f} seconds")

    # Visualize the Results
    metrics_df = pd.DataFrame({
        'Model': ['Naive Bayes', 'SVM', 'Neural Network'],
        'Accuracy': [nb_metrics[0], svm_metrics[0], nn_metrics[0]],
        'Precision': [nb_metrics[1], svm_metrics[1], nn_metrics[1]],
        'Recall': [nb_metrics[2], svm_metrics[2], nn_metrics[2]],
        'F1 Score': [nb_metrics[3], svm_metrics[3], nn_metrics[3]],
        'Training Time': [nb_metrics[4], svm_metrics[4], nn_metrics[4]]
    })

    plt.figure(figsize=(10, 6))
    sns.barplot(x='Model', y='Accuracy', data=metrics_df)
    plt.title('Accuracy of Models')
    plt.show()

    plt.figure(figsize=(10, 6))
    sns.barplot(x='Model', y='Precision', data=metrics_df)
    plt.title('Precision of Models')
    plt.show()

    plt.figure(figsize=(10, 6))
    sns.barplot(x='Model', y='Recall', data=metrics_df)
    plt.title('Recall of Models')
    plt.show()

    plt.figure(figsize=(10, 6))
    sns.barplot(x='Model', y='F1 Score', data=metrics_df)
    plt.title('F1 Score of Models')
    plt.show()

    plt.figure(figsize=(10, 6))
    sns.barplot(x='Model', y='Training Time', data=metrics_df)
    plt.title('Training Time of Models')
    plt.show()

    # User Input for Testing
    while True:
        print("\nChoose a model to test your email on:")
        print("1. Naive Bayes")
        print("2. SVM")
        print("3. Neural Network")
        print("4. Exit")
        choice = input("Enter the number of the model you want to use: ")

        if choice == '1':
            model = nb_model
            model_name = "Naive Bayes"
        elif choice == '2':
            model = svm_model
            model_name = "SVM"
        elif choice == '3':
            model = nn_model
            model_name = "Neural Network"
        elif choice == '4':
            print("Exiting...")
            break
        else:
            print("Invalid choice. Please choose again.")
            continue

        email_text = input("Enter the email text to classify: ")
        email_tfidf = vectorizer.transform([email_text])
        prediction = model.predict(email_tfidf)[0]
        result = "Spam" if prediction == 1 else "Not Spam"
        print(f"The email is classified as: {result} by the {model_name} model.")

    end_time = time.time()
    print(f"Total elapsed time: {end_time - start_time:.2f} seconds")


if __name__ == '__main__':
    main()
