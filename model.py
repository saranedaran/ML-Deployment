import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import sensitive as s

# Load the dataset
data = pd.read_csv("laws.csv")

# Handle NaN values
data.fillna(value="", inplace=True)  # Replace NaN values with an empty string

# Split the dataset into features (X) and target variable (y)
X = data["Attribute"]
y = data["Law/Regulation"]

# Vectorize the textual data using TF-IDF
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
X = tfidf_vectorizer.fit_transform(X)

# Split the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy*100)

new_attributes = ["District_Name", "Month"]
X_new = tfidf_vectorizer.transform(new_attributes)
predicted_laws = model.predict(X_new)
for attribute, law in zip(new_attributes, predicted_laws):
    # if(attribute in s.identify_sensitive_fields()):
        print("Attribute:", attribute, "| Predicted Law/Regulation:", law)
    # else:
    #     print("No need for masking")

    
# Save the trained model
with open('model.pkl', 'wb') as file:
    pickle.dump(model, file)

# Save the TF-IDF vectorizer
with open('tfidf_vectorizer.pkl', 'wb') as file:
    pickle.dump(tfidf_vectorizer, file)
