from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np

def generate_random_flower():

    sepal_length_range = (4.3, 7.9)
    sepal_width_range = (2.0, 4.4)
    petal_length_range = (1.0, 6.9)
    petal_width_range = (0.1, 2.5)

    random_flower = [
        np.random.uniform(*sepal_length_range),
        np.random.uniform(*sepal_width_range),
        np.random.uniform(*petal_length_range),
        np.random.uniform(*petal_width_range),
    ]
    return random_flower

iris = load_iris()
X = iris.data
y = iris.target

#this seperates the testing and training variables and sets them at an 80:20 split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#trains the model using the training data.
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

#this gets the feature importance scores.
feature_importances = clf.feature_importances_

#gets the feature names.
feature_names = iris.feature_names

#sort the importance in decending order
sorted_idx = np.argsort(feature_importances)[::-1]
sorted_importances = feature_importances[sorted_idx]
sorted_features = [feature_names[i] for i in sorted_idx]

y_pred = clf.predict(X_test)

#evaluates the model by comparing the test and the prediction.
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy: {accuracy: .2f}")

#predicts the species for a new flower
for _ in range(5):
    random_flower = generate_random_flower()
prediction = clf.predict([random_flower])
species = iris.target_names[prediction[0]]
print (f"Random Flower Features: {random_flower}")
print(f"The predicted species for the new flower is: {species}")

# Plot feature importance
plt.bar(range(len(sorted_importances)), sorted_importances, tick_label=sorted_features)
plt.title("Feature Importance")
plt.xlabel("Features")
plt.ylabel("Importance Score")
plt.xticks(rotation=45)
plt.show()

#n_estimators are the amount of decision trees in the forest
#random_state controls how random the random dataset is so that it can be easily reproducable
