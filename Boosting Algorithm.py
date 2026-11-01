import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

data = {
    'Age': [25, 45, 30, 50, 35, 48, 28, 55, 40, 60,
            23, 42, 33, 52, 38, 47, 29, 58, 36, 62],
    'Education': [0, 1, 0, 1, 0, 1, 0, 1, 1, 1,
                  0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
    'HoursPerWeek': [35, 50, 40, 60, 38, 55, 30, 65, 45, 60,
                     32, 48, 37, 58, 40, 52, 35, 62, 42, 65],
    'Income': [0, 1, 0, 1, 0, 1, 0, 1, 1, 1,
               0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
}

df = pd.DataFrame(data)

print("Dataset:")
print(df)

X = df[['Age', 'Education', 'HoursPerWeek']]
y = df['Income']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = GradientBoostingClassifier(
    n_estimators=50,
    random_state=42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\nActual Income:")
print(y_test.values)

print("\nPredicted Income:")
print(y_pred)

accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:", accuracy)
print("Accuracy Percentage:", accuracy * 100, "%")

cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

plt.imshow(cm, cmap='Blues')
plt.title("Gradient Boosting Confusion Matrix")
plt.xlabel("Predicted Income")
plt.ylabel("Actual Income")

plt.xticks([0, 1], ['<=50K', '>50K'])
plt.yticks([0, 1], ['<=50K', '>50K'])

for i in range(2):
    for j in range(2):
        plt.text(j, i, cm[i, j], ha='center', va='center')

plt.tight_layout()
plt.show()

plt.bar(X.columns, model.feature_importances_)

plt.title("Feature Importance")
plt.xlabel("Features")
plt.ylabel("Importance")

plt.show()
