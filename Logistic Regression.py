import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

data = {
    'Age': [22, 25, 35, 10, 28, 45, 18, 30, 8, 40,
            21, 32, 15, 50, 27, 12, 38, 24, 60, 19],
    'Sex': [0, 1, 0, 1, 0, 0, 1, 0, 1, 0,
            1, 0, 1, 0, 1, 1, 0, 1, 0, 0],
    'Pclass': [3, 1, 2, 3, 1, 2, 3, 1, 3, 2,
               2, 1, 3, 1, 2, 3, 1, 2, 3, 1],
    'Survived': [0, 1, 0, 1, 0, 0, 1, 0, 1, 0,
                 1, 0, 1, 0, 1, 1, 0, 1, 0, 0]
}

df = pd.DataFrame(data)

print("Dataset:")
print(df)

X = df[['Age', 'Sex', 'Pclass']]
y = df['Survived']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\nActual Survival:")
print(y_test.values)

print("\nPredicted Survival:")
print(y_pred)

print("\nAccuracy:", accuracy_score(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(
    y_test, y_pred, labels=[0, 1],
    target_names=['Did Not Survive', 'Survived'],
    zero_division=0
))

cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

plt.imshow(cm, cmap='Blues')
plt.title("Titanic Survival - Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.xticks([0, 1], ['Did Not Survive', 'Survived'])
plt.yticks([0, 1], ['Did Not Survive', 'Survived'])

for i in range(2):
    for j in range(2):
        plt.text(j, i, cm[i, j], ha='center', va='center')

plt.colorbar()
plt.tight_layout()
plt.show()
