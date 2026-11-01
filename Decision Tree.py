import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

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

model = DecisionTreeClassifier(max_depth=3, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\nActual Income:")
print(y_test.values)

print("\nPredicted Income:")
print(y_pred)

print("\nAccuracy:", accuracy_score(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(
    y_test, y_pred,
    labels=[0, 1],
    target_names=['Income <=50K', 'Income >50K'],
    zero_division=0
))

plt.figure(figsize=(14, 7))

plot_tree(
    model,
    feature_names=['Age', 'Education', 'HoursPerWeek'],
    class_names=['Income <=50K', 'Income >50K'],
    filled=True,
    rounded=True
)

plt.title("Decision Tree Income Classification")
plt.show()

cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

plt.imshow(cm, cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted Income")
plt.ylabel("Actual Income")
plt.xticks([0, 1], ['<=50K', '>50K'])
plt.yticks([0, 1], ['<=50K', '>50K'])

for i in range(2):
    for j in range(2):
        plt.text(j, i, cm[i, j], ha='center', va='center')

plt.tight_layout()
plt.show()
