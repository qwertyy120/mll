import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
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

dt = DecisionTreeClassifier(max_depth=3, random_state=42)
dt.fit(X_train, y_train)

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

dt_pred = dt.predict(X_test)
rf_pred = rf.predict(X_test)

dt_acc = accuracy_score(y_test, dt_pred)
rf_acc = accuracy_score(y_test, rf_pred)

print("\nDecision Tree Accuracy:", dt_acc)
print("Random Forest Accuracy:", rf_acc)

plt.bar(['Decision Tree', 'Random Forest'], [dt_acc, rf_acc])
plt.ylabel("Accuracy")
plt.title("Decision Tree vs Random Forest")
plt.ylim(0, 1.1)
plt.show()

cm = confusion_matrix(y_test, rf_pred, labels=[0, 1])

plt.imshow(cm, cmap='Blues')
plt.title("Random Forest Confusion Matrix")
plt.xlabel("Predicted Income")
plt.ylabel("Actual Income")
plt.xticks([0, 1], ['<=50K', '>50K'])
plt.yticks([0, 1], ['<=50K', '>50K'])

for i in range(2):
    for j in range(2):
        plt.text(j, i, cm[i, j], ha='center', va='center')

plt.tight_layout()
plt.show()
