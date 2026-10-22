import pandas as pd  # noqa: EXE002

df = pd.read_csv("titanic.csv")
df = df.drop(["PassengerId", "Ticket", "Cabin"], axis=1)

df["Embarked"] = df["Embarked"].fillna("S")


def make_age(row):
    if pd.isnull(row["Age"]):
        if row["Pclass"] == 1:
            return 37
        if row["Pclass"] == 2:
            return 29
        if row["Pclass"] == 3:
            return 24
    else:
        return row["Age"]


df["Age"] = df.apply(make_age, axis=1)

names = df["Name"]
df = df.drop(["Name"], axis=1)


def intonumber(sex):
    if sex == "male":
        return 0
    if sex == "female":
        return 1


df["Sex"] = df["Sex"].apply(intonumber)

df[list(pd.get_dummies(df["Embarked"]).columns)] = pd.get_dummies(df["Embarked"])
df = df.drop(["Embarked"], axis=1)


def is_alone(row):
    if row["SibSp"] + row["Parch"] == 0:
        return 1
    return 0


df["Alone"] = df.apply(is_alone, axis=1)

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

x = df.drop("Survived", axis=1)
y = df["Survived"]

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

names2 = names.loc[x_test.index].tolist()

sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

model = KNeighborsClassifier(n_neighbors=3)
model.fit(x_train, y_train)
y_pred = model.predict(x_test)


for i in range(len(names2)):
    Passenger = names2[i]
    result = y_pred[i]
    if result == 1:
        print(Passenger, " Survived")
    else:
        print(Passenger, "Dead")

print("     ")

from sklearn.metrics import accuracy_score, confusion_matrix

precent = accuracy_score(y_test, y_pred) * 100
print(precent, "- Процент точности показаний")

a = confusion_matrix(y_test, y_pred)
print(a[1][1], "True survived")
print(a[0][0], "True dead")
print(a[0][1], "False survived")
print(a[1][0], "False dead")


import matplotlib.pyplot as plt

verno = pd.Series([a[1][1], a[0][1]])
verno.plot(kind="pie")
plt.legend(["Survive", "False survive"])
plt.show()


neverno = pd.Series([a[0][0], a[1][0]])
neverno.plot(kind="pie")
plt.legend(["Dead", "False dead"])
plt.show()
