#!/usr/bin/env python
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPClassifier


# Get the path of the folder where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Change the current working directory to the folder where the script is located
os.chdir(script_dir)

# Load the dataset
df = pd.read_csv("diabetes.csv")

df["isdiab"] = df["diabetes"].map({"Diabetes": 1, "No diabetes": 0})
df["isdiab"].value_counts()[1]

# print(df.head())

# Removing the unimportatnt features
df.drop(
    [
        "waist",
        "hip",
        "waist_hip_ratio",
    ],
    inplace=True,
    axis=1,
)


# Exploratory data analysis
# sns.lineplot(y ='glucose',x='weight', hue='diabetes', data =df)
# plt.show()

# sns.lineplot(y ='glucose',x='age', hue='gender', data =df)
# plt.show()

# figure, axis = plt.subplots(1,2,figsize=(8,6))
# sns.lineplot(ax=axis[0],x='systolic_bp', y='glucose',hue='diabetes',data=df)
# sns.lineplot(ax=axis[1],x='diastolic_bp', y='glucose',hue='diabetes',data=df)
# plt.show()

# sns.lineplot(x=df.hdl_chol,y= df.glucose,hue=df.diabetes,data=df)
# plt.show()

# sns.countplot(x='diabetes',hue='gender',data=df)
# plt.show()

# sns.lineplot(x=df.cholesterol,y= df.glucose,hue=df.diabetes,data=df)
# plt.show()
# In general, It is seen that higher weight and old Age are two major factor causing diabetes.
# What do we conclude from these graphs?
# 1- Diabetic patients have Higher glucose rate and higher weight as comapred to Non -diabetic ones
# 2- The Age is not directly realted but higher gluose level in oldies can be a cause of Diabetes in them, also the males of age 40 to 80 have higher blood glucose level than females.
# 3- The BP is not directly related to the diabetes, as patients have highest BP are Found to be Non-diabetic.
# 4- Diabetic patients have lower HDL-Cholesterol
# 5- Females being diabetic are more than the males being diabetic.
# 6- Higher cholestrol is seen in the patients having diabetes


# Removing the unimortant features
df1 = df[
    [
        "patient_number",
        "cholesterol",
        "glucose",
        "hdl_chol",
        "age",
        "gender",
        "weight",
        "systolic_bp",
        "diastolic_bp",
        "isdiab",
    ]
]


# Splitting the dataset into X and Y
X = df1[
    [
        "cholesterol",
        "glucose",
        "hdl_chol",
        "age",
        "weight",
        "systolic_bp",
        "diastolic_bp",
    ]
]
y = df["isdiab"]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=1)


"""
The Highest K score is 5 with accuraccy of 91.67%
score = []
for i in range(1,20) :
    knn = KNeighborsClassifier(i)
    knn.fit(X_train, y_train)
    y_predict = knn.predict(X_test)
    accuracy = accuracy_score(y_test, y_predict)
    score.append(accuracy*100)
    print('KNN accuracy with {0} neighbors is : {1}'.format(i, accuracy))

plt.figure()
plt.plot(range(1,20), score)
plt.xlabel("K")
plt.ylabel('Accuraccy (%)')
plt.xlim((0,20))
plt.ylim((88,92))
plt.show()
"""
# KNN Model
kNN = KNeighborsClassifier(n_neighbors=5)
kNN.fit(X_train, y_train)
kNN_pred = kNN.predict(X_test)


# SVM Model
svm = svm.SVC(kernel="linear")
svm.fit(X_train, y_train)
predict = svm.predict(X_test)
# accuraccy 92.94%
# print('svm accuracy : ', accuracy_score(y_test, predict))


"""
Neural Network gives accuraccy of 83.3%, so it won't be used 
"""
nn = MLPClassifier(
    solver="lbfgs", alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1
)
nn.fit(X_train, y_train)
nnpredict = nn.predict(X_test)
# print('nn accuracy : ', accuracy_score(y_test, nnpredict))


def predict_diabetes_KNN(
    cholesterol, glucose, hdl_chol, age, weight, systolic_bp, diastolic_bp
):
    user_inputs = np.array(
        [
            cholesterol,
            glucose,
            hdl_chol,
            age,
            float(weight) * 2.2,
            systolic_bp,
            diastolic_bp,
        ]
    ).reshape(1, -1)
    user_df = pd.DataFrame(
        user_inputs,
        columns=[
            "cholesterol",
            "glucose",
            "hdl_chol",
            "age",
            "weight",
            "systolic_bp",
            "diastolic_bp",
        ],
    )

    prediction = kNN.predict(user_df)

    return prediction[0]


def predict_diabetes_SVM(
    cholesterol, glucose, hdl_chol, age, weight, systolic_bp, diastolic_bp
):
    user_inputs = np.array(
        [
            cholesterol,
            glucose,
            hdl_chol,
            age,
            float(weight) * 2.2,
            systolic_bp,
            diastolic_bp,
        ]
    ).reshape(1, -1)
    user_df = pd.DataFrame(
        user_inputs,
        columns=[
            "cholesterol",
            "glucose",
            "hdl_chol",
            "age",
            "weight",
            "systolic_bp",
            "diastolic_bp",
        ],
    )

    prediction = svm.predict(user_df)

    return prediction[0]


def predict_diabetes_NN(
    cholesterol, glucose, hdl_chol, age, weight, systolic_bp, diastolic_bp
):
    user_inputs = np.array(
        [
            cholesterol,
            glucose,
            hdl_chol,
            age,
            float(weight) * 2.2,
            systolic_bp,
            diastolic_bp,
        ]
    ).reshape(1, -1)
    user_df = pd.DataFrame(
        user_inputs,
        columns=[
            "cholesterol",
            "glucose",
            "hdl_chol",
            "age",
            "weight",
            "systolic_bp",
            "diastolic_bp",
        ],
    )

    prediction = nn.predict(user_df)

    return prediction[0]


############################################################################################################################

# regressor = LinearRegression()
# regressor.fit(X_train, y_train)

# y_pred = regressor.predict(X_test)

# mse = mean_squared_error(y_test, y_pred)
# r_squared = r2_score(y_test, y_pred)
# print("Mean Squared Error:", mse)
# print("R-squared:", r_squared)
"""
Mean Squared Error: 0.07666564199374006
R-squared: 0.4480073776450717
"""
# plt.figure(figsize=(8, 6))
# plt.scatter(y_test, y_pred)
# plt.xlabel('Actual')
# plt.ylabel('Predicted')
# plt.title('Actual vs Predicted values (Linear Regression)')
# plt.show()


############################################################################################################################
"""
decistion tree classification accuracy: 0.8589743589743589
"""
clf = DecisionTreeClassifier()

clf = clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
# print("decistion tree classification accuracy:",accuracy_score(y_test, y_pred))

# Post-Purning operation
# path=clf.cost_complexity_pruning_path(X_train,y_train)
# #path variable gives two things ccp_alphas and impurities
# ccp_alphas,impurities=path.ccp_alphas,path.impurities
# print("ccp alpha wil give list of values :",ccp_alphas)
# print("***********************************************************")
# print("Impurities in Decision Tree :",impurities)

# clfs=[]   #will store all the models here
# for ccp_alpha in ccp_alphas:
#     clf=DecisionTreeClassifier(random_state=0,ccp_alpha=ccp_alpha)
#     clf.fit(X_train,y_train)
#     clfs.append(clf)
# print("Last node in Decision tree is {} and ccp_alpha for last node is {}".format(clfs[-1].tree_.node_count,ccp_alphas[-1]))

# train_scores = [clf.score(X_train, y_train) for clf in clfs]
# test_scores = [clf.score(X_test, y_test) for clf in clfs]
# fig, ax = plt.subplots()
# ax.set_xlabel("alpha")
# ax.set_ylabel("accuracy")
# ax.set_title("Accuracy vs alpha for training and testing sets")
# ax.plot(ccp_alphas, train_scores, marker='o', label="train",drawstyle="steps-post")
# ax.plot(ccp_alphas, test_scores, marker='o', label="test",drawstyle="steps-post")
# ax.legend()
# plt.show()

clf = DecisionTreeClassifier(random_state=0, ccp_alpha=0.0060)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
# plt.figure(figsize=(12,8))
# tree.plot_tree(clf,rounded=True,filled=True)
# plt.show()

"""
Post Purning accuracy is : 0.8910256410256411
"""
# print("decistion tree classification accuracy:",accuracy_score(y_test, y_pred))


# text_representation = tree.export_text(clf)
# print(text_representation)

# fig = plt.figure(figsize=(25,20))
# _ = tree.plot_tree(clf,
#                   feature_names=['cholesterol','glucose','hdl_chol','age','weight','systolic_bp','diastolic_bp'],
#                    class_names=['diabetic','not diabetic'],
#                    filled=True)
# fig.savefig("decistion_tree_classification.png")


def predict_diabetes_DT(
    cholesterol, glucose, hdl_chol, age, weight, systolic_bp, diastolic_bp
):
    user_inputs = np.array(
        [
            cholesterol,
            glucose,
            hdl_chol,
            age,
            float(weight) * 2.2,
            systolic_bp,
            diastolic_bp,
        ]
    ).reshape(1, -1)
    user_df = pd.DataFrame(
        user_inputs,
        columns=[
            "cholesterol",
            "glucose",
            "hdl_chol",
            "age",
            "weight",
            "systolic_bp",
            "diastolic_bp",
        ],
    )

    prediction = clf.predict(user_df)

    return prediction[0]


###########################################################################################################################

# Regression Tree
# regressor = DecisionTreeRegressor()
# regressor.fit(X_train, y_train)

# y_pred = regressor.predict(X_test)

# y_pred_class = np.round(y_pred)
# y_test_class = np.round(y_test)

# mse = mean_squared_error(y_test, y_pred)
# accuracy = accuracy_score(y_test_class, y_pred_class)
# print("Mean Squared Error:", mse)
# print("decision tree regressor accuracy:", accuracy)


# text_representation = tree.export_text(regressor)
# print(text_representation)

# fig = plt.figure(figsize=(25, 20))
# _ = tree.plot_tree(
#     regressor,
#     feature_names=[
#         "cholesterol",
#         "glucose",
#         "hdl_chol",
#         "age",
#         "weight",
#         "systolic_bp",
#         "diastolic_bp",
#     ],
#     filled=True,
# )
# plt.show()
# fig.savefig("decision_tree_regression.png")

# # Post-Purning operation
# path = regressor.cost_complexity_pruning_path(X_train, y_train)
# # path variable gives two things ccp_alphas and impurities
# ccp_alphas, impurities = path.ccp_alphas, path.impurities
# print("ccp alpha wil give list of values :", ccp_alphas)
# print("***********************************************************")
# print("Impurities in Decision Tree :", impurities)

# clfs = []  # will store all the models here
# for ccp_alpha in ccp_alphas:
#     clf = DecisionTreeRegressor(random_state=0, ccp_alpha=ccp_alpha)
#     clf.fit(X_train, y_train)
#     clfs.append(clf)
# print(
#     "Last node in Decision tree is {} and ccp_alpha for last node is {}".format(
#         clfs[-1].tree_.node_count, ccp_alphas[-1]
#     )
# )

# train_scores = [clf.score(X_train, y_train) for clf in clfs]
# test_scores = [clf.score(X_test, y_test) for clf in clfs]
# fig, ax = plt.subplots()
# ax.set_xlabel("alpha")
# ax.set_ylabel("accuracy")
# ax.set_title("Accuracy vs alpha for training and testing sets")
# ax.plot(ccp_alphas, train_scores, marker="o", label="train", drawstyle="steps-post")
# ax.plot(ccp_alphas, test_scores, marker="o", label="test", drawstyle="steps-post")
# ax.legend()
# plt.show()

# clf = DecisionTreeClassifier(random_state=0, ccp_alpha=0.0030)
# clf.fit(X_train, y_train)
# y_pred = clf.predict(X_test)
# plt.figure(figsize=(12, 8))
# tree.plot_tree(clf, rounded=True, filled=True)
# plt.show()

# """
# Post Purning accuracy is : 0.8910256410256411
# """
# print("decistion tree regrission accuracy:", accuracy_score(y_test, y_pred))


# text_representation = tree.export_text(clf)
# print(text_representation)

# fig = plt.figure(figsize=(25, 20))
# _ = tree.plot_tree(
#     clf,
#     feature_names=[
#         "cholesterol",
#         "glucose",
#         "hdl_chol",
#         "age",
#         "weight",
#         "systolic_bp",
#         "diastolic_bp",
#     ],
#     class_names=["diabetic", "not diabetic"],
#     filled=True,
# )
# fig.savefig("decistion_tree_classification.png")
