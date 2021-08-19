import pandas as pd
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculatio
from sklearn import tree
import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB,MultinomialNB,BernoulliNB

# load dataset
col_names = ['pregnant', 'glucose', 'bp', 'skin', 'insulin', 'bmi', 'pedigree', 'age', 'label']
pima = pd.read_csv("diabetes.csv", header=None, names=col_names)

#info of data set
print(pima.head(10))
print(pima.shape)
print(pima.info(10))


#test and traininng of data set

feature_cols = ['pregnant', 'insulin', 'bmi', 'age','glucose','bp','pedigree']
X = pima[feature_cols] # Features
y = pima.label # Target+ variable
# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0) # 70% training and 20% test

#check usung KNN 

KNN=KNeighborsClassifier()
KNN.fit(X_train,y_train)
print("Accuracy of KNN Classifier on training set is  {:.2f}".format(KNN.score(X_train,y_train)))
print("Accuracy of KNN Classifier on testing set is  {:.2f}".format(KNN.score(X_test,y_test)))
predict=KNN.predict(X_test)
print(pd.DataFrame({"actual":y_test,"predict":predict}))



#check using naive byes
BNB=BernoulliNB()
BNB.fit(X_train,y_train)
print("Accuracy of BNB Classifier on training set is  {:.2f}".format(BNB.score(X_train,y_train)))
print("Accuracy of BNB Classifier on testing set is  {:.2f}".format(BNB.score(X_test,y_test)))
predict=BNB.predict(X_test)
print(pd.DataFrame({"actual":y_test,"predict":predict}))



# Create Decision Tree classifer object
clf = DecisionTreeClassifier(criterion="entropy", max_depth=3)

# Train Decision Tree Classifer
clf = clf.fit(X_train,y_train)
#Predict the response for test dataset
y_pred = clf.predict(X_test)
print(y_pred)
new_test=[[6,94,32,30,85,72,0.871],[6,94,32,30,85,72,0.871]]
print (clf.predict(new_test))

# Model Accuracy, how often is the classifier correct?

print("Accuracy of DecisionTree Classifier on training set is  {:.2f}".format(clf.score(X_train,y_train)))
print("Accuracy of DecisionTree Classifier on testing set is  {:.2f}".format(clf.score(X_test,y_test)))

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

df=pd.DataFrame({"actual":y_test,"predicted":y_pred})
print(df)

#plot the decision tree
tree.plot_tree(clf)
plt.show()
