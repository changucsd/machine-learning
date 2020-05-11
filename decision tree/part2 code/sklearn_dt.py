#
# authors:
# Ziqiao Gao 2157371827
# He Chang 5670527576
# Fanlin Qin 5317973858
#


import pandas as pd
import numpy as np
from sklearn import tree, metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO

# import pydotplus

df = pd.read_csv("dt.csv")
print (df)

labelEncoder = LabelEncoder()
df = df.apply(LabelEncoder().fit_transform)

print (df)

X = df.iloc[:,:-1]
print (X)
y = df.iloc[:,6]
print (y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test

print (list(X.columns))
regressor = tree.DecisionTreeClassifier(criterion="entropy")
regressor.fit(X_train,y_train)
y_pred = regressor.predict(X_test)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

X_in = [2, 0, 0, 0, 0, 0]
predicted = regressor.predict([X_in])

print (predicted)
print (regressor.score(X, y))

# r = export_text(regressor, feature_names=list(X.columns))
# print (r)

# dot_data = StringIO()
# export_graphviz(regressor, out_file= dot_data,
#                 filled=True, rounded=True,
#                 special_characters=True, feature_names=list(X.columns),class_names=['0','1'])
# graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
# graph.write_png('tree.png')