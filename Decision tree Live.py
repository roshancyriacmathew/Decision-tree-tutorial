#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree


# In[2]:


iris = datasets.load_iris()
X = iris.data
y = iris.target


# In[3]:


clf = tree.DecisionTreeClassifier()
clf = clf.fit(X,y)


# In[9]:


fig = plt.figure(figsize=(25,20))
tree.plot_tree(clf, filled = True, rounded = True, max_depth=3)
plt.show()


# In[10]:


get_ipython().system('pip install graphviz')


# In[12]:


import graphviz
data = tree.export_graphviz(clf,out_file = None,
                          feature_names=iris.feature_names,
                          class_names = iris.target_names,
                          filled=True)
graph = graphviz.Source(data, format='png')
graph


# In[13]:


graph.render("decision_tree_graphviz")


# In[14]:


from dtreeviz.trees import dtreeviz
tree = dtreeviz(clf, X, y,
               target_name = "target",
               feature_names = iris.feature_names,
               class_names = list(iris.target_names))
tree


# In[15]:


tree.save("decision_tree.svg")

