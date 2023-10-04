import numpy as np
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.metrics import classification_report

dataset = list()

category = ["Iris-setosa","Iris-versicolor","Iris-virginica"]

with open("iris.data") as fp:
    for i in fp:
        if(len(i) > 2 and i.find("?") == -1):
            line = i[:-1].replace(" ","").split(",")
            line[4] = category.index(line[4])
            dataset.append(line)

dataset = np.array(dataset)





np.random.shuffle(dataset)

nearest = KNN(n_neighbors=3)
nearest.fit(dataset[0:100,:-1].astype("float"), dataset[0:100,-1].astype("int"))

preds = nearest.predict(dataset[100:,:-1].astype(float))

print(classification_report(dataset[100:,-1].astype("int"), preds))

#dataset = np.array(dataset).astype("float")
#print(dataset)
