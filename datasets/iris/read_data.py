import numpy as np

dataset = list()

category = ["Iris-setosa","Iris-versicolor","Iris-virginica"]

with open("iris.data") as fp:
    for i in fp:
        if(len(i) > 2 and i.find("?") == -1):
            line = i[:-1].replace(" ","").split(",")
            line[4] = category.index(line[4])
            dataset.append(line)

dataset = np.array(dataset)
#dataset = np.array(dataset).astype("float")
print(dataset)
