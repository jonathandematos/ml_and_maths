import numpy as np

dataset = list()

with open("wine.data") as fp:
    for i in fp:
        if(len(i) > 2 and i.find("?") == -1):
            line = i[:-1].replace(" ","").split(",")
            dataset.append(line)

dataset = np.array(dataset)
#dataset = np.array(dataset).astype("float")
print(dataset)
