import arff
import numpy as np

dataset = arff.load(open('Dry_Bean_Dataset.arff'),'rb')

dataset = np.array(dataset["data"])

print(dataset)
print(dataset.shape)
