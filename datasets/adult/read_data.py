from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from prettytable import PrettyTable

workclass = ["Private", "Self-emp-not-inc", "Self-emp-inc", "Federal-gov", "Local-gov", "State-gov", "Without-pay", "Never-worked"]
education = ["Bachelors", "Some-college", "11th", "HS-grad", "Prof-school", "Assoc-acdm", "Assoc-voc", "9th", "7th-8th", "12th", "Masters", "1st-4th", "10th", "Doctorate", "5th-6th", "Preschool"]
maritalstatus = ["Married-civ-spouse", "Divorced", "Never-married", "Separated", "Widowed", "Married-spouse-absent", "Married-AF-spouse"]
occupation = ["Tech-support", "Craft-repair", "Other-service", "Sales", "Exec-managerial", "Prof-specialty", "Handlers-cleaners", "Machine-op-inspct", "Adm-clerical", "Farming-fishing", "Transport-moving", "Priv-house-serv", "Protective-serv", "Armed-Forces"]
relationship = ["Wife", "Own-child", "Husband", "Not-in-family", "Other-relative", "Unmarried"]
race = ["White", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other", "Black"]
sex = ["Female", "Male"]
nativecountry = ["United-States", "Cambodia", "England", "Puerto-Rico", "Canada", "Germany", "Outlying-US(Guam-USVI-etc)", "India", "Japan", "Greece", "South", "China", "Cuba", "Iran", "Honduras", "Philippines", "Italy", "Poland", "Jamaica", "Vietnam", "Mexico", "Portugal", "Ireland", "France", "Dominican-Republic", "Laos", "Ecuador", "Taiwan", "Haiti", "Columbia", "Hungary", "Guatemala", "Nicaragua", "Scotland", "Thailand", "Yugoslavia", "El-Salvador", "Trinadad&Tobago", "Peru", "Hong", "Holand-Netherlands"]
category = ["<=50K",">50K"]

#age,workclass,fnlwgt,education,education-num,marital-status,occupation,relationship,race,sex,capital-gain,capital-loss,hours-per-week,native-country

dataset = list()

with open("adult.data") as fp:
    for i in fp:
        if(len(i) > 2 and i.find("?") == -1):
            line = i[:-1].replace(" ","").split(",")
            line[1] = workclass.index(line[1])
            line[3] = education.index(line[3])
            line[5] = maritalstatus.index(line[5])
            line[6] = occupation.index(line[6])
            line[7] = relationship.index(line[7])
            line[8] = race.index(line[8])
            line[9] = sex.index(line[9])
            line[13] = nativecountry.index(line[13])
            line[14] = category.index(line[14])
            dataset.append(line)
#
dataset = np.array(dataset).astype("float")
X = dataset[:,:14] # atributos
Y = dataset[:,14].astype(int) # categorias (devem ser inteiros)
#
print(dataset.shape, X.shape, Y.shape)
#
# Treino e verificacao de acuracia com todos os dados (somente para teste)
#
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X,Y)
Y_pred = knn.predict(X)
print(classification_report(Y, Y_pred))
#
# Exemplo de Hold-out
#
X_train_val, X_test, Y_train_val, Y_test = train_test_split(X, Y, test_size=0.3)
X_train, X_val, Y_train, Y_val = train_test_split(X_train_val, Y_train_val, test_size=0.28)
#
print(X_train.shape, Y_train.shape)
print(X_val.shape, Y_val.shape)
print(X_test.shape, Y_test.shape)
#
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train,Y_train)
print("## Treino ##")
Y_train_pred = knn.predict(X_train)
print(classification_report(Y_train, Y_train_pred))
print("## Teste ##")
Y_test_pred = knn.predict(X_test)
print(classification_report(Y_test, Y_test_pred))
print("## Validacao ##")
Y_val_pred = knn.predict(X_val)
print(classification_report(Y_val, Y_val_pred))
#
# Exemplo de cross-validation com otimizacao de hiperparametro
#
parametros = {'metric':('cosine', 'minkowski', 'manhattan'), 'n_neighbors':[ 7, 10, 13]}
#parametros = {'metric':('cosine', 'minkowski'), 'n_neighbors':[7, 10, 13]}
knn = KNeighborsClassifier()
search = GridSearchCV(knn, parametros, verbose=2, n_jobs=4)
search.fit(X_train_val, Y_train_val)
#
tabela = PrettyTable()
#
linha = list()
linha.append("")
for i in range(search.cv_results_['split0_test_score'].shape[0]):
    linha.append(i)
tabela.field_names = linha
for j in search.cv_results_.keys():
    if(j == 'params'):
        continue
    linha = list()
    linha.append(j)
    for i in range(search.cv_results_['split0_test_score'].shape[0]):
        if(isinstance(search.cv_results_[j][i], float)):
            linha.append("{:5.2f}".format(search.cv_results_[j][i]))
        else:
            linha.append("{}".format(search.cv_results_[j][i]))
    tabela.add_row(linha)
#
print(tabela)
print("Best params:")
print(search.best_params_)
Y_test_pred = search.best_estimator_.predict(X_test)
print(classification_report(Y_test, Y_test_pred))
