# Balancear
# RandomTree e RandomForest
# mostrar taxa de erros

# CLASSIFICADORES
# Pipeline
# 1.
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
dados = pd.read_csv('Aula5\\fertility_Diagnosis.txt')

from imblearn.over_sampling import SMOTE

dados_atributos = dados.drop(columns=['Diagnostico'])
dados_classes = dados['Diagnostico']

print(dados_classes)

from collections import Counter
classes_count = Counter(dados_classes)
print(classes_count)

resampler = SMOTE()
dados_atributos_b, dados_classes_b = resampler.fit_resample(dados_atributos, dados_classes)

print('Frequencia de classes após balanceamento')

classes_count = Counter(dados_classes_b)
print(classes_count)

dados_atributos_b = pd.DataFrame(dados_atributos_b)
print(dados_atributos_b)
dados_atributos_b.columns = dados.columns[:-1]

dados_classes_b = pd.DataFrame(dados_classes_b)
dados_classes_b.columns = [dados.columns[-1]]

dados_finais = dados_atributos_b.join(dados_classes_b, how='left')

#4. Treinar
from sklearn.model_selection import train_test_split
dados_atributos = dados_finais.drop(columns=['Diagnostico'])
dados_classe = dados_finais['Diagnostico']


atributos_train, atributos_test, classes_train, classes_test = train_test_split(dados_atributos, dados_classe, test_size = 0.3)

#4.3 Treinar o modelo
#Será usada uma árvore, mas que pode ser subistuída por outro indutor
# from sklearn.tree import DecisionTreeClassifier #Importar o indutor da árvore de decisão
#construir um objeto a partir do indutor
tree = RandomForestClassifier()
#treinar o modelo
fertility_tree = tree.fit(atributos_train, classes_train)
from pickle import dump
dump(fertility_tree, open("Aula6\\fertility_tree.pkl", "wb"))

#pretestar o modelo
Classe_test_predict = fertility_tree.predict(atributos_test)

#Comparar as clases inferidas no teste com as classes preservadas no split
i = 0
for i in range(0, len(classes_test)):
  print(classes_test.iloc[i][0], ' -  ', Classe_test_predict[i])

#Acurácia global do modelo
from sklearn import metrics
print('Acurácia global (provisória):', metrics.accuracy_score(classes_test,Classe_test_predict))  

#MATRIZ DE CONTINGÊNCIA
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay #para o gráfico
from sklearn.metrics import confusion_matrix
# ConfusionMatrixDisplay(fertility_tree, atributos_test, classes_test)
cm = confusion_matrix(classes_test, Classe_test_predict, labels=fertility_tree.classes_)
print(cm)

# Matrix de contingência modo gráfico
grafico = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=fertility_tree.classes_)
grafico.plot()
plt.show()

#Salvar o modelo para uso posterior
from pickle import dump
dump(fertility_tree, open('fertiliy_tree_model.pkl', 'wb'))



##################################
# OTIMIZAÇÃO DOS HIPERPARAMETROS #
##################################
import numpy as np

# Montagem da grade de parâmentros
# Número de árvores na floresta
n_estimators = [int(x) for x in np.linspace(start = 100, stop = 300, num = 3)]
# Número de atributos considerados em cada segmento
max_features = ['log2', 'sqrt']
# Número máximo de folhas em cada árvore
max_depth = [int(x) for x in np.linspace(10,110, num = 3)]
max_depth.append(None)
# Número mínimo de instâncias requeridas para segmentar cada nó
min_samples_split = [2,5,10]
# Número mínimo de amostras necessárias em cada nó
min_samples_leaf = [1,2,4]
# Método de seleção de amostras para treinar cada árvore
bootstrap = [True, False]
from sklearn.model_selection import GridSearchCV
# random_grid = {'n_estimators' : n_estimators,
#                'max_features' : max_features,
#                'max_depth' : max_depth,
#                'min_samples_split' : min_samples_split,
#                'min_samples_leaf' : min_samples_leaf,
#                'bootstrap' : bootstrap,}

# Alternative
random_grid = {'n_estimators' : n_estimators,
               'max_features' : max_features,
               'max_depth' : max_depth}

from pprint import pprint
pprint(random_grid)

# INICIAR A BUSCA PELO MELHORES HIPERPARAMETROS

# rf = instanciação da randomForest
rf_grid = GridSearchCV(tree,random_grid,refit=True,verbose=2)
rf_grid.fit(atributos_train, classes_train)

print("##### MELHORES HIPERPARÂMETROS #####")
print(type(rf_grid))
print(rf_grid.best_params_)

# Treinar com os melhores parâmetros