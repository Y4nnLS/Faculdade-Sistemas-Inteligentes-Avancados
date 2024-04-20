# CLASSIFICADORES
# Pipeline
# 1.
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
from sklearn.tree import DecisionTreeClassifier #Importar o indutor da árvore de decisão
#construir um objeto a partir do indutor
tree = DecisionTreeClassifier()
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

#AVALIAÇÃO DA ACURÁCIA COM CROSS-VALIDATION
from pprint import pprint
#1. Quando a avaliação é realizada com Cross_validation, dipensa-se o Split da base
from sklearn.tree import DecisionTreeClassifier
#Construir um objeto para representar o indutor
tree = DecisionTreeClassifier() #Construir o objeto do indutor

#2. Treinar o modelo
#2.1 Requitos:
#    a) dados normalizados e balancedados
#    b) dados segmentados em atributo e classes
fertility_tree_cross = tree.fit(dados_atributos_b, dados_classes_b)

#3. Avaliar a acurácia com Cross-Validation
from sklearn.model_selection import cross_validate, cross_val_score
scoring = ['precision_macro', 'recall_macro']
scores_cross = cross_validate(tree,dados_atributos_b,dados_classes_b, cv =10, scoring = scoring)
print(scores_cross['test_precision_macro'].mean())
print(scores_cross['test_recall_macro'].mean())
# scores_cross_val= cross_val_score(tree, dados_atributos_b, dados_classes_b, cv=10)
# print(scores_cross_val.mean(), ' - ', scores_cross_val.std())

#Matriz de contingência com o modelo avaliado com o cross-validation
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay #para o gráfico
from sklearn.metrics import confusion_matrix
# ConfusionMatrixDisplay(fertility_tree_cross, dados_atributos_b,dados_classes_b)
# plt.show

#SALVAR O MODELO PARA USO POSTERIOR
from pickle import dump
dump(fertility_tree_cross, open('Aula6/fertility_tree_cross.pkl', 'wb'))
