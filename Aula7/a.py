##################################
# OTIMIZAÇÃO DOS HIPERPARAMETROS #
##################################
import numpy as np

# Montagem da grade de parâmentros
# Número de árvores na floresta
n_estimators = [int(x) for x in np.linspace(start = 100, stop = 300, num = 3)]
# Número de atributos considerados em cada segmento
max_features = ['auto', 'srqt']
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
rf_grid = GridSearchCV(rf,random_grid,refit=True,verbose=2)
rf_grid.fit(Atr_train_b, Class_train_b)

print("##### MELHORES HIPERPARÂMETROS #####")
print(type(rf_grid))
print(rf_grid.best_params_)

# Treinar com os melhores parâmetros