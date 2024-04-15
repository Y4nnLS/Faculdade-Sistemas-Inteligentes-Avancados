# CLASSIFICADORES
# Pipeline
# 1.
import pandas as pd
dados = pd.read_csv('Aula5\\fertility_Diagnosis.txt')

from imblearn.over_sampling import SMOTE

dados_Atribuidos = dados.drop(columns=['Diagnostico'])
dados_classes = dados['Diagnostico']

print(dados_classes)

from collections import Counter
classes_count = Counter(dados_classes)
print(classes_count)

resampler = SMOTE()
dados_atribuidos_b, dados_classes_b = resampler.fit_resample(dados_Atribuidos, dados_classes)

print('Frequencia de classes após balanceamento')
classes_count = Counter(dados_classes_b)
print(classes_count)