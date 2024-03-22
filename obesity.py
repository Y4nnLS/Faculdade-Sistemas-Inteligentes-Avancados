import pandas as pd
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from pickle import dump
# Normalizar os dados

dados = pd.read_csv('C:\\Users\\yann_\\OneDrive\\Documentos\\GitHub\\Faculdade-Sistemas-Inteligentes-Avancados\\Aula15-03\\ObesityDataSet_raw_and_data_sinthetic.csv')
print(dados.head(5))

# Separar atributos numéricos e categóricos
numeric_features = dados.drop(columns=['Gender', 'family_history_with_overweight', 'FAVC', 'CAEC', 'SMOKE', 'SCC', 'CALC', 'MTRANS', 'NObeyesdad'])
categorical_features = dados[['Gender', 'family_history_with_overweight', 'FAVC', 'CAEC', 'SMOKE', 'SCC', 'CALC', 'MTRANS', 'NObeyesdad']]

# Pipeline para pré-processamento
numeric_transformer = Pipeline(steps=[
    ('scaler', MinMaxScaler())
])

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder())
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features.columns),
        ('cat', categorical_transformer, categorical_features.columns)
    ])

# Normalizar dados
processed_data = preprocessor.fit_transform(dados)

# Salvar modelo normalizador
dump(preprocessor, open('normalizador.pkl', 'wb'))

# Convertendo de volta para DataFrame
columns = list(numeric_features.columns) + list(preprocessor.named_transformers_['cat']['onehot'].get_feature_names_out(categorical_features.columns))
normalized_dados = pd.DataFrame(processed_data, columns=columns)

# Exibindo os dados normalizados
print(normalized_dados.head(5))