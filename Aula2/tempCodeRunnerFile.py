dados_legiveis_final = dados_numericos_legiveis.join(dados_categoricos_normalizados, how='left')
categorias_originais = dados_categoricos_normalizados.idxmax(axis=1).str.replace('sexo_', '')
dados_legiveis_final['sexo'] = categorias_originais
dados_legiveis_final = dados_legiveis_final.drop(columns=['sexo_F','sexo_M'])
print(dados_legiveis_final.head(10))