# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 18:04:00 2020

@author: rayll
"""

import pandas as pd
import matplotlib.pyplot as plt

data=pd.read_csv('C:/Users/rayll/Documents/Mestrado - geohazard/Ciencia de dados-PCS5787/Projeto/Aviacao/Opendata AIG Brazil/CENIPA-integrado-v4.csv')

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesClassifier
from sklearn import metrics

#SEPARATING FEATURES AND TARGET
f=['quantidade_fatores','ocorrencia_cidade','ocorrencia_uf','ocorrencia_aerodromo','ocorrencia_ano','ocorrencia_mes','ocorrencia_horas','aeronave_tipo_veiculo','aeronave_fabricante','aeronave_modelo','aeronave_motor_tipo','aeronave_motor_quantidade','aeronave_pmd','aeronave_assentos','aeronave_ano_fabricacao','aeronave_registro_categoria','aeronave_registro_segmento','aeronave_voo_origem','aeronave_voo_destino','aeronave_fase_operacao','aeronave_tipo_operacao']
features=data[f]
target=data['ocorrencia_classificacao']

#PRE-PROCESSING: TRANSFORMING STR INTO INT
cod=preprocessing.LabelEncoder()
for n in features.columns:
    features[n]=cod.fit_transform(d[n])

#SEPARATING 10% OF DATA FOR TESTING THE MODEL (ftr=feature, cl=class)
ftr_train, ftr_test, cl_train, cl_test = train_test_split(features, target, test_size=0.1)

#TRAINING
forest = ExtraTreesClassifier(n_estimators=250,random_state=0)
forest.fit(ftr_train, cl_train)

#CALCULATING THE IMPORTANCE OF FEATURES
imp = forest.feature_importances_
importances=pd.DataFrame()
importances['Features']=f
importances['Importance']=imp
print(importances)
importances.to_csv('C:/Users/rayll/Documents/Mestrado - geohazard/Ciencia de dados-PCS5787/Projeto/Aviacao/Opendata AIG Brazil/CENIPA-forest-importancia_atributos.csv',index=False)
#PLOTTING THE IMPORTANCES
plt.figure(figsize=(5,4))
plt.style.use('bmh')
plt.title('Importância dos Atributos na Classificação da Ocorrência',fontsize=12)
importances=importances.sort_values(['Importance'])
plt.barh(importances['Features'],importances['Importance'], color=(0.1,0.5,0.5,0.5))
plt.savefig('C:/Users/rayll/Documents/Mestrado - geohazard/Ciencia de dados-PCS5787/Projeto/Aviacao/Opendata AIG Brazil/CENIPA-forest-importancia_atributos.jpg',dpi=200,bbox_inches='tight')

#TESTING
prediction=forest.predict(ftr_test)

#ACCURACY
acc=[metrics.accuracy_score(cl_test,prediction)]
acc=pd.DataFrame(acc,columns = ['Acuracia'])
acc.to_csv('C:/Users/rayll/Documents/Mestrado - geohazard/Ciencia de dados-PCS5787/Projeto/Aviacao/Opendata AIG Brazil/CENIPA-forest-acuracia.csv',index=False)
print(acc)