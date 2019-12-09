# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.3.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# %%
df = pd.read_csv(r'microdados_perfil_discente_2018.csv',encoding="latin1", sep = ';')

# %%
df_num = df[['Qual é o seu ano de ingresso na UFABC-','Qual é a sua idade-','Qual é o tempo médio necessário, em minutos, para você comparecer à UFABC-','Você já foi reprovado em alguma disciplina- ','Você já efetuou trancamento total de matrícula-','Qual é o seu CR-','Qual é o seu CA-','Quantas horas, em média, você permanece na UFABC por semana-','Qual é a renda média bruta mensal de sua família-','Quantidade de pessoas, incluindo você, que vivem da renda média bruta mensal familiar- ','Qual é, em média, a quantidade de dinheiro que você recebe mensalmente-']]

# %%
reprovacoes_num = pd.DataFrame({'Você já foi reprovado em alguma disciplina- ': df_num['Você já foi reprovado em alguma disciplina- '].unique(), 'reprovacoes': [4,5,0,7,1,6,2,3,0,0]})

# %%
trancamentos_num = pd.DataFrame({'Você já efetuou trancamento total de matrícula-': df_num['Você já efetuou trancamento total de matrícula-'].unique(), 'trancamentos': [5,0,1,2,0,3,4,0]})

# %%
tempo_ufabc_num = pd.DataFrame({'Quantas horas, em média, você permanece na UFABC por semana-': df_num['Quantas horas, em média, você permanece na UFABC por semana-'].unique(), 'tempo_UFABC': [2.5,23,31,8,18,13,28,0,0]})

# %%
renda_familia_num = pd.DataFrame({'Qual é a renda média bruta mensal de sua família-': df_num['Qual é a renda média bruta mensal de sua família-'].unique(), 'renda_familia': [3500,2500,500,8000,0,6000,4500,14000,1500,25000,17500,10500,50000,40000]})

# %%
tam_familia_num = pd.DataFrame({'Quantidade de pessoas, incluindo você, que vivem da renda média bruta mensal familiar- ': df_num['Quantidade de pessoas, incluindo você, que vivem da renda média bruta mensal familiar- '].unique(), 'tam_familia': [4,3,2,0,5,6,1,8,7,10,9]})

# %%
renda_individual_num = pd.DataFrame({'Qual é, em média, a quantidade de dinheiro que você recebe mensalmente-': df_num['Qual é, em média, a quantidade de dinheiro que você recebe mensalmente-'].unique(), 'renda_individual': [250,750,0,2500,1500,0.1,3500,4500,6000]})
# %%
tempos_trajetos_num = pd.DataFrame({'Qual é o tempo médio necessário, em minutos, para você comparecer à UFABC-': df_num['Qual é o tempo médio necessário, em minutos, para você comparecer à UFABC-'].unique(), 'tempos_trajeto': [120,22.5,52.5,90,7.5,37.5,0]})

# %%
df_num = df_num.merge(tempos_trajetos_num, on='Qual é o tempo médio necessário, em minutos, para você comparecer à UFABC-')
df_num = df_num.drop('Qual é o tempo médio necessário, em minutos, para você comparecer à UFABC-',1)

# %%
df_num = df_num.merge(reprovacoes_num, on='Você já foi reprovado em alguma disciplina- ')
df_num = df_num.drop('Você já foi reprovado em alguma disciplina- ',1)

# %%
df_num = df_num.merge(trancamentos_num, on='Você já efetuou trancamento total de matrícula-')
df_num = df_num.drop('Você já efetuou trancamento total de matrícula-',1)

# %%
df_num = df_num.merge(tempo_ufabc_num, on='Quantas horas, em média, você permanece na UFABC por semana-')
df_num = df_num.drop('Quantas horas, em média, você permanece na UFABC por semana-',1)

# %%
df_num = df_num.merge(renda_familia_num, on='Qual é a renda média bruta mensal de sua família-')
df_num = df_num.drop('Qual é a renda média bruta mensal de sua família-',1)

# %%
df_num = df_num.merge(tam_familia_num, on='Quantidade de pessoas, incluindo você, que vivem da renda média bruta mensal familiar- ')
df_num = df_num.drop('Quantidade de pessoas, incluindo você, que vivem da renda média bruta mensal familiar- ',1)

# %%
df_num = df_num.merge(renda_individual_num , on='Qual é, em média, a quantidade de dinheiro que você recebe mensalmente-')
df_num = df_num.drop('Qual é, em média, a quantidade de dinheiro que você recebe mensalmente-',1)

# %%
df_num.columns = ['ano_ingresso','idade','CR','CA','tempo_trajeto','reprovacoes','trancamentos','tempo_UFABC','renda_familia','tam_familia','renda_individual']

# %%
df_teste = df_num.dropna(axis=0, how = 'any') 
print(df_teste.shape)

# %%
df_teste = df_teste.loc[df_teste['tempo_trajeto'] != 0]
print(df_teste.shape)

# %%
df_teste = df_teste.loc[df_teste['renda_familia'] != 0]
print(df_teste.shape)

# %%
df_teste = df_teste.loc[df_teste['tam_familia'] != 0]
print(df_teste.shape)

# %%
df_teste = df_teste.loc[df_teste['renda_individual'] != 0]
print(df_teste.shape)

# %%
df_teste = df_teste.loc[df_teste['tempo_UFABC'] != 0]
print(df_teste.shape)

# %%
df_teste = df_teste.loc[(df_teste['ano_ingresso'] != "Prefiro não responder") & (df_teste['idade'].empty == False)]
print(df_teste.shape)

# %%
df_teste['ano_ingresso'] = pd.to_numeric(df_teste['ano_ingresso'])

# %%
CR_lista = df_teste['CR'].values.tolist()
CA_lista = df_teste['CA'].values.tolist()

# %%
df_teste['renda_per_capita'] = df_teste['renda_familia']/df_teste['tam_familia']

# %%
df_teste = df_teste.drop((['renda_familia','tam_familia']),1)

# %%
CR_lista = list(map(lambda x: x.replace(',','.'),CR_lista))
CA_lista = list(map(lambda x: x.replace(',','.'),CA_lista))

# %%
df_teste['CR'] = (np.asarray(CR_lista)).astype(float)
df_teste['CA'] = (np.asarray(CA_lista)).astype(float)

# %%
#Correlação
df_teste.dtypes

# %%
corr = df_teste.corr()
ax = sns.heatmap(
    corr, 
    vmin=-1, vmax=1, center=0,
    cmap=sns.diverging_palette(20, 220, n=200),
    square=True
)
ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=45,
    horizontalalignment='right'
);

# %%
### Transformando dataframe em array ###
x = df_teste[['reprovacoes','renda_per_capita','idade','ano_ingresso']]
y = df_teste['CR']

# %%
y.shape
# %%
x = x.to_numpy()
y = y.to_numpy()

# %%
x.shape

# %%
pca = PCA(n_components = 2)
x_pca = pca.fit_transform(x)
print(x_pca.shape)

# %%
x_pca

# %%
plt.figure()
plt.scatter(x=x_pca[:, 0], y=x_pca[:, 1], cmap='viridis')
plt.xlim(min(x_pca[:,0]), max(x_pca[:,0]))
plt.ylim(min(x_pca[:,1]), max(x_pca[:,1]))
plt.xlabel('PCA Axis 1')
plt.ylabel('PCA Axis 2')
plt.title('Amostras')
plt.grid(True)
plt.legend()

# %%
x_pca[:,0]

df_teste

