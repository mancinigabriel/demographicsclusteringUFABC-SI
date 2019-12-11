# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
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

# %% [markdown]
# # Import de bibliotecas necessárias

# %%
import pandas as pd
import numpy as np
import seaborn as sns
from IPython.display import display
from sklearn.decomposition import PCA
from sklearn.preprocessing import Normalizer
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import math
from IPython.display import display
from sklearn.preprocessing import normalize

# %% [markdown]
# # Carregar dataset e tratar dos dados

# %%
# Carrega o dataset
df = pd.read_csv(r'microdados_perfil_discente_2018.csv',encoding="latin1", sep = ';')

# Seleciona colunas
df_num = df[['Qual é o seu ano de ingresso na UFABC-','Qual é a sua idade-','Qual é o tempo médio necessário, em minutos, para você comparecer à UFABC-','Você já foi reprovado em alguma disciplina- ','Você já efetuou trancamento total de matrícula-','Qual é o seu CR-','Qual é o seu CA-','Quantas horas, em média, você permanece na UFABC por semana-','Qual é a renda média bruta mensal de sua família-','Quantidade de pessoas, incluindo você, que vivem da renda média bruta mensal familiar- ','Qual é, em média, a quantidade de dinheiro que você recebe mensalmente-']]

# Cria versões numéricas das colunas
reprovacoes_num = pd.DataFrame({'Você já foi reprovado em alguma disciplina- ': df_num['Você já foi reprovado em alguma disciplina- '].unique(), 'reprovacoes': [4,5,0,7,1,6,2,3,0,0]})
trancamentos_num = pd.DataFrame({'Você já efetuou trancamento total de matrícula-': df_num['Você já efetuou trancamento total de matrícula-'].unique(), 'trancamentos': [5,0,1,2,0,3,4,0]})
tempo_ufabc_num = pd.DataFrame({'Quantas horas, em média, você permanece na UFABC por semana-': df_num['Quantas horas, em média, você permanece na UFABC por semana-'].unique(), 'tempo_UFABC': [2.5,23,31,8,18,13,28,0,0]})
renda_familia_num = pd.DataFrame({'Qual é a renda média bruta mensal de sua família-': df_num['Qual é a renda média bruta mensal de sua família-'].unique(), 'renda_familia': [3500,2500,500,8000,0,6000,4500,14000,1500,25000,17500,10500,50000,40000]})
tam_familia_num = pd.DataFrame({'Quantidade de pessoas, incluindo você, que vivem da renda média bruta mensal familiar- ': df_num['Quantidade de pessoas, incluindo você, que vivem da renda média bruta mensal familiar- '].unique(), 'tam_familia': [4,3,2,0,5,6,1,8,7,10,9]})
renda_individual_num = pd.DataFrame({'Qual é, em média, a quantidade de dinheiro que você recebe mensalmente-': df_num['Qual é, em média, a quantidade de dinheiro que você recebe mensalmente-'].unique(), 'renda_individual': [250,750,0,2500,1500,0.1,3500,4500,6000]})
tempos_trajetos_num = pd.DataFrame({'Qual é o tempo médio necessário, em minutos, para você comparecer à UFABC-': df_num['Qual é o tempo médio necessário, em minutos, para você comparecer à UFABC-'].unique(), 'tempos_trajeto': [120,22.5,52.5,90,7.5,37.5,0]})

# Substitui colunas de texto por versões numéricas, removendo a coluna de texto:
df_num = df_num.merge(tempos_trajetos_num, on='Qual é o tempo médio necessário, em minutos, para você comparecer à UFABC-')
df_num = df_num.drop('Qual é o tempo médio necessário, em minutos, para você comparecer à UFABC-',1)

df_num = df_num.merge(reprovacoes_num, on='Você já foi reprovado em alguma disciplina- ')
df_num = df_num.drop('Você já foi reprovado em alguma disciplina- ',1)

df_num = df_num.merge(trancamentos_num, on='Você já efetuou trancamento total de matrícula-')
df_num = df_num.drop('Você já efetuou trancamento total de matrícula-',1)

df_num = df_num.merge(tempo_ufabc_num, on='Quantas horas, em média, você permanece na UFABC por semana-')
df_num = df_num.drop('Quantas horas, em média, você permanece na UFABC por semana-',1)

df_num = df_num.merge(renda_familia_num, on='Qual é a renda média bruta mensal de sua família-')
df_num = df_num.drop('Qual é a renda média bruta mensal de sua família-',1)

df_num = df_num.merge(tam_familia_num, on='Quantidade de pessoas, incluindo você, que vivem da renda média bruta mensal familiar- ')
df_num = df_num.drop('Quantidade de pessoas, incluindo você, que vivem da renda média bruta mensal familiar- ',1)

df_num = df_num.merge(renda_individual_num , on='Qual é, em média, a quantidade de dinheiro que você recebe mensalmente-')
df_num = df_num.drop('Qual é, em média, a quantidade de dinheiro que você recebe mensalmente-',1)

# Seleciona colunas
df_num.columns = ['ano_ingresso','idade','CR','CA','tempo_trajeto','reprovacoes','trancamentos','tempo_UFABC','renda_familia','tam_familia','renda_individual']

# Remove respostas com campos vazios
df_teste = df_num.dropna(axis=0, how = 'any') 
print(df_teste.shape)

# Remove respostas iguais a 0 e prefiro não responder
df_teste = df_teste.loc[df_teste['tempo_trajeto'] != 0]
df_teste = df_teste.loc[df_teste['renda_familia'] != 0]
df_teste = df_teste.loc[df_teste['tam_familia'] != 0]
df_teste = df_teste.loc[df_teste['renda_individual'] != 0]
df_teste = df_teste.loc[df_teste['tempo_UFABC'] != 0]
df_teste = df_teste.loc[(df_teste['ano_ingresso'] != "Prefiro não responder") & (df_teste['idade'].empty == False)]

# Converte ano de ingresso, CR e CA para numérico
df_teste['ano_ingresso'] = pd.to_numeric(df_teste['ano_ingresso'])
CR_lista = df_teste['CR'].values.tolist()
CA_lista = df_teste['CA'].values.tolist()
CR_lista = list(map(lambda x: x.replace(',','.'),CR_lista))
CA_lista = list(map(lambda x: x.replace(',','.'),CA_lista))
df_teste['CR'] = (np.asarray(CR_lista)).astype(float)
df_teste['CA'] = (np.asarray(CA_lista)).astype(float)

#Adiciona coluna com CR arredondado
df_teste['CR_inteiro'] = df_teste['CR'].apply(lambda x: math.floor(x))

# Calcula a renda percapta = renda familia / tamanho da família
df_teste['renda_per_capita'] = df_teste['renda_familia']/df_teste['tam_familia']
df_teste = df_teste.drop((['renda_familia','tam_familia']),1)

df_teste['anos_ufabc'] = 2018 - df_teste['ano_ingresso']
df_teste['renda_per_capita_norml'] = df_teste['renda_per_capita'] / 1000 

# %% [markdown]
# # Analise de correlações

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
# Ranking correlações

sol = (corr.where(np.triu(np.ones(corr.shape), k=1).astype(np.bool))
                 .stack()
                 .sort_values(ascending=False))

display(sol)

# %% [markdown]
# # Ajuste PCA

# %%
x = normalize(df_teste[['reprovacoes','renda_per_capita_norml','idade','anos_ufabc']].to_numpy())
y = df_teste['CR'].to_numpy()
pca = PCA(n_components = 2)
x_pca = pca.fit_transform(x)

# %%
plt.figure()
plt.scatter(x=x_pca[:, 0], y=x_pca[:, 1], cmap='viridis', c=y)
plt.xlim(min(x_pca[:,0]), max(x_pca[:,0]))
plt.ylim(min(x_pca[:,1]), max(x_pca[:,1]))
plt.xlabel('PCA Axis 1')
plt.ylabel('PCA Axis 2')
plt.title('Amostras')
plt.grid(True)

# %%
KMeans(n_clusters=4).fit(x_pca, y)
kmeans = KMeans(n_clusters=4).fit(x_pca, y)
cm = kmeans.cluster_centers_
plt.figure()
plt.scatter(x = x_pca[:,0], y = x_pca[:,1], c= kmeans.labels_.astype(float), s=50, alpha=0.5)
plt.grid(True)

plt.scatter(x=cm[:,0], y=cm[:,1], c='r', s=150, marker='X', label='Centroid')
plt.legend()

# %%
x_df = df_teste[['CR']]
y_df = df_teste[['idade']]

plt.figure()
plt.scatter(x=x_df, y=y_df, cmap='viridis')
plt.xlabel('CR')
plt.ylabel('IDADE')
plt.title('Amostras')
plt.grid(True)

# %%
x_df = df_teste[['ano_ingresso']]
y_df = df_teste[['idade']]

plt.figure()
plt.scatter(x=x_df, y=y_df, cmap='viridis')
plt.xlabel('Ano de Ingresso')
plt.ylabel('Idade')
plt.title('Amostras')
plt.grid(True)

# %%
x_df = df_teste['ano_ingresso']
y_df = df_teste['reprovacoes']
colors = df_teste['CR'].to_numpy()

plt.figure()
plt.scatter(x=x_df, y=y_df, c=colors, cmap='viridis')
plt.xlabel('ano')
plt.ylabel('reprova')
plt.title('Amostras')
plt.grid(True)

# %%

x = df_teste[['reprovacoes']]
y = df_teste[['idade']]

x = x.to_numpy()
y = y.to_numpy()

pca = PCA(n_components = 1)
x_pca = pca.fit_transform(x)


KMeans(n_clusters=4).fit(x, y)
kmeans = KMeans(n_clusters=4).fit(x, y)
centroids = kmeans.cluster_centers_
print(centroids)
plt.figure()
plt.scatter(y[:,0], x[:,0], c= kmeans.labels_.astype(float), s=50, alpha=0.5)

x = df_teste[['reprovacoes', 'CR']]
y = df_teste[['idade']]

x = x.to_numpy()
y = y.to_numpy()

pca = PCA(n_components = 1)
x_pca = pca.fit_transform(x)

KMeans(n_clusters=4).fit(x, y)
kmeans = KMeans(n_clusters=4).fit(x, y)
centroids = kmeans.cluster_centers_
print(centroids)
plt.figure()
plt.scatter(y[:,0], x[:,1], c= kmeans.labels_.astype(float), s=50, alpha=0.5)
