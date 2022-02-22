from pyexpat import features
import streamlit as st
from joblib import load
import pandas as pd
from utils import Transformador


#definindo a função para avaliação de crédito
def avaliar_mau(dict_respostas):
    modelo = load('objetos/modelo.joblib')
    features = load('objetos/features.joblib')

    if dict_respostas['Anos_desempregado'] > 0:
        dict_respostas['Anos_empregado'] = dict_respostas['Anos_desempregado'] * -1 #transformação para o modelo entender 

    #criando um dataframe para as respostas
    respostas = []
    for coluna in features:
        respostas.append(dict_respostas[coluna])

    df_novo_cliente = pd.DataFrame(data=[respostas], columns=features)

    mau = modelo.predict(df_novo_cliente)[0] #o zero é para considerar a primeira resposta (primeira classificação)

    return mau


st.image('img/bytebank_logo.png')
st.write('# Siumulador de Avaliação de Crédito')

my_expander_1 = st.expander("Trabalho")

my_expander_2 = st.expander("Pessoal")

my_expander_3 = st.expander("Família")

dict_respostas = {}
lista_campos = load('objetos/lista_campos.joblib')


with my_expander_1:

    col1_form, col2_form = st.columns(2)

    dict_respostas['Categoria_de_renda'] = col1_form.selectbox('Qual a categoria de renda?', lista_campos['Categoria_de_renda'])

    dict_respostas['Ocupacao'] = col1_form.selectbox('Qual a ocupação?', lista_campos['Ocupacao'])

    dict_respostas['Tem_telefone_trabalho'] = 1 if col1_form.selectbox('Tem telefone de trabalho?', ['Sim', 'Não']) == 'Sim' else 0

    dict_respostas['Rendimento_Anual'] = col2_form.slider('Qual é o salário mensal?', help='Podemos mover o ponto vermelho ao longo da barra', min_value=0, max_value=35000, step=500) * 12

    dict_respostas['Anos_empregado'] = col2_form.slider('Quantos anos empregado?', help='Podemos mover a barra usando as setas do teclado', min_value=0, max_value=50, step=1)

    dict_respostas['Anos_desempregado'] = col2_form.slider('Quantos anos desempregado?', help='Podemos mover a barra usando as setas do teclado', min_value=0, max_value=50, step=1)
    


with my_expander_2:

    col3_form, col4_form = st.columns(2)

    dict_respostas['Grau_Escolaridade'] = col3_form.selectbox('Qual o grau de escolaridade?', lista_campos['Grau_Escolaridade'])

    dict_respostas['Estado_Civil'] = col3_form.selectbox('Qual o estado civil?', lista_campos['Estado_Civil'])

    dict_respostas['Tem_Carro'] = 1 if col3_form.selectbox('Tem carro?', ['Sim', 'Não']) == 'Sim' else 0

    dict_respostas['Tem_telefone_fixo'] = 1 if col4_form.selectbox('Tem telefone fixo?', ['Sim', 'Não']) == 'Sim' else 0    

    dict_respostas['Tem_email'] = 1 if col4_form.selectbox('Tem e-mail?', ['Sim', 'Não']) == 'Sim' else 0    

    dict_respostas['Idade'] = col4_form.slider('Qual a idade?', help='Podemos mover a barra usando as setas do teclado', min_value=0, max_value=50, step=1)



with my_expander_3:

    col5_form, col6_form = st.columns(2)

    dict_respostas['Moradia'] = col5_form.selectbox('Qual o tipo de moradia?', lista_campos['Moradia'])

    dict_respostas['Tem_Casa_Propria'] = 1 if col5_form.selectbox('Tem casa própria?', ['Sim', 'Não']) == 'Sim' else 0

    dict_respostas['Tamanho_Familia'] = col6_form.slider('Qual o tamanho da família?', help='Podemos mover a barra usando as setas do teclado', min_value=0, max_value=30, step=1)

    dict_respostas['Quantos_Filhos'] = col6_form.slider('Quantos filhos?', help='Podemos mover a barra usando as setas do teclado', min_value=0, max_value=20, step=1)

if st.button('Avaliar crédito'):
    if avaliar_mau(dict_respostas):
        st.error('Crédito negado')
    else:
        st.sucess('Crédito aprovado')