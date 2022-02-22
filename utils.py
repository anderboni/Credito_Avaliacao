#funções úteis para o projeto


#criando a classe "transformador"

#imports para pré-processamento. Escalonamento das variáveis

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler #nosso modelo não sabe qual valor é maior ou menor. O Scaler diferencia, por exemplo, quem tem um carro e quem tem 10 carros, colocando numa régua.
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd

# PEP8 padrão CamelCase (aplicando nos dados reais após aprender no FIT)

class Transformador(BaseEstimator, TransformerMixin):
    def __init__( self, colunas_continuas, colunas_categoricas):
        self.colunas_continuas = colunas_continuas
        self.colunas_categoricas = colunas_categoricas
        self.enc = OneHotEncoder()
        self.scaler = MinMaxScaler()

    def fit(self, X, y = None ):
        self.enc.fit(X[self.colunas_categoricas])
        self.scaler.fit(X[self.colunas_continuas])
        return self 

    def transform(self, X, y = None):
      
      X_categoricas = pd.DataFrame(data=self.enc.transform(X[self.colunas_categoricas]).toarray(),
                                  columns= self.enc.get_feature_names(self.colunas_categoricas))
      
      X_continuas = pd.DataFrame(data=self.scaler.transform(X[self.colunas_continuas]),
                                  columns= self.colunas_continuas)
      
      X = pd.concat([X_continuas, X_categoricas], axis=1)

      return X