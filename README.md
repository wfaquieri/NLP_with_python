# NLP_with_python
learn natural language processing (NLP) techniques, such as how to identify and separate words, how to extract topics in a text, and how to build your own fake news classifier. libraries: NLTK, alongside libraries which utilize deep learning to solve common NLP problems. 

## Classificação de Texto

1. Bag-of-words
2. Treinando um classificador Naive Bayes
3. Treinando um classificador de árvore de decisão
4. Treinando um classificador de entropia máxima
5. Treinando classificadores scikit-learn
6. Medindo a precisão e o recall de um classificador
7. Calculando palavras de alta informação
8. Combinando classificadores com votação
9. Classificando com vários classificadores binários
10. Treinando um classificador com o NLTK-Trainer

## Bag-of-words

O modelo do saco de palavrasé o método mais simples; Este método não se importa com a ordem das palavras, ou quantas vezes uma palavra ocorre, tudo o que importa é se a palavra está **presente** em uma lista de palavras. A idéia é converter uma lista de palavras em um \blue{dict}, onde cada palavra se torna uma chave com o valor 'True'.

``` python
def bag_of_words(words): 
  return dict([(word, True) for word in words])  
```
Podemos usar com uma lista de palavras. Setença: "the quick brown fox"

``` python
from featx import bag_of_words
bag_of_words(['the', 'quick', 'brown', 'fox'])
{'quick': True, 'brown': True, 'the': True, 'fox': True}
```
