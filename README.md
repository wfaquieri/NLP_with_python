# NLP with Python

![](capa.png)

Resumo de algumas técnicas de natural language processing (NLP). Bibliotecas: NLTK, além de bibliotecas que utilizam deep learning para resolver problemas comuns de NLP. Muitas das notas contidas aqui foram retiradas do livro "Natural Language Processing: Python and NLTK" de Hardeniya et al.



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

Bag-of-words é o método mais simples; Este método não se importa com o ordenamento das palavras, ou quantas vezes uma palavra ocorre, tudo o que importa é se a palavra está **presente** em uma lista de palavras. A idéia é converter uma lista de palavras em um dict, onde cada palavra se torna uma chave com o valor 'True'.

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
Um dos problemas desse método é tratar todas as palavras igualmente, pois algumas palavras podem caracterizar mais o texto/documento do que outras. Além disso, é necessário retirar palavras conhecidas como 'stopwords'. Aquelas palavras que não agregam muito semanticamente à análise, tais como artigos, preposições, etc. a biblioteca NLTK já nos fornece uma função 'stopwords' que aplica esse tipo de tratamento:

``` python
def bag_of_words_not_in_set(words, badwords):
  return bag_of_words(set(words) - set(badwords))

from featx import bag_of_words_not_in_set
bag_of_words_not_in_set(['the', 'quick', 'brown', 'fox'], ['the'])
{'quick': True, 'brown': True, 'fox': True}

# Stopwords
from nltk.corpus import stopwords

def bag_of_non_stopwords(words, stopfile='english'):
  badwords = stopwords.words(stopfile)
  return bag_of_words_not_in_set(words, badwords)

# Note que o artigo 'the' foi excluido da bag-of-wrods
from featx import bag_of_non_stopwords
bag_of_non_stopwords(['the', 'quick', 'brown', 'fox'])
{'quick': True, 'brown': True, 'fox': True}

``` 
Agora já temos um array de palavras únicas ou um conjunto de unigramas. Todavia, em alguns contextos, pode fazer mais sentido combinar palavras, através de bigramas ou trigramas, ao invés de analisar cada palavra isoladamente. Podemos utilizar 'BigramCollocationFinder' para encontrar bigramas significativos:

```  Python
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures

def bag_of_bigrams_words(words, score_fn=BigramAssocMeasures.chi_sq, n=200):
  bigram_finder = BigramCollocationFinder.from_words(words)
  bigrams = bigram_finder.nbest(score_fn, n)
  return bag_of_words(words + bigrams)
  
from featx import bag_of_bigrams_words
bag_of_bigrams_words(['the', 'quick', 'brown', 'fox'])
{'brown': True, ('brown', 'fox'): True, ('the', 'quick'): 
True, 'fox': True, ('quick', 'brown'): True, 'quick': True, 'the': True}

``` 
## Treinando um classificador Naive Bayes
