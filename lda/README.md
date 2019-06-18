
# Latent Dirichlent Allocation (LDA)

Made by: Cristian E. Nuno

Date: June 16, 2019

---------

This tutorial follows the [Beginners Guide to Topic Modeling in Python](https://www.analyticsvidhya.com/blog/2016/08/beginners-guide-to-topic-modeling-in-python/)

## Background

> In [natural language processing (NLP)](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3168328/), [latent Dirichlet allocation (LDA)](http://jmlr.csail.mit.edu/papers/v3/blei03a.html) is a generative statistical model that allows sets of observations to be explained by unobserved groups that explain why some parts of the data are similar. For example, if observations are words collected into documents, it posits that each document is a mixture of a small number of topics and that each word's presence is attributable to one of the document's topics. [LDA](https://scikit-learn.org/stable/modules/decomposition.html#latentdirichletallocation) is an example of a [topic model](https://cacm.acm.org/magazines/2012/4/147361-probabilistic-topic-models/fulltext#F3). - Wikipedia ([source](https://en.wikipedia.org/wiki/Latent_Dirichlet_allocation))

*Note: Hyperlinks are my own.*

## Load necessary libraries and modules

The [`gensim`](https://radimrehurek.com/gensim/) library is used for topic modeling. The [`nltk` library](https://www.nltk.org/) is one the popular Python packages used for working with text data. The `string` library is imported to make use of its [constants](https://docs.python.org/3.4/library/string.html?highlight=string%20module#string-constants).

<!-- Finally, the [`List`](https://docs.python.org/3/library/typing.html) module is useful when writing functions. -->


```python
import gensim
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import string
```

## Load necessary data


```python
doc1 = "Sugar is bad to consume. My sister likes to have sugar, but not my father."
doc2 = "My father spends a lot of time driving my sister around to dance practice."
doc3 = "Doctors suggest that driving may cause increased stress and blood pressure."
doc4 = "Sometimes I feel pressure to perform well at school, but my father never seems to drive my sister to do better."
doc5 = "Health experts say that Sugar is not good for your lifestyle."

# compile documents
doc_complete = [doc1, doc2, doc3, doc4, doc5]
```

## Cleaning and processing

### Stop Words

> Sometimes, some extremely common words which would appear to be of little value in helping select documents matching a user need are excluded from the vocabulary entirely. These words are called *stop words*. - [Dropping common terms: stop words](https://nlp.stanford.edu/IR-book/html/htmledition/dropping-common-terms-stop-words-1.html) section in [Introduction to Information Retrevial](https://nlp.stanford.edu/IR-book/information-retrieval-book.html)


```python
# install the stopwords corpus from the command line
#!python -m nltk.downloader stopwords
```

Transforming the `list` object to type [`set`](https://docs.python.org/3.7/library/stdtypes.html#set-types-set-frozenset) to speed up the testing of membership in future `if` statements.


```python
stop = set(stopwords.words('english'))
```

Note that we are transforming all words to lowercase.


```python
stop_free = " ".join([word for word in doc1.lower().split(sep=" ") if word not in stop])
stop_free
```




    'sugar bad consume. sister likes sugar, father.'



### Exclusion

Punctuation marks - i.e. `!@#$%^&*(),;:'"` - don't tell us anything about the words in a corpus. Removing them helps remove unnecessary characters from each word.


```python
exclude = set(string.punctuation)
```


```python
punc_free = "".join([char for char in stop_free if char not in exclude])
punc_free
```




    'sugar bad consume sister likes sugar father'



### Lemmatization
> *Stemming* usually refers to a crude heuristic process that chops off the ends of words in the hope of achieving this goal correctly most of the time, and often includes the removal of derivational affixes. *Lemmatization* usually refers to doing things properly with the use of a vocabulary and morphological analysis of words, normally aiming to remove inflectional endings only and to return the base or dictionary form of a word, which is known as the [*lemma*](https://simple.wikipedia.org/wiki/Lemma_(linguistics). If confronted with the token saw, stemming might return just s, whereas lemmatization would attempt to return either see or saw depending on whether the use of the token was as a verb or a noun. - [Stemming and Lemmatization](https://nlp.stanford.edu/IR-book/information-retrieval-book.html) section in [Introduction to Information Retrevial](https://nlp.stanford.edu/IR-book/information-retrieval-book.html)


```python
lemma = WordNetLemmatizer()
```


```python
# install the wordnet corpus from the command line
#!python -m nltk.downloader wordnet
```


```python
normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split())
normalized
```




    'sugar bad consume sister like sugar father'



Now let's put it all together in a function.


```python
def normalize_text(doc: str) -> str:
    """
    Takes in a raw string and returns normalized string.
    
    Normalization includes:
     - lowercase spelling
     - removing stopwords
     - removing punctuation
     - lemmatizing remaining words
    """
    stop_free = " ".join([word for word in doc.lower().split(sep=" ") if word not in stop])
    punc_free = ''.join(char for char in stop_free if char not in exclude)
    normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split())
    return normalized
```

Let's use `normalize_text()` in a list comprehension to return a list of words belonging to each document.


```python
doc_clean = [normalize_text(doc=doc).split(sep=" ") for doc in doc_complete] 
doc_clean
```




    [['sugar', 'bad', 'consume', 'sister', 'like', 'sugar', 'father'],
     ['father',
      'spends',
      'lot',
      'time',
      'driving',
      'sister',
      'around',
      'dance',
      'practice'],
     ['doctor',
      'suggest',
      'driving',
      'may',
      'cause',
      'increased',
      'stress',
      'blood',
      'pressure'],
     ['sometimes',
      'feel',
      'pressure',
      'perform',
      'well',
      'school',
      'father',
      'never',
      'seems',
      'drive',
      'sister',
      'better'],
     ['health', 'expert', 'say', 'sugar', 'good', 'lifestyle']]



## Document Term Matrix

Creating the term dictionary of our courpus, where every unique term is assigned an index.


```python
dictionary = gensim.corpora.Dictionary(doc_clean)
```

Converting list of documents (corpus) into Document Term Matrix using dictionary prepared above.


```python
doc_term_matrix = [dictionary.doc2bow(doc) for doc in doc_clean]
doc_term_matrix
```




    [[(0, 1), (1, 1), (2, 1), (3, 1), (4, 1), (5, 2)],
     [(2, 1), (4, 1), (6, 1), (7, 1), (8, 1), (9, 1), (10, 1), (11, 1), (12, 1)],
     [(8, 1),
      (13, 1),
      (14, 1),
      (15, 1),
      (16, 1),
      (17, 1),
      (18, 1),
      (19, 1),
      (20, 1)],
     [(2, 1),
      (4, 1),
      (18, 1),
      (21, 1),
      (22, 1),
      (23, 1),
      (24, 1),
      (25, 1),
      (26, 1),
      (27, 1),
      (28, 1),
      (29, 1)],
     [(5, 1), (30, 1), (31, 1), (32, 1), (33, 1), (34, 1)]]




```python
LDA = gensim.models.ldamodel.LdaModel
```


```python
model = LDA(doc_term_matrix, 
            num_topics=3,
            id2word=dictionary,
            passes=50)
```


```python
model.print_topics(num_words=3)
```




    [(0, '0.071*"father" + 0.071*"sister" + 0.041*"better"'),
     (1, '0.050*"driving" + 0.050*"pressure" + 0.050*"stress"'),
     (2, '0.126*"sugar" + 0.071*"like" + 0.071*"bad"')]



Each line is a topic with individual topic terms and weights. 

* Topic 0 can be termed as stress;
* Topic 1 can be termed as time spent driving; and
* Topic 3 can be termed as family health.

## Conclusion

LDA can be used to obtain topics from documents. To improve the LDA model, some future work may include any of the following:

* Frequency filtering;
* Part of Speech (POS) filtering; and
* Batch wise LDA
