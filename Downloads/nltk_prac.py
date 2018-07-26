#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 18 15:44:01 2018

@author: niladri
"""

import nltk
#need to download all of nltk first before use
#nltk.download()

#Use of Tokenizer
from nltk.tokenize import sent_tokenize, word_tokenize

example_text1 = "Hello there, how are you doing today? The weather is fine today and Python is awesome. The sky is pinkish-blue. You should not eat cardboards." 

example_text2 = "Hello Mr. Gandu, how are you doing today? The weather is fine today and Python is awesome. The sky is pinkish-blue. You should not eat cardboards." 

print (sent_tokenize(example_text1))
print (sent_tokenize(example_text2))
print (word_tokenize(example_text1))
print (word_tokenize(example_text2))

for i in word_tokenize(example_text2):
    print (i)
    
#Use of Stopwords (a,and,the etc. words useless for data analysis but used a lot in paragraphs or text)

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

example_sentence = "This is an example showing off stopword filtration."

stop_words = set(stopwords.words("english"))

words = word_tokenize(example_sentence)

filtered_sentence = []

for w in words:
    if w not in stop_words:
        filtered_sentence.append(w)
        
print(filtered_sentence)

#same thing in one liner famous 39-43

filtered_sentence = [w for w in words if w not in stop_words]

print(filtered_sentence)

#Use of stemming (affixers like ing,er,ed will be removed after stemming)

from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

ps = PorterStemmer()

example_words = ["python","pythoning","pythoned","pythonli","pythonly","pythoner"]

for w in example_words:
    print(ps.stem(w))
    
new_text = "It is very important to be pythonly when you are pythoning with python. All pythoners have pythoned poorly at least once."

words_new = word_tokenize(new_text)

for w in words_new:
    print(ps.stem(w))
    

#Part of speech tagging

from nltk.corpus import state_union
from nltk.tokenize import PunktSentenceTokenizer #This is some machine learning based sentence tokenizer which can be trained

train_text = state_union.raw("2006-GWBush.txt")
real_text = state_union.raw("2005-GWBush.txt")

custom_sent_tokenizer = PunktSentenceTokenizer(train_text)
real_tokenized_text = custom_sent_tokenizer.tokenize(real_text)

def process_content():
    try:
        for i in real_tokenized_text:
            words_real = nltk.word_tokenize(i)
            words_tagged = nltk.pos_tag(words_real) 
            print(words_tagged)
    except Exception as e:
        print(str(e))

#Call the function to get all the part of speech tagging of 2005 state union address of GW Bush
process_content()

#In order to get all possible tagging and their meaning
nltk.help.upenn_tagset()

#Chunking words of differrent parts of speech together

def process_content_chunk():
    try:
        for i in real_tokenized_text:
            words_real = nltk.word_tokenize(i)
            words_tagged = nltk.pos_tag(words_real) 
            #r means regular expression, we can call it Chunk or anything else, RB for adverbs, RB. means one extra character allowed like RBS,RBR other adverbs,then ? means 0 or 1 that is any adverb of those three, then * means zero or more of those adverb and then similar to verb, proper noun and noun
            ChunkGram = r"""Chunk: {<RB.?>*<VB.?>*<NNP.?>*<NN.?>*}"""
            ChunkGram1 = r"""Chunk1: {<RB.?>*<VB.?>*<JJ>*<NNP.?>*<NN.?>*<PRP.?>}"""
            ChunkParser = nltk.RegexpParser(ChunkGram1)
            
            Chunked = ChunkParser.parse(words_tagged)
            
            print(Chunked)
            
            #Chunked.draw()
    
    except Exception as e:
        print(str(e))
            

process_content_chunk()

#Chinking which is separating words from Chunks
def process_content_chink():
    try:
        for i in real_tokenized_text:
            words_real = nltk.word_tokenize(i)
            words_tagged = nltk.pos_tag(words_real) 
            #r means regular expression, we can call it Chunk or anything else, RB for adverbs, RB. means one extra character allowed like RBS,RBR other adverbs,then ? means 0 or 1 that is any adverb of those three, then * means zero or more of those adverb and then similar to verb, proper noun and noun
            ChunkGram = r"""Chink: {<.*>+}
                                   }<VB.?|IN|DT|PRP|JJ|TO|CC>+{ """ #}{ is chinking opposite to chunking, we chunk everything except verb, adjective, interjection, to and conjuction, personal pronoun; | means or
            
            ChunkParser = nltk.RegexpParser(ChunkGram)
            
            Chunked = ChunkParser.parse(words_tagged)
            
            print(Chunked)
            
            #Chunked.draw()
    
    except Exception as e:
        print(str(e))
            

process_content_chink()

#Named Entity recognition
#NE types
#ORGANIZATION   Georgia-Pacific Corp.,WHO
#PERSON         Eddy Dante,President Obama
#DATE           June,2008-06-29
#TIME           two fifty a m,1:30 p.m.
#LOCATION       Murray River,Mount Everest
#MONEY          175 million Canadian Dollars,GBP 10.40
#PERCENT        twenty pct,18.75%
#GPE            South East Asia,New York
#FACILITY       Washington Monument,Stonehenge
###################################################

def process_content_ne():
    try:
        for i in real_tokenized_text:
            words_real = nltk.word_tokenize(i)
            words_tagged = nltk.pos_tag(words_real) 
            #ne chunk by types returns a tree structure always
            namedEnt = nltk.ne_chunk(words_tagged)
            print(namedEnt)
            #ne chunk not explicitly by any type each time
            namedEnt1 = nltk.ne_chunk(words_tagged,binary=True)
            print(namedEnt1)
    
    except Exception as e:
        print(str(e))
            

process_content_ne()

#Named Entity recognition to a continuous Python list
from nltk import ne_chunk,pos_tag,word_tokenize
from nltk.tree import Tree

def get_continuous_chunks(text):
    chunked = ne_chunk(pos_tag(word_tokenize(text)))
    #prev = None
    continuous_chunk = []
    current_chunk = []
    for i in chunked:
        if type(i) == Tree:
            current_chunk.append(" ".join([token for token, pos in i.leaves()]))
        elif current_chunk:
            named_entity = " ".join(current_chunk)
            if named_entity not in continuous_chunk:
                continuous_chunk.append(named_entity)
                current_chunk = []
        else:
            continue
    
    return continuous_chunk

my_text = "WASHINGTON -- In the wake of a string of abuses by New York police officers in the 1990s, Loretta E. Lynch, the top federal prosecutor in Brooklyn, spoke forcefully about the pain of a broken trust that African-Americans felt and said the responsibility for repairing generations of miscommunication and mistrust fell to law enforcement."

list1 = get_continuous_chunks(my_text)

print(list1)

my_text1 = state_union.raw("2006-GWBush.txt")

list2 = get_continuous_chunks(my_text1)

print(list2)

#Named Entity recognition in python list with labels
import nltk
for sent in nltk.sent_tokenize(my_text1):
   for chunk in nltk.ne_chunk(nltk.pos_tag(nltk.word_tokenize(sent))):
      if hasattr(chunk, 'label'):
         print(chunk.label(), ' '.join(c[0] for c in chunk))


#Lemmatizer, similar to stemming but always give a meaningful word, plus have different pos forms other than default noun(n) 

from nltk.stem import WordNetLemmatizer

Lemmatizer = WordNetLemmatizer()

print(Lemmatizer.lemmatize("cats"))  
print(Lemmatizer.lemmatize("mindfullness",pos="a"))  
print(Lemmatizer.lemmatize("better"))  
print(Lemmatizer.lemmatize("better",pos="a"))  
print(Lemmatizer.lemmatize("running",pos="v"))  
print(Lemmatizer.lemmatize("running"))         
print(Lemmatizer.lemmatize("pythonly",pos="n"))  









     