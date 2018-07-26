#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 25 18:59:03 2018

@author: niladri
"""
import io
import nltk
import random
#from nltk.corpus import movie_reviews
from nltk.classify.scikitlearn import SklearnClassifier
#import pickle
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from nltk.classify import ClassifierI
from statistics import mode
from nltk.tokenize import word_tokenize

short_pos = io.open("positive.txt",encoding ='latin-1').read()
short_neg = io.open("negative.txt",encoding ='latin-1').read()

documents = []

for r in short_pos.split("\n"):
    documents.append((r,"pos"))

for r in short_neg.split("\n"):
    documents.append((r,"neg"))


short_pos_words = word_tokenize(short_pos)
short_neg_words = word_tokenize(short_neg)

all_words = []

for w in short_pos_words:
    all_words.append(w.lower())
    
for w in short_neg_words:
    all_words.append(w.lower())
    
all_words_withfreq = nltk.FreqDist(all_words)

print(all_words_withfreq.most_common(25))
#print(all_words_withfreq.most_common(25))

print(all_words_withfreq["stupid"])

words_features = list(all_words_withfreq.keys())[:1000]

def find_features(document):
    words = word_tokenize(document)
    features = {} #empty dictionary
    for w in words_features:
        features[w] = (w in words) #if the word also in the most common 3000 word features, then corresponding feature will be true else false, features is a binary value list of every words in the document
        
    return features

#print(find_features(movie_reviews.words("neg/cv000_29416.txt")))

featuresets = [(find_features(rev),category) for (rev,category) in documents]

#print(featuresets)
#total 2000 reviews, 1000 positive and 1000 negative which randomly shuffled, 1900 train, rest 100 test

#Positive test set
#training_set = featuresets[:1900]
#testing_set = featuresets[1900:]

#Negative test set
#training_set = featuresets[100:]
#testing_set = featuresets[:100]

random.shuffle(featuresets)
training_set = featuresets[:10000]
testing_set = featuresets[10000:]
#Using nltk based simple, scalable Naive Bayes Classifier
Classifier = nltk.NaiveBayesClassifier.train(training_set)


#classifier_file = open("naivebayes.pickle","rb")
#Classifier = pickle.load(classifier_file)
#classifier_file.close()

print("Checking Original Naive Bayes Algo Accuracy Percentage :", nltk.classify.accuracy(Classifier,testing_set)*100)
Classifier.show_most_informative_features(20)

#Saving the classifier using pickle
#In pickle write should be in bytes after python 3 and so on
#save_classifier = open("naivebayes.pickle","wb")
#which you need to save where
#pickle.dump(Classifier,save_classifier)
#closing
#save_classifier.close()

MNB_classifier = SklearnClassifier(MultinomialNB())
MNB_classifier.train(training_set)
print("Checking MNB Classifier Accuracy Percentage :", nltk.classify.accuracy(MNB_classifier,testing_set)*100)


#GaussianNB_classifier = SklearnClassifier(GaussianNB())
#GaussianNB_classifier.train(training_set)
#print("Checking GaussianNB Classifier Accuracy Percentage :", nltk.classify.accuracy(GaussianNB_classifier,testing_set)*100)


BernoulliNB_classifier = SklearnClassifier(BernoulliNB())
BernoulliNB_classifier.train(training_set)
print("Checking BernoulliNB Classifier Accuracy Percentage :", nltk.classify.accuracy(BernoulliNB_classifier,testing_set)*100)

LogisticRegression_classifier = SklearnClassifier(LogisticRegression())
LogisticRegression_classifier.train(training_set)
print("Checking LogisticRegression Classifier Accuracy Percentage :", nltk.classify.accuracy(LogisticRegression_classifier,testing_set)*100)

SGD_classifier = SklearnClassifier(SGDClassifier())
SGD_classifier.train(training_set)
print("Checking SGD Classifier Accuracy Percentage :", nltk.classify.accuracy(SGD_classifier,testing_set)*100)

SVC_classifier = SklearnClassifier(SVC())
SVC_classifier.train(training_set)
print("Checking SVC Classifier Accuracy Percentage :", nltk.classify.accuracy(SVC_classifier,testing_set)*100)

LinearSVC_classifier = SklearnClassifier(LinearSVC())
LinearSVC_classifier.train(training_set)
print("Checking LinearSVC Classifier Accuracy Percentage :", nltk.classify.accuracy(LinearSVC_classifier,testing_set)*100)

NuSVC_classifier = SklearnClassifier(NuSVC())
NuSVC_classifier.train(training_set)
print("Checking NuSVC Classifier Accuracy Percentage :", nltk.classify.accuracy(NuSVC_classifier,testing_set)*100)

KNeighbors_classifier = SklearnClassifier(KNeighborsClassifier())
KNeighbors_classifier.train(training_set)
print("Checking KNeighbors Classifier Accuracy Percentage :", nltk.classify.accuracy(KNeighbors_classifier,testing_set)*100)

RandomForest_classifier = SklearnClassifier(RandomForestClassifier())
RandomForest_classifier.train(training_set)
print("Checking RandomForest Classifier Accuracy Percentage :", nltk.classify.accuracy(RandomForest_classifier,testing_set)*100)


#Combining Algos with vote

class voteclassifier(ClassifierI):
    def __init__(self,*classifiers):
        self._classifiers = classifiers
        
    def classify(self,features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)
        return mode(votes)
    
    def confidence(self,features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)
            
        choice_votes = votes.count(mode(votes))
        conf = choice_votes / len(votes)
        
        return conf


voted_classifier = voteclassifier(Classifier,
                                  RandomForest_classifier,
                                  BernoulliNB_classifier,
                                  LogisticRegression_classifier,
                                  SGD_classifier,
                                  LinearSVC_classifier,
                                  NuSVC_classifier,
                                  SVC_classifier,
                                  MNB_classifier)

print("Checking voted_classifier Classifier Accuracy Percentage :", nltk.classify.accuracy(voted_classifier,testing_set)*100)

print("Classification:",voted_classifier.classify(testing_set[0][0]),"Confidence Percentage:",voted_classifier.confidence(testing_set[0][0])*100)
print("Classification:",voted_classifier.classify(testing_set[1][0]),"Confidence Percentage:",voted_classifier.confidence(testing_set[1][0])*100)
print("Classification:",voted_classifier.classify(testing_set[2][0]),"Confidence Percentage:",voted_classifier.confidence(testing_set[2][0])*100)
print("Classification:",voted_classifier.classify(testing_set[3][0]),"Confidence Percentage:",voted_classifier.confidence(testing_set[3][0])*100)
print("Classification:",voted_classifier.classify(testing_set[4][0]),"Confidence Percentage:",voted_classifier.confidence(testing_set[4][0])*100)
print("Classification:",voted_classifier.classify(testing_set[5][0]),"Confidence Percentage:",voted_classifier.confidence(testing_set[5][0])*100)
print("Classification:",voted_classifier.classify(testing_set[6][0]),"Confidence Percentage:",voted_classifier.confidence(testing_set[6][0])*100)
print("Classification:",voted_classifier.classify(testing_set[7][0]),"Confidence Percentage:",voted_classifier.confidence(testing_set[7][0])*100)
print("Classification:",voted_classifier.classify(testing_set[8][0]),"Confidence Percentage:",voted_classifier.confidence(testing_set[8][0])*100)
print("Classification:",voted_classifier.classify(testing_set[9][0]),"Confidence Percentage:",voted_classifier.confidence(testing_set[9][0])*100)










