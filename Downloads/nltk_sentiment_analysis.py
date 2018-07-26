#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 26 08:22:03 2018

@author: dino
"""

import nltk_sentiment_mod as s

sample_text1 = "The movie was wonderful. There was full of pythons. Plot was very good, acting was awesome too....So yeah baby!"

sample_text2 = "The movie was utter junk. There was 0 python. Acting was horrible. No excitement, no energy. Waste of time. 0/10"


print(s.sentiment(sample_text1))

print(s.sentiment(sample_text2))

