#!/opt/anaconda3/bin/python

import sys  # For interacting with the Unix Shell
import os   # For OS information (mainly path manipulation)
import time # For data and time
import argparse # For command line argument parsing
import re
import csv
import nltk
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
from nltk.corpus import stopwords
import numpy as np
import pandas as pd
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import matplotlib.pyplot as plt

# Python boiler plate call.
if __name__ == "__main__":
  # Argument parsing
  parser = argparse.ArgumentParser(
    description='Script to analyze data from a text file then output some stats',
    epilog='''
Examples:
    # Simple run
    analyzeText PropositionsConstitutionDz.txt
    # Specify Max number of keywords to be considered, default is 100
    analyzeText PropositionsConstitutionDz.txt --count 50
    # Specify Max number of keywords to be considered and a custom stopWords file
    analyzeText PropositionsConstitutionDz.txt --count 50 --stopWords customStopWords.txt
    # Dump clean topnes file
    analyzeText PropositionsConstitutionDz.txt --dumpTokens cleanTokens.txt
    # Dump CSV Report
    analyzeText PropositionsConstitutionDz.txt --report report.csv

''',
    formatter_class=argparse.RawDescriptionHelpFormatter)

  parser.add_argument('dataFile', type=argparse.FileType('r'), help='Input data file in text format')
  parser.add_argument('--count', type=int, default=100, help='Number of keywords to report')
  parser.add_argument('--stopWords', type=argparse.FileType('r'), help='space separated File for ustom stop words')
  parser.add_argument('--dumpTokens', type=argparse.FileType('w'), help='Output file with clean tokens')
  parser.add_argument('--report', type=argparse.FileType('w'), help='Output file with clean tokens')

  args = parser.parse_args()
  # Open the data file then cleanup up blank strings like \n, \t ...etc
  with open(args.dataFile.name, 'r') as file:
    data = file.read().replace('\n', ' ')
  ' '.join(data.split())
  # Somme further cleanup
  # Delete special caracters and numbers
  data = re.sub('\W+',' ', data)
  data = re.sub('\d+', '', data)
  # Remove short customStopWords
  data = re.sub(r'\b\w{1,2}\b', '', data)
  # Passing the string text into word tokenize for breaking the sentences
  tokens = word_tokenize(data)
  # Filter the tokens
  stopwords = nltk.corpus.stopwords.words('arabic')
  # Append custom stop words
  if args.stopWords is not None:
    with open(args.stopWords.name, 'r') as file:
      customStopWords = file.read().replace('\n', ' ').replace(',', ' ')
      stopwords.extend(customStopWords.split())
  cleanTokens = [x for x in tokens if x not in stopwords]
  if args.dumpTokens is not None:
    with open(args.dumpTokens.name, 'w') as file:
      for item in cleanTokens:
        file.write('{0}\n'.format(item))
  # Look at Frequency distribution
  dataFreqDist = FreqDist(cleanTokens)
  print(dataFreqDist.most_common(args.count))
  # Dump Report
  if args.report is not None:
    with open(args.report.name,'w') as file:
      csvReport=csv.writer(file)
      csvReport.writerow(['Keyword','Frequency'])
      for item in dataFreqDist.most_common(args.count):
        csvReport.writerow(item)
  # Create and generate a word cloud image:
  wordcloud = WordCloud(max_words=100).generate(' '.join(cleanTokens))
  # Display the generated image:
  plt.imshow(wordcloud, interpolation='bilinear')
  plt.axis("off")
  plt.show()
