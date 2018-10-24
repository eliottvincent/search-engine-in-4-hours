#!/usr/bin/python3

#
# comment:

# Purpose:
#
# Comment:

# Code:

##########################################################################
#                            INITIALIZATION                              #
##########################################################################

import argparse
from collections import defaultdict
import os
import subprocess
import string
import codecs
import sys
import glob
import re
import getopt
import random
import operator
import pickle
#import cPickle as pickle
import math
#import psyco
# psyco.full()
fnull = open(os.devnull, 'w')
# reload(sys)
# sys.setdefaultencoding('utf-8')


prg = sys.argv[0]


def P(output=''): input(output + "\nDebug point; Press ENTER to continue")


def Info(output='', ending='\n'):  # print(output, file=sys.stderr)
    sys.stderr.write(str(output) + ending)
    sys.stderr.flush()

#######################################
# special imports

#######################################
# files

#######################################
# variables
separator = '%#%#%#%#\n'
message_class = 'class_20ng:'
message_number = 0


#########################################
# USAGE


parser = argparse.ArgumentParser()

parser.add_argument("-tr", "--train", dest="file_train",
                    help="train file", metavar="STR")

parser.add_argument("-te", "--test", dest="file_test",
                    help="test file", metavar="STR")

parser.add_argument("-s", "--stop", dest="file_stop",
                    default='./stop_words.en.txt',
                    help="FILE contains stop words", metavar="FILE")


parser.add_argument("-o", "--out", dest="prefix",
                    help="PREFIX for output files", metavar="STR")

parser.add_argument("-v", "--verbose",
                    action="store_false", dest="verbose", default=True,
                    help="print status messages to stdout")


args = parser.parse_args()

################################################################################
################################################################################
##                                                                            ##
##                                 FUNCTIONS                                  ##
##                                                                            ##
################################################################################
################################################################################


################################################################################
################################################################################
##                                                                            ##
##                                   MAIN                                     ##
##                                                                            ##
################################################################################
################################################################################

##################################################################
Info('Reading stop word file')

t_stopwords = []
with codecs.open(args.file_stop, 'r', 'utf-8') as file:
    t_stopwords = file.read().splitlines()
print('Stop word file has been loaded ({} words)'.format(len(t_stopwords)))


##################################################################
Info('Reading training data and building inverted file')


def Tokenize(text):
    t_words = []
    # TODO
    # split into words, do not forget to remove punctuation and carriage return...
    translator = text.maketrans('', '', string.punctuation)
    t_words = text.translate(translator).split()
    return t_words


h_word2did2tf = defaultdict(lambda: defaultdict(lambda: 0))
h_train_id2real_class = {}
# TODO read the training data
#str_test = "hello  world\n my name, is thomas."
#print(Tokenize(str_test))
with codecs.open(args.file_train,'r', 'utf-8') as fp:
    t_messages = {}
    message = []
    message_label = ''

    for line in fp:
        current_message_class = ''
        m = re.search('^class_20ng: (.+)',line)
        

        if m is not None:
            # Classe trouvée - Début de message
            message_label = 'msg_{}'.format(message_number)

            t_class_str = line.split('class_20ng: ')
            h_train_id2real_class[message_label] = t_class_str[1]

        elif line == separator:
            # Séparateur trouvé - Fin de message
            t_messages[message_label] = message
            message_number += 1
            message = []
        else:
            # Corps trouvé
            message.append(line)

    print(len(t_messages))

    #fp_str = fp.read()
    #t_messages = fp_str.split(separator)
    
    # Nombre de messages
    # print(len(t_messages))
    # message_number = 0
    for message_label in t_messages:
        t_message_lines = t_messages[message_label]
        message = ''
        message_words = []
        for line in t_message_lines:
            message_words = message_words + (Tokenize(line))
            #print(message_words)

        for message_word in message_words:
            # Removing stopwords form inverted file
            h_word2did2tf[message_word][message_label] += 1


    #print(h_word2did2tf['number'])
    print(h_train_id2real_class['msg_8305'])


##################################################################
Info('Removing stopwords form inverted file')
for stopword in t_stopwords: 
    if stopword in h_word2did2tf:
        del h_word2did2tf[stopword]


##################################################################
Info('Computing IDF for each word')

h_word2IDF = {}
# TODO
for word in h_word2did2tf:
    nb_messages_for_word = 0
    for msg in h_word2did2tf[word]:
        nb_messages_for_word += 1
    #print ('Number of messages for ' + word + ' : {}'.format(nb_messages_for_word))
    h_word2IDF[word] = math.log10(message_number/nb_messages_for_word)

print(h_word2IDF)

##################################################################
Info('Computing norm for each message')

h_norm = defaultdict(float)
# TODO

for word in h_word2did2tf: 
    # Computes TF . IDF for each word
    for message_label in h_word2did2tf[word]:
        h_norm[message_label] += math.pow(h_word2did2tf[word][message_label] * h_word2IDF[word],2)

for message_label in h_norm:
    h_norm[message_label] = math.sqrt(h_norm[message_label])  


##################################################################
##################################################################
Info('Reading test data')


def Find_closest_neighbours(text, k=1):
    # find the k nearest neighbour (cosine with TF.IDF) of a test message

    h_train_id2score = defaultdict(lambda: 0)
    # TO DO : compute the cosine for the training messages

    if len(h_train_id2score) < k:
        print('Warning: not enough neighbours')
        k = len(h_train_id2score)

    return sorted(list(h_train_id2score.keys()), key=lambda did: -h_train_id2score[did])[0:k]


def Vote(t_knn):
    # output the majority class from the k-nearest neighbour list

    h_class2vote = defaultdict(int)

    # TODO

    return predicted


h_test_id2predicted_class = {}
h_test_id2real_class = {}

# TODO
# read the test data and for each message,
# find the nearest_neighbors with Find_closest_neighbours()
# and make them vote for their category with Vote()


##################################################################
Info('Evaluation')

# DO NOT CHANGE ANYTHING

TP, AP, TN, AN = 0, 0, 0, 0
for test_id in h_test_id2real_class:
    AP += 1
    if test_id not in h_test_id2predicted_class:
        print('noting predicted for', test_id)
        continue
    if h_test_id2predicted_class[test_id] == h_test_id2real_class[test_id]:
        TP += 1

acc = 0
if AP > 0:
    acc = TP / AP
print('accurracy=', acc)
