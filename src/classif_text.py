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
import codecs
import sys
import glob
import re
import getopt
import random
import operator
import pickle
#import cPickle as pickle
from math import *
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
# TO DO


##################################################################
Info('Reading training data and building inverted file')


def Tokenize(text):
    t_words = []
    # TODO
    # split into words, do not forget to remove punctuation and carriage return...

    return t_words


h_word2did2tf = defaultdict(lambda: defaultdict(lambda: 0))
h_train_id2real_class = {}
# TODO read the training data


##################################################################
Info('Removing stopwords form inverted file')

# TODO


##################################################################
Info('Computing IDF for each word')

h_word2IDF = {}
# TODO


##################################################################
Info('Computing norm for each message')

h_norm = defaultdict(int)
# TODO


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
