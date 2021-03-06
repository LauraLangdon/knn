import gzip
import csv
import re
from typing import List, Tuple
import string
import json
from collections import Counter
import numpy as np
import pickle
import math

# Unpickle corpus, tweets, tweet vectors, randomized tweet vectors,
#                   train set, test set,
infile = open('corpus.txt', 'rb')
corpus = pickle.load(infile)
infile.close()

infile = open('tweets.txt', 'rb')
all_tweets = pickle.load(infile)
infile.close()

infile = open('num_trump_tweets.txt', 'rb')
num_trump_tweets = pickle.load(infile)
infile.close()

# infile = open('train_set.txt', 'rb')
# train_set = pickle.load(infile)
# infile.close()
#
# infile = open('test_set.txt', 'rb')
# test_set = pickle.load(infile)
# infile.close()

infile = open('small_train_set.txt', 'rb')
train_set = pickle.load(infile)
infile.close()

infile = open('small_test_set.txt', 'rb')
test_set = pickle.load(infile)
infile.close()

infile = open('tweet_vectors.txt', 'rb')
tweet_vectors = pickle.load(infile)
infile.close()

infile = open('randomized_tweet_vectors.txt', 'rb')
randomized_tweet_vectors = pickle.load(infile)
infile.close()


# Get list of stop words
get_stop_words = open('stop_words.txt', 'r')
stop_words_list = get_stop_words.read()


def read_file(file_name: str, key_name='') -> list:
    """
    Open and read csv.gz, .tsv, .csv, or JSON file; return as list

    :param file_name: Name of file
    :param key_name: Name of JSON key (optional)

    :return: data_list: Data in list form
    """

    # Determine whether file is of type csv.gz, tsv, csv, or JSON
    if file_name[-6:] == 'csv.gz':
        data_file = gzip.open(file_name, mode='rt')
        data_list = csv_list_maker(data_file)

    elif file_name[-3:] == 'tsv':
        data_file = open(file_name, newline="")
        data_list = csv_list_maker(data_file, delimiter="\t")

    elif file_name[-3] == 'csv':
        data_file = open(file_name)
        data_list = csv_list_maker(data_file)

    elif file_name[-4:] == 'json':
        with open(file_name, 'r') as read_file:
            data_file = json.load(read_file)
            data_list = []
            for item in range(len(data_file)):
                data_list.append(data_file[item][key_name])
    else:
        print('Unusable file type. Please submit file of type csv.gz, .csv, .tsv, or JSON')
        return []

    return data_list


def csv_list_maker(data_file, delimiter=',') -> list:
    """
    Turn data in csv form into list form

    :param data_file: file containing the data
    :param delimiter: character delimiting the data

    :return: data_list: data in list form
    """

    data_reader = csv.reader(data_file, delimiter=delimiter)
    data_list = list(data_reader)

    return data_list


def clean_text(corpus, input_string: str) -> list:
    """
    Clean text data and add to corpus

    :param corpus: list of all words in the data
    :param input_string: string of words to be added to the corpus

    :return: output_string_as_list: cleaned list of words from input string
    """
    input_string = re.split(r'\W+', input_string)
    output_string_as_list = []
    for word in input_string:
        if word in stop_words_list:
            continue
        for char in word:
            if char in string.punctuation or char.isnumeric() or char == ' ':
                word = word.replace(char, '')
        if word == '':
            continue
        if word.lower() not in corpus:
            corpus.append(word.lower())
        output_string_as_list.append(word.lower())

    return output_string_as_list


def randomize_vectors(tweet_vectors):
    """

    :param tweet_vectors:

    :return: randomized_tweet_vectors: a Numpy array of tweet vectors that have
                 been randomly shuffled
    """
    #Initialize randomized tweet vectors
    randomized_tweet_vectors = np.zeros((tweet_vectors.shape[0], tweet_vectors.shape[1]), dtype=int)
    for x in range(tweet_vectors.shape[0]):
        for y in range(tweet_vectors.shape[1]):
            randomized_tweet_vectors[x][y] = tweet_vectors[x][y]
    np.random.shuffle(randomized_tweet_vectors)

    return randomized_tweet_vectors

def split_train_test(tweet_vectors, randomized_tweet_vectors) -> tuple:
    """
    Split into train and test sets

    :param tweet_vectors: tweets in vector form

    :return: train_set, test_set tuple of train set and test set
    """
    x_train_dim = math.floor(0.8 * tweet_vectors.shape[0])  # Use 80% of data for train set
    x_test_dim = math.ceil(0.2 * tweet_vectors.shape[0])  # Use 20% of data for test set
    y_dim = tweet_vectors.shape[1]

    train_set = np.zeros((x_train_dim, y_dim), dtype=int)
    test_set = np.zeros((x_test_dim, y_dim), dtype=int)

    for x in range(x_train_dim):
        for y in range(y_dim):
            train_set[x][y] = randomized_tweet_vectors[x][y]
    for x in range(x_test_dim):
        for y in range(y_dim):
            test_set[x][y] = randomized_tweet_vectors[x + x_train_dim][y]

    return train_set, test_set


def individual_tweet_vectorizer(corpus, tweet, index=0, author=''):
    """
    Formats a single tweet as a vector

    :param corpus: list of all words in tweets
    :param tweet: tweet to be vectorized
    :param index: index of tweet in main list of tweets
    :param author: Trump or general

    :return: Single tweet in vector form
    """
    individual_tweet_vector = np.zeros((1, len(corpus) + 2), dtype=int)
    for word in range(len(corpus)):
        if corpus[word] in tweet:
            individual_tweet_vector[0][word] = 1
    if author != '':  # If author is specified, set the last value of the tweet vector to 1
        individual_tweet_vector[0][-1] = 1
    individual_tweet_vector[0][-2] = index  # Keep track of index of tweet for interpretation
    return individual_tweet_vector


def get_distance(tweet1_vector, tweet2_vector) -> int:
    """
    Implement Minkowski distance metric

    :param tweet1_vector: vector of first tweet
    :param tweet2_vector: vector of second tweet

    :return: Minkowski distance between tweets
    """
    distance = 0
    for x in range(tweet1_vector.shape[0] - 2):
        distance = distance + abs(tweet1_vector[x] - tweet2_vector[x])

    return distance


def knn(tweet_vector, train_set, k) -> list:
    """
    Find k nearest neighbors of a given tweet

    :param tweet_vector: vector of tweet whose neighbors we seek
    :param train_set: training set
    :param k: desired number of nearest neighbors

    :return: list of indices in main tweet list of k nearest neighbors, and distances of those
            neighbors to given tweet
    """
    knn_indices_and_distances = []
    distance = 0

    for x in range(len(train_set)):
        distance = get_distance(tweet_vector[0], train_set[x])
        knn_indices_and_distances.append([train_set[x][-2], distance])

    #  Sort by distance (smallest to largest)
    knn_indices_and_distances.sort(key=lambda x: x[1])

    return knn_indices_and_distances[:k]


def majority_vote(tweet_vector, train_set, k) -> str:
    """
    Count how many of the k-NN tweets were written by Trump or not-Trump,
    and return whichever is larger

    :param tweet_vector: vector of given tweet
    :param train_set: training set
    :param k: desired number of nearest neighbors

    :return: Whether tweet was authored by Trump, not Trump, or draw
    """
    if k % 2 == 0:
        k = k - 1
    knn_indices_and_distances = knn(tweet_vector, train_set, k)

    trump_votes = 0
    general_votes = 0
    for x in range(len(knn_indices_and_distances)):
        if train_set[x][-1] == 1:
            trump_votes += 1
            # print('trump vote')
            # print(trump_votes)
        else:
            general_votes += 1
            # print('general vote')
            # print(general_votes)

    if trump_votes > general_votes:
        return 'Trump'
    elif general_votes > trump_votes:
        return 'Not Trump'
    else:
        return 'Draw'


def predict(tweet_vector, train_set, k = 9) -> str:
    """
    Predict whether a given tweet was written by Trump

    :param tweet_vector: vector of given tweet
    :param train_set: training set
    :param k: desired number of nearest neighbors

    :return: prediction of whether tweet was authored by Trump
    """
    if majority_vote(tweet_vector, train_set, k) == 'Trump':
        return 'Trump'
    elif majority_vote(tweet_vector, train_set, k) == 'Not Trump':
        return 'Not Trump'
    else:
        if k > 2:
            predict(tweet_vector, train_set, k = k - 2)
        else:
            return 'Draw: no prediction'


# # Initial setup of corpus and vectorizers, all of which then get pickled
#
# corpus = []
# all_tweets = []
#
# Get and clean tweet data for Trump (also adds words from tweets to corpus)
# trump_tweets = read_file('tweets.json', key_name='text')
# num_trump_tweets = len(trump_tweets)    # Store this value to use later for labeling
#
# for item in range(len(trump_tweets)):
#     tweet = trump_tweets[item]
#     tweet = clean_text(corpus, tweet)
#     trump_tweets[item] = tweet
#     all_tweets.append(tweet)
#
# print('Finished reading Trump tweets')
#
# # Get and clean general tweet data (also adds words from tweets to corpus)
# raw_general_tweets = read_file('all_annotated.tsv')
# general_tweets = []
# for x in raw_general_tweets[1:]:
#     if x[4] == '1':  # Indicates Tweet is written in English
#         general_tweets.append(clean_text(corpus, x[3]))
#         all_tweets.append(clean_text(corpus, x[3]))
#
# print('Finished reading general tweets')
#
# # Initialize tweet vectors
# tweet_vectors = np.zeros((len(all_tweets), len(corpus) + 2), dtype = int)
# for word in range(len(corpus)):
#     for tweet in range(len(all_tweets)):
#         if corpus[word] in all_tweets[tweet]:
#             tweet_vectors[tweet][word] = 1
#
# print('Finished initializing tweet vectors ')
#
# # Label tweets as Trump or general, and keep track of index
# for i in range(len(all_tweets)):
#     if i <= num_trump_tweets:
#         tweet_vectors[i][-1] = 1  # Label last value with 1 for Trump
#     else:
#         tweet_vectors[i][-1] = 0
#
#     tweet_vectors[i][-2] = i   # Store index as second-to-last variable for interpretation
#                                                              # after randomization
#
# print('Finished labelling tweets')
# print('tweet_vectors.shape: ')
# print(tweet_vectors.shape)
#
# # Make train and test sets
# randomized_tweet_vectors = randomize_vectors(tweet_vectors)
# print('randomized vectors shape')
# print(randomized_tweet_vectors.shape)
#
# train_set, test_set = split_train_test(tweet_vectors, randomized_tweet_vectors)
# print('train set shape')
# print(train_set.shape)
# print('test set shape')
# print(test_set.shape)
#
# # Pickle corpus, tweets, tweet vectors, randomized tweet vectors, train set, and test set
# filename = 'corpus.txt'
# outfile = open(filename, 'wb')
# pickle.dump(corpus, outfile)
# outfile.close()
#
# filename = 'tweets.txt'
# outfile = open(filename, 'wb')
# pickle.dump(all_tweets, outfile)
# outfile.close()
#
# filename = 'num_trump_tweets.txt'
# outfile = open(filename, 'wb')
# pickle.dump(num_trump_tweets, outfile,)
# outfile.close()
#
# filename = 'tweet_vectors.txt'
# outfile = open(filename, 'wb')
# pickle.dump(tweet_vectors, outfile, protocol=4)
# outfile.close()
#
# filename = 'randomized_tweet_vectors.txt'
# outfile = open(filename, 'wb')
# pickle.dump(randomized_tweet_vectors, outfile, protocol=4)
# outfile.close()
#
# filename = 'train_set.txt'
# outfile = open(filename, 'wb')
# pickle.dump(train_set, outfile, protocol=4)
# outfile.close()
#
# filename = 'test_set.txt'
# outfile = open(filename, 'wb')
# pickle.dump(test_set, outfile, protocol=4)
# outfile.close()
#
# # Get small subsets of train and test sets to make prototyping faster
# small_train_set = np.zeros((100, train_set.shape[1]), dtype = int)
# small_test_set = np.zeros((100, test_set.shape[1]), dtype = int)
#
# for x in range(100):
#     for y in range(train_set.shape[1]):
#         small_train_set[x][y] = train_set[x][y]
#         small_test_set[x][y] = test_set[x][y]
#
# # Pickle prototyping sets
# filename = 'small_train_set.txt'
# outfile = open(filename, 'wb')
# pickle.dump(small_train_set, outfile)
# outfile.close()
#
# filename = 'small_test_set.txt'
# outfile = open(filename, 'wb')
# pickle.dump(small_test_set, outfile)
# outfile.close()
#
#  # Test using a known Trump tweet
# tweet = "$55.15M will be on its way to @KYTC to widen @mtnparkway from two lanes
# to four lanes between the KY 191 overpass and the KY 205 interchange. Must keep the
# people of Kentucky moving efficiently and safely!"
# tweet = clean_text(corpus, tweet)
# tweet_vector = individual_tweet_vectorizer(corpus, tweet, 0, 'trump')
# print(predict(tweet_vector, train_set))
#
# print(all_tweets[-200])
# # Test using a known general string
# # tweet = all_tweets[-200]
# tweet_vector = individual_tweet_vectorizer(corpus, all_tweets[-200], -200)
# print(predict(tweet_vector, train_set))
# print(all_tweets[11303])
# print(all_tweets[6289])

# Get a random tweet from the test set, and predict Trump vs. general
index = np.random.randint(0, len(test_set), dtype=int)
tweet_vector = np.zeros((1, test_set.shape[1]), dtype=int)
for x in range(test_set.shape[1]):
    tweet_vector[0][x] = test_set[index][x]

print(predict(tweet_vector, train_set))
index = tweet_vector[0][-2]
print(all_tweets[index])
