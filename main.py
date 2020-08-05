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

# # Unpickle corpus.txt, tweets.txt, tweet vectors, randomized tweet vectors,
#                   train set, test set,
# infile = open('corpus.txt', 'rb')
# corpus = pickle.load(infile)
# infile.close()
#
# infile = open('tweets.txt', 'rb')
# all_tweets = pickle.load(infile)
# infile.close()
#
# infile = open('train_set.txt', 'rb')
# train_set = pickle.load(infile)
# infile.close()
#
# infile = open('test_set.txt', 'rb')
# test_set = pickle.load(infile)
# infile.close()

infile = open('num_tweet_types.txt', 'rb')
num_trump_tweets = pickle.load(infile)
infile.close()

infile = open('small_train_set.txt', 'rb')
train_set = pickle.load(infile)
infile.close()

infile = open('small_test_set.txt', 'rb')
test_set = pickle.load(infile)
infile.close()

infile = open('tweet_vectors.txt', 'rb')
tweet_vectors = pickle.load(infile)
infile.close()
#
# infile = open('randomized_tweet_vectors.txt', 'rb')
# randomized_tweet_vectors = pickle.load(infile)
# infile.close()
#

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
    Clean text data and add to corpus.txt

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


def split_train_test(tweet_vectors) -> Tuple:
    """
    Create train and test sets

    :param tweet_vectors: tweets in vector form

    :return: train_set, test_set, randomized_tweet_vectors: tuple of train set, test set, and
                                                            randomized tweet vectors
    """
    randomized_tweet_vectors = np.zeros((tweet_vectors.shape[0], tweet_vectors.shape[1]), dtype=int)
    for x in range(tweet_vectors.shape[0]):
        for y in range(tweet_vectors.shape[1]):
            randomized_tweet_vectors[x][y] = tweet_vectors[x][y]
    np.random.shuffle(randomized_tweet_vectors)

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

    return train_set, test_set, randomized_tweet_vectors


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

    knn_indices_and_distances.sort(key=lambda x: x[1])

    return knn_indices_and_distances[:k]


def majority_vote(tweet_vector, train_set, k) -> str:
    """
    Count how many of the k-NN tweets.txt were written by Trump or not-Trump,
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


# # Initial setup of corpus.txt and vectorizers, all of which then get pickled
#
# corpus.txt = []
# all_tweets = []
#
# # Get and clean tweet data for Trump (also adds words from tweets.txt to corpus.txt)
# trump_tweets = read_file('tweets.txt.json', key_name='text')
# for item in range(len(trump_tweets)):
#     tweet = trump_tweets[item]
#     tweet = clean_text(corpus.txt, tweet)
#     trump_tweets[item] = tweet
#     all_tweets.append(tweet)
#
# print('Finished reading Trump tweets.txt')
#
# # Get and clean general tweet data (also adds words from tweets.txt to corpus.txt)
# raw_general_tweets = read_file('all_annotated.tsv')
# general_tweets = []
# for x in raw_general_tweets[1:]:
#     if x[4] == '1':  # Indicates Tweet is written in English
#         general_tweets.append(clean_text(corpus.txt, x[3]))
#         all_tweets.append(clean_text(corpus.txt, x[3]))
#
# print('Finished reading general tweets.txt')
#
# # Vectorize tweets.txt
# tweet_vectors.txt = np.zeros((len(all_tweets), len(corpus.txt) + 2), dtype = int)
# for word in range(len(corpus.txt)):
#     for tweet in range(len(all_tweets)):
#         if corpus.txt[word] in all_tweets[tweet]:
#             tweet_vectors.txt[tweet][word] = 1
#
#
# # Label tweets.txt as Trump or general, and keep track of index
# for tweet in range(len(all_tweets)):
#     if tweet <= len(trump_tweets):
#         tweet_vectors.txt[tweet][-1] = 1  # Label second-to-last value with 1 for Trump
#     else:
#         tweet_vectors.txt[tweet][-1] = 0
#
#     tweet_vectors.txt[tweet][-2] = tweet    # Keep track of index in all_tweets for interpretation
#                                         # after randomization



# # tweet_vectors.txt = individual_tweet_vectorizer(corpus.txt, trump_tweets[0], author = 'trump')
# for tweet in range(1, len(trump_tweets) - 1):
#     tweet_vector = individual_tweet_vectorizer(corpus.txt, trump_tweets[tweet], index = tweet, author = 'trump')
#     tweet_vectors.txt = np.append(tweet_vectors.txt, tweet_vector, axis = 0)
#     if tweet % 1000 == 0:
#         print('Vectorizing Trump tweets.txt')
#         print(tweet_vectors.txt.shape)
#
# # Add binary vectors for general tweets.txt compared to corpus.txt
# for tweet in range(len(general_tweets)):
#     tweet_vector =individual_tweet_vectorizer(corpus.txt, general_tweets[tweet], index = len(trump_tweets) + 1)
#     tweet_vectors.txt = np.append(tweet_vectors.txt, tweet_vector, axis = 0)
#     print('Vectorizing general tweets.txt')
#     if tweet % 1000 == 0:
#         print(tweet_vectors.txt.shape)

# Make train and test sets
# train_set.txt, test_set.txt, randomized_tweet_vectors.txt = split_train_test(tweet_vectors.txt)
# print('train set shape')
# print(train_set.txt.shape)
# print('test set shape')
# print(test_set.txt.shape)
#
# print('randomized vectors shape')
# print(randomized_tweet_vectors.txt.shape)

# Pickle corpus.txt, tweets.txt, tweet vectors, randomized tweet vectors, train set, test set,
# number of Trump tweets.txt, and number of general tweets.txt
# filename = 'corpus.txt'
# outfile = open(filename, 'wb')
# pickle.dump(corpus.txt, outfile)
# outfile.close()
#
# filename = 'tweets.txt'
# outfile = open(filename, 'wb')
# pickle.dump(all_tweets, outfile)
# outfile.close()
#
# filename = 'tweet_vectors.txt'
# outfile = open(filename, 'wb')
# pickle.dump(tweet_vectors.txt, outfile)
# outfile.close()
#
# filename = 'randomized_tweet_vectors.txt'
# outfile = open(filename, 'wb')
# pickle.dump(randomized_tweet_vectors.txt, outfile)
# outfile.close()
#
# filename = 'train_set.txt'
# outfile = open(filename, 'wb')
# pickle.dump(train_set.txt, outfile)
# outfile.close()
#
# filename = 'test_set.txt'
# outfile = open(filename, 'wb')
# pickle.dump(test _set.txt, outfile)
# outfile.close()
#
# # Get small subsets of train and test sets to make prototyping faster
# small_train_set.txt = np.zeros((100, train_set.txt.shape[1]), dtype = int)
# small_test_set.txt = np.zeros((100, test_set.txt.shape[1]), dtype = int)
#
# for x in range(100):
#     for y in range(train_set.txt.shape[1]):
#         small_train_set.txt[x][y] = train_set.txt[x][y]
#         small_test_set.txt[x][y] = test_set.txt[x][y]
#
# # Pickle prototyping sets
# filename = 'small_train_set.txt'
# outfile = open(filename, 'wb')
# pickle.dump(small_train_set.txt, outfile)
# outfile.close()
#
# filename = 'small_test_set.txt'
# outfile = open(filename, 'wb')
# pickle.dump(small_test_set.txt, outfile)
# outfile.close()

# num_trump_tweets = len(trump_tweets)
# filename = 'num_tweet_types'
# outfile = open(filename, 'wb')
# pickle.dump(num_trump_tweets, outfile)
# outfile.close()

#  # Test using a known Trump tweet
# tweet = "$55.15M will be on its way to @KYTC to widen @mtnparkway from two lanes to four lanes between the KY 191 overpass and the KY 205 interchange. Must keep the people of Kentucky moving efficiently and safely!"
# tweet = clean_text(corpus.txt, tweet)
# tweet_vector = individual_tweet_vectorizer(corpus.txt, tweet, 0, 'trump')
# print(predict(tweet_vector, train_set.txt))
#
# print(all_tweets[-200])
# # Test using a known general string
# # tweet = all_tweets[-200]
# tweet_vector = individual_tweet_vectorizer(corpus.txt, all_tweets[-200], -200)
# print(predict(tweet_vector, train_set.txt))
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


# Old ideas, will remove later if not needed


# Associate Trump words with their frequencies
# counts = Counter(trump_words)


# Test using .csv email data
# data = read_file('email_metadata_merged_with_newsletters.csv')
#
#
# # Get only email text data from knn
# text_blocks = []
# for i in range(1, len(data)):
#     text_blocks.append(data[i][4])
#
# print(text_blocks[:10])
# words = get_words((text_blocks))
# print(words[:100])

# Incorrect apostrophe handling
# print(text_blocks[0])
# words = get_words(text_blocks[0])
# print(words[:10])

# But other punctuation is handled correctly
# print(text_blocks[4])
# words_test1 = get_words(text_blocks[4])
# print(words_test1[:1000])

# And apostrophes *are* handled correctly for data entered manually
# words_test2 = get_words(["laura's testing this code", 'testing testy test'])
# print(words_test2[:10])
