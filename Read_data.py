"""
This script processes the input data for the deep learning network.
input files: File containing sequences with identifiers & File containing
identifiers for the positive samples associated with go terms.

to use: Python3 Read_data.py sequence_file_directory annotation_directory

seq files should have extention .seq
pos files should have extension .pos
"""

#Import statements:
import glob
import torch
from torch import nn
from sklearn.model_selection import train_test_split
import numpy as np
from numpy import argmax
from sys import argv
from torch.utils.data import DataLoader, TensorDataset

#Global parameters
max_sequence_length = 200

#Function that makes training and test data:
def split_data(seqfile):
    entries = []

    with open(seqfile[0], "r") as reader:
        for line in reader:
            entrie = line.split("\t")[0]
            entries.append(entrie.strip("\n"))

    X_train, X_test = train_test_split(entries, test_size=0.2, train_size=0.8, random_state=1)

    return X_train, X_test

#Function that extracts the amino acid sequences:
def extract_sequence(X_train, X_test, seqfile):
    seq_train = []
    seq_test = []

    with open(seqfile[0], "r") as reader:
        for line in reader:
            entrie = line.split("\t")[0].strip("\n")
            sequence = line.split("\t")[1].strip("\n")
            if entrie in X_train:
                seq_train.append(sequence)
            else:
                seq_test.append(sequence)

    return seq_train, seq_test


#Funtion to make output labels for each go term:
def make_label(posfiles, X):

    pos_list = []
    final_labels = []
    classes = len(posfiles)

    for i in posfiles:
        entries = []
        with open(i, "r") as reader:
            for entrie in reader:
                entries.append(entrie.strip("\n"))
        pos_list.append(entries)

    for i in pos_list:
        labels = []
        for num in range(0, len(X)):
            if X[num] in i:
                labels.append(int(1))
            else:
                labels.append(int(0))
        final_labels.append(labels)

    label_array = np.column_stack(final_labels)
    return label_array


def onehot_encode(sequence_list):

    arrays = []

    #Create dictionary of amino acids.
    alphabet = "ACDEFGHIKLMNPQRSTVWY"
    char_to_int = dict((c, i) for i, c in enumerate(alphabet))
    int_to_char = dict((i, c) for i, c in enumerate(alphabet))

    #Loop over sequences
    for seq in sequence_list:
        onehot_sequence = []
        integer_encoded = [char_to_int[char] for char in seq]
        for i in integer_encoded:
            list = [0] * 20
            list[i] = (1)
            onehot_sequence.append(list)

        #Add sequence padding
        if len(onehot_sequence) < max_sequence_length:
            list = [0] * 20
            for i in range(len(onehot_sequence), max_sequence_length):
                onehot_sequence.append(list)

        array = np.column_stack(onehot_sequence)
        arrays.append(array)

    arrays = np.stack(arrays)

    return arrays


def main(seqfile, posfiles):

    #Split data in testing and training:
    X_train, X_test = split_data(seqfile)

    #Make True y labels for training and testing set:
    Y_train, Y_test = make_label(posfiles, X_train), make_label(posfiles, X_test)

    #Extract sequences for training and test data:
    seq_train, seq_test = extract_sequence(X_train, X_test, seqfile)

    #Sequences are made into one hot encoded tensor:
    X_train, X_test = onehot_encode(seq_train), onehot_encode(seq_test)

    #Make tensor dataset
    X_train = torch.from_numpy(X_train)
    X_test = torch.from_numpy(X_test)

    Y_train = torch.from_numpy(Y_train)
    Y_test = torch.from_numpy(Y_test)

    train_set = TensorDataset( X_train, Y_train )
    test_set = TensorDataset( X_test, Y_test )

    return train_set, test_set

if __name__ == "__main__":
    #Reading files
    seq_file_dir = argv[1]
    pos_file_dir = argv[2]
    seqfile = glob.glob(str(seq_file_dir) + "*.seq")
    posfiles = glob.glob(str(pos_file_dir) + "*.pos")
    main(seqfile, posfiles)
