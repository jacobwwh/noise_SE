import os
import json
import random
import torch
import csv
from torch.utils.data import DataLoader, Dataset
from utils.dataset_utils import convert_examples_to_features_noisyclassification
data_dir='/mnt/nvme/noise_se/under/data/devign'
num_classes=104

def generate_devigndata():
    csv.field_size_limit(10 * 1024 * 1024)
    train_data = []
    test_data = []
    valid_data = []
    for i in range(num_classes):
        train_data.append([])
        valid_data.append([])
        test_data.append([])
    with open(os.path.join(data_dir, 'train.csv'), 'r') as csv_file:
        # Create a CSV reader object.
        csv_reader = csv.reader(csv_file)
        # Read and process each row in the CSV file.
        for row in csv_reader:
            # Each 'row' is a list of values from a line in the CSV file.
            # You can access the values using list indexing.
            # For example, row[0] will give you the first column's value.
            lab = int(row[2])
            train_data[lab].append({'code':row[1].replace('\n',' '), 'label':lab,'original_label':lab})

    with open(os.path.join(data_dir, 'valid.csv'), 'r') as csv_file:
        # Create a CSV reader object.
        csv_reader = csv.reader(csv_file)
        # Read and process each row in the CSV file.
        for row in csv_reader:
            # Each 'row' is a list of values from a line in the CSV file.
            # You can access the values using list indexing.
            # For example, row[0] will give you the first column's value.
            lab = int(row[2])
            valid_data[lab].append({'code':row[1].replace('\n',' '), 'label':lab,'original_label':lab})

    with open(os.path.join(data_dir, 'test.csv'), 'r') as csv_file:
        # Create a CSV reader object.
        csv_reader = csv.reader(csv_file)
        # Read and process each row in the CSV file.
        for row in csv_reader:
            # Each 'row' is a list of values from a line in the CSV file.
            # You can access the values using list indexing.
            # For example, row[0] will give you the first column's value.
            lab = int(row[2])
            test_data[lab].append({'code':row[1].replace('\n',' '), 'label':lab,'original_label':lab})

    return train_data, valid_data, test_data