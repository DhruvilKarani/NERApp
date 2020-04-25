import re
import os
import sys
import json


def get_lines(path, filename):
    file_path = os.path.join(path, filename)
    with open(file_path, "r") as file:
        lines = file.readlines()
    
    data = []
    group = []

    for line in lines:
        if line == "\n":
            data.append(group)
            group = []
        else:
            pair = line.split()
            group.append((pair[1], pair[0]))       
    return data

def get_token_tags(data_generator):
    tags = []
    tokens = []
    dataset_size = 0
    for data in data_generator:
        dataset_size+=1
        for pair in data:
            tags.append(pair[1])
            tokens.append(pair[0])
    return tags, tokens

def save_dict(data, path, name):
    if type(data) != dict:
        raise TypeError("Data should be a dictionary")
    else:
        filepath = os.path.join(path, name)
        with open(filepath, "w") as f:
            json.dump(data, f)
            f.close()

def load_dict(path, name):
    filepath = os.path.join(path, name)
    with open(filepath, "r") as f:
        data = json.load(f)
        f.close()
    return data


if __name__ == "__main__":
    get_lines("../data", "train.txt")    