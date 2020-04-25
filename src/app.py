import numpy as np 
import os
import re
import sys
from collections import Counter
import gc
import random
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from itertools import chain
import torch
import torch.nn as nn
import torch.optim as optim
import torch.functional as F
from torch.utils.data import DataLoader, Dataset
from textblob import Word
from nltk.stem import WordNetLemmatizer 
from nltk.stem import PorterStemmer
import json
from utils import *
from preprocess import *
from flask import Flask, jsonify, request,render_template

app = Flask(__name__)

@app.route("/", methods=["GET","POST"])
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    sentence = request.form.get("Sentence")
    MODEL_PATH = "../model/modelv1.zip"
    JSON_PATH = "../data"
    device = "cpu"

    model = torch.jit.load(MODEL_PATH).to(device)

    word2idx = load_dict(JSON_PATH, "word2idx.json")
    word2idx = {key:int(value) for key, value in word2idx.items()}
    label2idx = load_dict(JSON_PATH, "label2idx.json")
    label2idx = {key:int(value) for key, value in label2idx.items()}


    idx2word = {value: key  for key, value in word2idx.items()}
    idx2label = {value: key  for key, value in label2idx.items()}

    response_json = predict_tags(sentence, model,word2idx=word2idx, 
                idx2word=idx2word, label2idx=label2idx, idx2label=idx2label)

    print(response_json)
    return render_template("response.html", data=response_json)

def generate_predictions(model, X, seq, device="cpu"):
    '''
    generate predictions for model, X and length sequence

    --Parameters:
        model - torchscript model
        X - indexed sequence of sentence
        seq - length of the sequence
    
    --Returns
        tensor of length seq having indices of the predictions

    '''
    X = X.to(device)
    seq = seq.to(device)
    pred = model(X, seq)
    pred_labels = torch.argmax(pred, 2)
    pred_labels = pred_labels.view(-1)
    return pred_labels


def predict_tags(sentence, model, word2idx, 
                idx2word, label2idx, idx2label):
    '''
    inference functions that generates a json response of token wise predictions

    --Parameters:
        sentence - string. Raw input
        model - torchscript model graph
        word2idx - mapping from words to unique indices
        idx2word - reverse mapping of word2idx
        label2idx - mapping from labels to unique indices
        idx2label - reverse mapping of label2idx
    
    --Returns:
        response_json - {
            "token1":"prediction_label1",
            "token2":"prediction_label2".
            .
            .
            .
        }

    '''
    tokens = tokenize(sentence)
    length = len(tokens)
    tokens_idx = []
    for token in tokens:
        if token not in word2idx.keys():
            tokens_idx.append(word2idx["UNK"])
        else:
            tokens_idx.append(word2idx[token])
    
    tokens_idx = torch.LongTensor(tokens_idx).unsqueeze(0)
    sequence = torch.LongTensor([length])
    response_json = {}
    print(tokens_idx, sequence)
    predictions = generate_predictions(model, tokens_idx, sequence)
    for token, label in zip(tokens, predictions):
        pred_tag = idx2label[label.item()]
        response_json[token] = pred_tag
        print(token," ----> ", pred_tag)
    return response_json





if __name__ == "__main__":
    app.run(debug=True)