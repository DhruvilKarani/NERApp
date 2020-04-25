import re
import os
import sys
import json
sys.path.append("../src")
from preprocess import *


def test_replace_apostrophe():
    sentence = "He's a great guy"
    assert replace_apostrophe(sentence) == "He a great guy"
    sentence = "He's done. It's pretty"
    assert replace_apostrophe(sentence) == "He done. It pretty"
    sentence = "He is a great guy. Its good"
    assert replace_apostrophe(sentence) == "He is a great guy. Its good"

def test_replace_num():
    NUM = "NUM"
    sentence = "He was born in 19982"
    assert replace_num(sentence, NUM) == "He was born in NUM" 
    sentence = "He was born in 1998s2"
    assert replace_num(sentence, NUM) == "He was born in 1998s2"

def test_replace_nt():
    sentence = "They cant and shouldnt do that to him. It isnt fair"
    assert replace_nt(sentence) == "They can not and should not do that to him. It is not fair"

