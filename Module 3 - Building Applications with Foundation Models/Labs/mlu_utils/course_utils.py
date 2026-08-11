import matplotlib.pyplot as plt
import matplotlib.image as img
import pandas as pd
import os
import os.path
from os import path
from IPython.display import display, HTML
import codecs

######################################
def answer_html(message):
    f=codecs.open("../mlu_utils/solutions/"+ message + ".html", 'r')
    display(HTML(f.read()))
    return