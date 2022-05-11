import torch
from minicons import cwe
from bs4 import BeautifulSoup
import argparse
import glob
import plotly.express as px
import numpy as np
from sklearn.manifold import TSNE
import pandas as pd
from nltk.tokenize import sent_tokenize
import textwrap


DISCARD = "[<< | >>]"


def wrap_text(text):
    return "<br>".join(textwrap.wrap(text))


def get_text_from_file(filename):
    with open(filenames[0]) as f:
        soup = BeautifulSoup(f, "html.parser")
    lines = soup.get_text().replace(u"\xa0", u" ").split("\n")
    lines = filter(lambda x: not (DISCARD in x or x == ""), lines)
    text = "\n".join(lines)
    text = " ".join(text.split())
    return text


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--pattern", default="data/summa/FP/FP001.html")
    args = parser.parse_args()
    filenames = glob.glob(args.pattern)

    model = cwe.CWE("bert-base-uncased", device="cpu")

    for filename in filenames:
        text = get_text_from_file(filename)

    sentences = list(filter(lambda x: x != "" and x != " ", sent_tokenize(text)))[:500]

    phrases = list(zip(sentences, sentences))
    representations = model.extract_representation(phrases)

    X = np.array(representations)
    print("Shape of Representations:", X.shape)

    X_embedded = TSNE(n_components=2, init="pca").fit_transform(X)

    df = pd.DataFrame(
        {
            "sentence": sentences,
            "sentence_wrap": list(map(wrap_text, sentences)),
            "x": X_embedded[:, 0],
            "y": X_embedded[:, 1],
        }
    )

    fig = px.scatter(df, x="x", y="y", hover_data=["sentence_wrap"])
    fig.show()
