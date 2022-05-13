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
from nltk.tokenize.punkt import PunktSentenceTokenizer, PunktParameters
import textwrap
import regex as re
from tqdm import tqdm


DISCARD = "[<< | >>]"


def wrap_text(text):
    return "<br>".join(textwrap.wrap(text))


def get_text_from_file(filename):
    with open(filename) as f:
        soup = BeautifulSoup(f, "html.parser")
    lines = soup.get_text().replace("\xa0", " ").split("\n")
    lines = filter(lambda x: not (DISCARD in x or x == ""), lines)
    text = "\n".join(lines)
    text = " ".join(text.split())
    text = text.replace("---", " --- ")
    return text


def get_abbrevs():
    regex = re.compile("\([^\)]*\.")
    abbrevs = set()

    for filename in tqdm(glob.glob(args.pattern)):
        text = get_text_from_file(filename)
        occurences = re.findall(regex, text, overlapped=True)
        occurences = [
            x.replace("(", "").replace(".", "").split()[-1] for x in occurences
        ]
        abbrevs = abbrevs.union(occurences)

    with open(args.abbrev_file, "w") as f:
        f.write("\n".join(abbrevs))


def analyze():
    # initialize transformer model
    model = cwe.CWE("bert-base-uncased", device="cpu")

    # read abbreviations from file and create tokenizer
    with open(args.abbrev_file, "r") as f:
        abbrevs = f.read().lower().split()
    punkt_param = PunktParameters()
    punkt_param.abbrev_types = set(abbrevs)
    tokenizer = PunktSentenceTokenizer(punkt_param)

    # read texts
    texts = []
    for filename in tqdm(glob.glob(args.pattern)):
        texts.append(get_text_from_file(filename))
    text = " ".join(texts)

    sentences = tokenizer.tokenize(text)
    print(f"Number of sentences: {len(sentences)}")

    # get sentences and phrases
    sentences = list(filter(lambda x: x != "" and x != " ", sentences))[:500]

    phrases = list(zip(sentences, sentences))
    representations = model.extract_representation(phrases)

    X = np.array(representations)
    print(f"Shape of Representations: {X.shape}")

    # dimensionality reduction
    X_embedded = TSNE(n_components=2, init="pca").fit_transform(X)

    # plotting
    df = pd.DataFrame(
        {
            "sentence": sentences,
            "sentence_wrap": list(map(wrap_text, sentences)),
            "x": X_embedded[:, 0],
            "y": X_embedded[:, 1],
            "representation": list(representations),
        }
    )
    df.to_csv("sentences.csv", index=False)
    fig = px.scatter(df, x="x", y="y", hover_data=["sentence_wrap"])
    fig.write_image("img/sentences.png")
    fig.show()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--pattern", default="data/summa/*/*.html")
    parser.add_argument("--abbrev_file", default="abbrevs.txt")
    args = parser.parse_args()

    # get_abbrevs()
    analyze()
