import string
import re
import random
import os
import urllib.request
import zipfile

import torchtext
from torchtext.vocab import Vectors


def fasteText():
    url = "https://dl.fbaipublicfiles.com/fasttext/vectors-english/wiki-news-300d-1M.vec.zip"  # nopep
    save_path = "./data/wiki-news-300d-1M.vec.zip"
    if not os.path.exists(save_path):
        urllib.request.urlretrieve(url, save_path)

    zip = zipfile.ZipFile("./data/wiki-news-300d-1M.vec.zip")
    zip.extractall("./data/")  # ZIPを解凍
    zip.close()


def pp_text(text):
    # 改行コードを消去
    text = re.sub('<br />', '', text)

    for p in string.punctuation:
        if (p == '.') or (p == ','):
            continue
        else:
            text = text.replace(p, ' ')
    text = text.replace('.', ' . ')
    text = text.replace(',', ' , ')
    return text


def split_text(text):
    return text.strip().split()


def tokenzier_with_pp(text):
    text = pp_text(text)
    ret = split_text(text)
    return ret


def get_dl():
    max_length = 256

    TEXT = torchtext.data.Field(
        sequential=True,
        tokenize=tokenzier_with_pp,
        use_vocab=True,  # word to id
        lower=True,
        include_lengths=True,  # padをいれるか
        batch_first=True,
        fix_length=max_length,
        init_token='<cls>',
        eos_token='<eos>')

    LABEL = torchtext.data.Field(
        sequential=False,
        use_vocab=False
    )

    train_val_ds, test_ds = torchtext.data.TabularDataset.splits(
        path='./data/',
        train='train.tsv',
        test='test.tsv',
        format='tsv',
        fields=[('Text', TEXT), ('Label', LABEL)])

    train_ds, val_ds = train_val_ds.split(
        split_ratio=0.8,
        random_state=random.seed(2021))

    fasteText()
    english_word_to_vector = Vectors(name='data/wiki-news-300d-1M.vec')
    TEXT.build_vocab(train_ds, vectors=english_word_to_vector, min_freq=5)

    train_dl = torchtext.data.Iterator(
        train_ds, batch_size=32, train=True, shuffle=True)

    val_dl = torchtext.data.Iterator(
        val_ds, batch_size=32, train=False, sort=False)

    test_dl = torchtext.data.Iterator(
        test_ds, batch_size=32, train=False, sort=False)

    dl_dict = {'train': train_dl, 'val': val_dl, 'test': test_dl}

    return dl_dict, TEXT


if __name__ == '__main__':
    get_dl()
