import os
import urllib.request
import zipfile
import tarfile
import glob
import io
import string
import re
import random

import torchtext
from torchtext.vocab import Vectors


class GetDataLoader():
    def __call__(self):
        self.make_data_folder()
        self.display_print('finish make folder')

        self.get_imdb()
        self.display_print('get imdb')

        self.open_tar()
        self.display_print('open tar file')

        self.data_to_tsv()
        self.display_print('to tsv')

        dl_dict, TEXT = self.make_ds_dl()
        self.display_print('finish dataloader')
        return dl_dict, TEXT

    def display_print(self, word: str):
        print('-'*15)
        print(word)
        print('-'*15)

    def make_data_folder(self):
        data_dir = "./data/"
        if not os.path.exists(data_dir):
            os.mkdir(data_dir)

    def open_tar(self):
        tar = tarfile.open('./data/aclImdb_v1.tar.gz')
        tar.extractall('./data/')  # 解凍
        tar.close()

    def get_imdb(self):
        url = 'http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz'

        save_path = "./data/aclImdb_v1.tar.gz"
        if not os.path.exists(save_path):
            urllib.request.urlretrieve(url, save_path)

    def data_to_tsv(self):
        train_test = ['train', 'test']
        data_dict = {'1': 'pos', '0': 'neg'}
        for i in train_test:
            fs = open('./data/{}.tsv'.format(i), 'w')

            for num, kind in data_dict.items():
                path = './data/aclImdb/{}/{}/'.format(i, kind)

                for fname in glob.glob(os.path.join(path, '*.txt')):
                    with io.open(fname, 'r', encoding="utf-8") as f:
                        text = f.readline()

                        text = text.replace('\t', ' ')

                        text = text + '\t' + num + '\t' + '\n'

                        fs.write(text)

    def fasteText(self):
        url = "https://dl.fbaipublicfiles.com/fasttext/vectors-english/wiki-news-300d-1M.vec.zip"
        save_path = "./data/wiki-news-300d-1M.vec.zip"
        if not os.path.exists(save_path):
            urllib.request.urlretrieve(url, save_path)

        zip = zipfile.ZipFile("./data/wiki-news-300d-1M.vec.zip")
        zip.extractall("./data/")  # ZIPを解凍
        zip.close()

    def pp_text(self, text):
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

    def split_text(self, text):
        return text.strip().split()

    def tokenzier_with_pp(self, text):
        text = self.pp_text(text)
        ret = self.split_text(text)
        return ret

    def make_ds_dl(self):
        max_length = 256

        TEXT = torchtext.data.Field(
            sequential=True,
            tokenize=self.tokenzier_with_pp,
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

        self.fasteText()
        english_word_to_vector = Vectors(name='data/wiki-news-300d-1M.vec')
        TEXT.build_vocab(
            train_ds, vectors=english_word_to_vector, min_freq=10)

        train_dl = torchtext.data.Iterator(
            train_ds, batch_size=32, train=True)
        # train_dl = torchtext.data.Iterator(
        #     train_ds, batch_size=32, train=True, shuffle=True

        val_dl = torchtext.data.Iterator(
            val_ds, batch_size=32, train=False, sort=False)

        test_dl = torchtext.data.Iterator(
            test_ds, batch_size=32, train=False, sort=False)

        dl_dict = {'train': train_dl, 'val': val_dl, 'test': test_dl}

        return dl_dict, TEXT


dl_class = GetDataLoader()
dl_dict, TEXT = dl_class()
