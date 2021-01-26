import os
import urllib.request
import zipfile
import tarfile


def make_data_folder():
    data_dir = "./data/"
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)


def open_tar():
    tar = tarfile.open('./data/aclImdb_v1.tar.gz')
    tar.extractall('./data/')  # 解凍
    tar.close()


def main():
    make_data_folder()
    url = 'http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz'

    save_path = "./data/aclImdb_v1.tar.gz"
    if not os.path.exists(save_path):
        urllib.request.urlretrieve(url, save_path)

    open_tar()


if __name__ == '__main__':
    main()
