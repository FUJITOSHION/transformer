import glob
import os
import io
import string


def main():
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


if __name__ == '__main__':
    main()
