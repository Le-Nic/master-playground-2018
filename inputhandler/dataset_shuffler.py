from sklearn.model_selection import StratifiedShuffleSplit
import pathlib
import numpy as np
import tables as tb


class DatasetShuffler:
    def __init__(self, configs):
        self.io = configs
        self.splitter = None

        # dataset
        input_path = pathlib.Path(self.io['input_path'])
        if input_path.is_file():
            self.dataset = {
                'name': input_path.stem,
                'reader': tb.open_file(self.io['input_path'], mode='r')
            }
            print("[hd5huffler] File used for shuffling >", self.io['input_path'])
        else:
            raise FileNotFoundError("[Error] [hd5huffler] Input file not given")

        # meta
        meta_path = pathlib.Path(self.io['meta_path'])
        if meta_path.is_file():
            self.meta = {
                'name': meta_path.stem,
                'reader': tb.open_file(self.io['meta_path'], mode='r')
            }
            print("[hd5huffler] Meta file used >", self.io['meta_path'])
        else:
            raise FileNotFoundError("[Error] [hd5huffler] Meta file not given")

        try:
            self.y_dict = dict(enumerate(self.meta['reader'].get_node('/y3').read().astype('str')))
            self.y = self.dataset['reader'].get_node('/y/y3').read()
            print("[hd5huffler] Labels chosen:", self.y_dict)

        except tb.exceptions.NoSuchNodeError:
            self.y_dict = dict(enumerate(self.meta['reader'].get_node('/y1').read().astype('str')))
            self.y = self.dataset['reader'].get_node('/y/y1').read()
            print("[hd5huffler] Labels chosen:", self.y_dict)

        self.x_dummy = np.zeros(self.dataset['reader'].get_node('/x').shape[0])
        print("[hd5huffler] Total dataset size:", self.x_dummy.shape[0])

        self.dataset['reader'].close()
        self.meta['reader'].close()

    def shuffle(self, n_splits=1, test_size=0.1):
        self.splitter = StratifiedShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=147)

        from collections import Counter
        for train_i, test_i in self.splitter.split(self.x_dummy, self.y):
            Counter(self.y[test_i])
