from sklearn.model_selection import StratifiedShuffleSplit
from collections import Counter
import numpy as np
import tables as tb
import pathlib
import time


class DatasetShuffler:
    def __init__(self, configs):
        self.io = configs
        self.splitter = None

        # dataset
        input_path = pathlib.Path(self.io['input_path'])
        if input_path.is_file():
            self.io['input_name'] = input_path.stem
            print("[hd5huffler] File used for shuffling >", self.io['input_path'])
        else:
            raise FileNotFoundError("[Error] [hd5huffler] Input file not given")

        # meta
        meta_path = pathlib.Path(self.io['meta_path'])
        if meta_path.is_file():
            print("[hd5huffler] Meta file used >", self.io['meta_path'])
        else:
            raise FileNotFoundError("[Error] [hd5huffler] Meta file not given")

        input_r = tb.open_file(self.io['input_path'], mode='r')
        meta_r = tb.open_file(self.io['meta_path'], mode='r')

        try:
            self.y = input_r.get_node('/y/y3').read()
            self.y_dict = dict(enumerate(meta_r.get_node('/y3').read().astype('str')))
            self.y_len = 4
            print("[hd5huffler] Labels chosen:", self.y_dict)

        except tb.exceptions.NoSuchNodeError:
            self.y = input_r.get_node('/y/y1').read()
            self.y_dict = dict(enumerate(meta_r.get_node('/y1').read().astype('str')))
            self.y_len = 2
            print("[hd5huffler] Labels chosen:", self.y_dict)

        if len(self.y[0]) > 1:
            self.seq = input_r.get_node('/seq').read()
            self.y = self.y[range(self.y.shape[0]), self.seq-1]
            print("[hd5huffler] M:N labeling used, shuffling based on label in the last sequence")
        else:
            self.seq = None

        self.x_dummy = np.zeros(input_r.get_node('/x').shape[0])
        print("[hd5huffler] Total dataset size:", self.x_dummy.shape[0])

        input_r.close()
        meta_r.close()

    def _get_writers(self, h5_w, x_r, seq_r, t_r, ip_r, ys_r):
        x_shape = x_r[0].shape

        x_w = h5_w.create_earray(h5_w.root, "x", tb.Float64Atom(), (0, x_shape[0], x_shape[1]), x_r._g_gettitle())

        t_w = h5_w.create_earray(h5_w.root, "t", tb.Float64Atom(), (0,), t_r._g_gettitle())
        ip_w = h5_w.create_earray(h5_w.root, "ip", tb.Int32Atom(), (0, 2), ip_r._g_gettitle())
        seq_w = h5_w.create_earray(h5_w.root, "seq", tb.Int32Atom(), (0,), seq_r._g_gettitle())

        y_group = h5_w.create_group(h5_w.root, "y")
        ys_w = [h5_w.create_earray(y_group, "y" + str(n), tb.Int32Atom(), (0,) if self.seq is None else (0, x_shape[0]),
                                   y._g_gettitle()) for n, y in enumerate(ys_r)]

        return x_w, t_w, ip_w, seq_w, ys_w

    def shuffle(self, n_splits=1, test_size=0.1):
        self.splitter = StratifiedShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=147)

        h5_r = tb.open_file(self.io['input_path'], mode='r')
        x_r = h5_r.get_node('/x')
        t_r = h5_r.get_node('/t')
        ip_r = h5_r.get_node('/ip')
        seq_r = h5_r.get_node('/seq')
        ys_r = [h5_r.get_node('/y/y' + str(i)) for i in range(self.y_len)]

        if t_r.shape[0] == 0 or ip_r.shape[0] == 0:
            extra_contents = False
        else:
            extra_contents = True

        t_read = t_r.read()
        ip_read = ip_r.read()
        seq_read = seq_r.read()
        ys_read = [y_r.read() for y_r in ys_r]

        pathlib.Path(self.io['output_dir']).mkdir(parents=True, exist_ok=True)

        for n_split, (train_i, test_i) in enumerate(self.splitter.split(self.x_dummy, self.y)):  # loop [n_splits]

            time_elapsed = time.time()

            if n_split == 0:
                print("Train set", Counter(self.y[train_i]))
                print("Test Set", Counter(self.y[test_i]))

            for dataset_i, indexes in enumerate([train_i, test_i]):  # loop [train, test]

                if dataset_i == 0:
                    output_name = self.io['output_dir'] + "/" + self.io['input_name'] + "_train_" + str(n_split)
                    h5_w = tb.open_file(output_name + ".hd5", mode='w')

                else:
                    output_name = self.io['output_dir'] + "/" + self.io['input_name'] + "_test_" + str(n_split)
                    h5_w = tb.open_file(output_name + ".hd5", mode='w')

                # HD5 Writer
                x_hd5, t_hd5, ip_hd5, seq_hd5, ys_hd5 = self._get_writers(
                    h5_w, x_r, seq_r, t_r, ip_r, ys_r)

                flow_n = 0
                for i in indexes:
                    x = x_r[i]
                    seq = seq_read[i]

                    # x
                    x_hd5.append([x])

                    # others
                    seq_hd5.append([seq])
                    if extra_contents:
                        t_hd5.append([t_read[i]])
                        ip_hd5.append([ip_read[i]])

                    # y
                    for n, y_w in enumerate(ys_hd5):
                        y = [ys_read[n][i]]
                        y_w.append(y)

                    flow_n += 1
                    print(flow_n, end='\r')

                h5_w.close()

                if self.seq is not None:
                    counts = {}
                    for i in indexes:
                        for each_seq in range(seq_read[i]):
                            try:
                                counts[ys_read[-1][i][each_seq]] += 1
                            except KeyError:
                                counts[ys_read[-1][i][each_seq]] = 1

                    print("Actual Labels:", ", ".join([str(k) + ": " + str(v) for k, v in counts.items()]))

            print("[hd5huffler] dataset", n_split, "created with ", test_size, "test ratio, time elapsed:",
                  time.strftime("%H:%M:%S", time.gmtime(time.time() - time_elapsed)))

        h5_r.close()
