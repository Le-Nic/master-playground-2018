from inputhandler.input_reader import InputReader
from sklearn.preprocessing import OneHotEncoder
from datetime import datetime
import numpy as np
import tables as tb
import math
import pathlib
import time


class PreProcessing:

    def __init__(self, configs, mappings=None):

        # get configs. information
        self.io = configs['io']
        self.col = configs['pp']
        self.normalization = configs['normalization']
        self.lbl = configs['label']

        if mappings:
            None  # TODO: check and get dictionaries mapping for 1-hot, able to continue where it left off
        else:
            self.x_map = {}  # mappings for features
            self.y_map = []  # mappings for labels
            self.x_unq = {}  # sets for uniques
            self.y_unq = {}
            self.x_len = {}  # length for arrays init.
            self.val = {}  # values for normalization

        # get datasets readers from specified directory
        self.trainsets = self._get_files(self.io['train_dir'])
        self.testsets = self._get_files(self.io['test_dir'])

        if self.trainsets is None and self.testsets is None:
            raise FileNotFoundError("[Error] [PreProcessing] No input files found")

        # variables initialization
        self.instances = 0  # used for Mean / Std Dev. Calculation
        self.features_n = self.trainsets[0]['reader'].features_n  # used for preallocating arrays

        # check labels input
        if self.lbl['i']:
            self.check_lbl_unq = self._unq_lbl
            self.lbl['i'] = list(range(len(self.lbl['i'])))  # re-index labels

            if len(self.lbl['i']) == 2 and len(self.lbl['lbl_normal']) == 2:
                self.lbl_type_n = 4
            else:
                self.lbl_type_n = 2
        else:
            self.lbl_type_n = 0
            self.check_lbl_unq = self._none

        # features mapping
        self.columns_map = list(range(self.features_n))  # used for mapping original column(s) to processed output(s)
        self.unprocessed_i = self.columns_map.copy()  # used for tracking remaining unprocessed column(s)
        for col_num in sorted([col_num for cols in self.col.values() for col_num in cols], reverse=True):
            del self.unprocessed_i[col_num]

        # initialize sets storing uniques
        for col_num in (self.col['1hot'] +  # 1-hot columns
                        ([self.col['ips'][0]] if self.col['ips'] else [])):  # 1st column for storing IPs
            self.x_unq[col_num] = set()

        for col_num in self.lbl['i']:
            self.y_unq[col_num] = set()

        # initialize dict for Normalization
        for col_num in self.col['norm']:
            self.val[col_num] = {
                'v1': 999999999.,  # Min / Mean
                'v2': -999999999.  # Max / Std Dev.
            }

        # normalization function assignment
        if self.normalization == 'minmax1r':
            self.normalize = self._minmax1r_norm
            self.get_norm_val = self._get_minmax
            print("[PreProcessing] Normalization chosen: Min-Max (0 to 1)")

        elif self.normalization == 'minmax2r':
            self.normalize = self._minmax2r_norm
            self.get_norm_val = self._get_minmax
            print("[PreProcessing] Normalization chosen: Min-Max (-1 to 1)")

        elif self.normalization == 'zscore':
            self.normalize = self._zscore_norm
            self.get_norm_val = self._get_sum
            print("[PreProcessing] Normalization chosen: Z-Score (zero Mean, unit Variance)")

            for col_num in self.col['norm']:
                self.val[col_num]['v1'] = .0
        else:
            self.normalize = self._none
            self.get_norm_val = self._none
            print("[PreProcessing] Normalization omitted")

    def _get_files(self, data_dir):
        """ assign Reader for each dataset found """
        if data_dir is None:
            return None

        datasets = []
        file_count = 0

        data_path = pathlib.Path(data_dir)

        if data_path.is_dir():
            for child in data_path.iterdir():
                file_count += 1

                datasets.append({
                    'name': child.stem,
                    'reader': InputReader(str(child), label_loc=self.lbl['i'], dtypes=self.io['dtypes_in'],
                                          parse_dates=self.io['dates'], read_chunk_size=self.io['read_chunk_size'])
                }) if pathlib.Path(child).is_file() else None

        elif data_path.is_file():
            file_count += 1

            datasets.append({
                'name': data_path.stem,
                'reader': InputReader(data_dir, label_loc=self.lbl['i'], dtypes=self.io['dtypes_in'],
                                      parse_dates=self.io['dates'], read_chunk_size=self.io['read_chunk_size'])
            })

        print("[PreProcessing]", file_count, "file(s) found in >", data_dir)
        return datasets

    def get_metadata(self):
        """ obtain all required info. for Pre-Processing step """

        # first iteration (obtain uniques, median/max/min)
        for trainset in self.trainsets:
            t = time.time()
            print("[PreProcessing] [metadata] Processing >", trainset['name'])

            x, y = (None,) * 2
            next_chunk = True
            instance_num = 0

            while next_chunk:
                x, y, next_chunk = trainset['reader'].next()
                instance_num += self.io['read_chunk_size']

                for col_num in (self.col['1hot']):  # 1-hot feature(s)
                    self.x_unq[col_num] |= set(x[:, col_num])

                for col_num in self.col['ips']:  # IPs
                    self.x_unq[self.col['ips'][0]] |= set(x[:, col_num])

                for col_num in self.val:  # Norm.
                    col = x[:, col_num]
                    self.get_norm_val(col, col_num)

                self.check_lbl_unq(y)

            instance_num -= (self.io['read_chunk_size'] - x.shape[0])
            self.instances += instance_num
            print("[PreProcessing] [metadata]", instance_num, "intance(s) iterated, time elapsed:", time.time() - t)

        # finishing calculation of mean / output info.
        for col_num in self.val:
            if self.normalization == 'zscore':
                self.val[col_num]['v1'] /= self.instances

            elif self.normalization == 'minmax1r' or self.normalization == 'minmax2r':
                print("[PreProcessing] [metadata] Attribute", col_num, "[Min]", self.val[col_num]['v1'],
                      "[Max]", self.val[col_num]['v2'])

        # second iteration (calculation of Std Dev.)
        if self.normalization == 'zscore':
            self._calc_stddev()

        # labels map creation
        if self.lbl['i']:
            labels = []
            # NOTE: prone to error if lbl_normal is not the same length with col['lbl']
            for i, col_num in enumerate(self.lbl['i']):
                labels.append(list(self.y_unq[col_num]))
                labels[i].insert(0, labels[i].pop(labels[i].index(self.lbl['lbl_normal'][i])))  # "normal" lbl to front

            # [0] -> 2 / others (binary)
            self.y_map.append({v: 1 for k, v in enumerate(labels[0])})
            self.y_map[-1][labels[0][0]] = 0

            # [0] -> 3 / others (multi-label, 1 column)
            self.y_map.append({v: k for k, v in enumerate(labels[0])})

            if self.lbl_type_n == 4:
                # [*] -> 5 (multi-label, "binary")
                # lbl_extra = {labels[0][0]: 0}
                # for k, v in enumerate(labels[1][1:]):
                #     lbl_extra[v] = k + 1
                self.y_map.append({v: k for k, v in enumerate(labels[1])})

                k = 1
                # [*] -> 9 (multi-label, 2 columns)
                lbl_extra = {labels[0][0] + labels[1][0]: 0}
                for v1 in labels[0][1:]:
                    for v2 in labels[1][1:]:
                        lbl_extra[v1+v2] = k
                        k += 1
                self.y_map.append(lbl_extra)

        # mapping features' column and increase dimension of output
        for col_num in self.col['1hot']:
            print("[PreProcessing] [metadata] Attribute", col_num, "[Uniques]", list(self.x_unq[col_num]))

            cols_ex = len(self.x_unq[col_num]) - 1  # length n after performing 1-hot encoding
            self.features_n += cols_ex  # add to the total number of columns

            for i, item in enumerate(self.columns_map[col_num + 1:]):  # make way for extra columns (1hot)
                self.columns_map[i + col_num + 1] += cols_ex

        for col_num in self.col['t']:  # epoch time, sin (day), cos (day), sin (week), sin (week)
            self.features_n += 4

            for i, item in enumerate(self.columns_map[col_num + 1:]):
                self.columns_map[i + col_num + 1] += 4

        for col_num in (self.col['1hot'] + [self.col['ips'][0]]):
            self.x_map[col_num] = {v: k for k, v in enumerate(self.x_unq[col_num])}
            self.x_len[col_num] = len(self.x_map[col_num])  # unused for IP field

        for col_num in self.col['flg']:  # length-6 binary (flg)
            self.features_n += 5

            for i, item in enumerate(self.columns_map[col_num + 1:]):
                self.columns_map[i + col_num + 1] += 5

        for col_num in self.col['8bit']:  # length-8 binary (fwd and tos)
            self.features_n += 7

            for i, item in enumerate(self.columns_map[col_num + 1:]):
                self.columns_map[i + col_num + 1] += 7

        for col_num in self.col['16bit']:  # length-16 binary (ports)
            self.features_n += 15

            for i, item in enumerate(self.columns_map[col_num + 1:]):
                self.columns_map[i + col_num + 1] += 15

        for col_num in self.col['rm']:  # column(s) to remove
            self.features_n -= 1
            self.columns_map[col_num] = None

            for i, item in enumerate(self.columns_map[col_num + 1:]):
                if item is not None:
                    self.columns_map[i + col_num + 1] -= 1

        print("[PreProcessing] [metadata] Saving meta info. >", self.io['output_dir'] + "/mappings.hd5")

        pathlib.Path(self.io['output_dir']).mkdir(parents=True, exist_ok=True)
        meta_fo = tb.open_file(self.io['output_dir'] + "/mappings.hd5", mode='w')

        # features' header
        x_map = np.empty(self.features_n, dtype="S32")
        x_map[:] = ''
        for col, col_nums in self.col.items():
            if col == 't':
                cyclic_type = ["_sin", "_cos"]
                date_type = [" (day)", " (day)", " (month)", " (month)"]
                for i, col_num in enumerate(col_nums):
                    x_map[self.columns_map[col_num]] = col + str(i + 1)
                    for r in range(4):
                        x_map[self.columns_map[col_num]+r+1] = col + str(i+1) + cyclic_type[r % 2] + date_type[r]

            if col == '1hot':
                for i, col_num in enumerate(col_nums):
                    for j, unq in enumerate(self.x_unq[col_num]):
                        x_map[self.columns_map[col_num]+j] = col + str(i+1) + ": " + unq
            elif col == 'rm':
                pass
            else:
                for i, col_num in enumerate(col_nums):
                    x_map[self.columns_map[col_num]] = col + str(i+1)

        meta_fo.create_array(meta_fo.root, "x", x_map, "Features Header")

        # IPs List
        if self.col['ips']:
            meta_fo.create_array(meta_fo.root, "ip", np.array(list(self.x_unq[self.col['ips'][0]])), "Indexed IPs")

        # Labels Mapping
        for i, mapping in enumerate(self.y_map):
            y_map = np.empty(len(mapping), dtype="S32")

            for lbl, k in mapping.items():
                y_map[k] = lbl

            if i == 2:
                y_map[0] = self.lbl['lbl_normal'][0]
            meta_fo.create_array(meta_fo.root, "y"+str(i), y_map, "Labels Mapping")

        meta_fo.close()

        # lbl_extra = {labels[0][0]: 0}
        # for k, v in enumerate(labels[1][1:]):
        #     lbl_extra[v] = k + 1

    def _unq_lbl(self, y):
        for col_num in (self.lbl['i']):
            self.y_unq[col_num] |= set(y[:, col_num])

    def _get_minmax(self, col, col_num):
        """ get minimum and maximum of iterated values """
        min_i = np.argmin(col)
        max_i = np.argmax(col)

        min_val = col[min_i]
        max_val = col[max_i]

        if min_val < self.val[col_num]['v1']:
            self.val[col_num]['v1'] = min_val
            print("[PreProcessing] [metadata] Attribute", col_num, "[new Min]", min_val)

        if max_val > self.val[col_num]['v2']:
            self.val[col_num]['v2'] = max_val
            print("[PreProcessing] [metadata] Attribute", col_num, "[new Max]", max_val)

    def _get_sum(self, col, col_num):
        """ sum all iterated values for Mean calc. """
        self.val[col_num]['v1'] += np.sum(col, axis=0)

    def _calc_stddev(self):
        """ datasets iterated again to obtain Std Dev. """
        print("[PreProcessing] [metadata] Calculating Std Dev. ...")

        for trainset in self.trainsets:
            next_chunk = True
            while next_chunk:
                x, y, next_chunk = trainset['reader'].next()

                for col_num in self.val:
                    col = x[:, col_num]
                    self.val[col_num]['v2'] += np.sum((col - self.val[col_num]['v1']) ** 2)

        for col_num in self.val:
            self.val[col_num]['v2'] = math.sqrt(self.val[col_num]['v2'] / (self.instances - 1))
            print("[PreProcessing] [metadata] Attribute", col_num, "[Mean]", self.val[col_num]['v1'],
                  "[Std Dev.]", self.val[col_num]['v2'])

    @staticmethod
    def _get_normalized_time(epoch):
        datetime_utc = datetime.utcfromtimestamp(epoch)

        return (datetime_utc.isoweekday() / 7), \
            ((datetime_utc.hour * 3600) + (datetime_utc.minute * 60) + datetime_utc.second) / 86400  # 24*60*60

    def transform_trainset(self):
        """ starts preprocessing of training dataset(s) """
        pathlib.Path(self.io['output_dir']).mkdir(parents=True, exist_ok=True)

        for trainset in self.trainsets:
            print("[PreProcessing] [operation] Processing trainset >", trainset['name'])
            train_fo = tb.open_file(self.io['output_dir'] + "/" + trainset['name'] + ".hd5", mode='w')

            array_x = train_fo.create_earray(train_fo.root, "x", tb.Float64Atom(shape=()),
                                             (0, self.features_n), "Feature Data")

            array_ys = []
            lbl_group = train_fo.create_group(train_fo.root, "y")
            for n in range(self.lbl_type_n):
                array_ys.append(train_fo.create_earray(lbl_group, "y" + str(n), tb.Int32Atom(),
                                                       (0,), "Label type " + str(n)))

            next_chunk = True
            while next_chunk:
                x, y, next_chunk = trainset['reader'].next()
                cur_shape = self.io['read_chunk_size'] if next_chunk else x.shape[0]
                x_new = np.zeros((cur_shape, self.features_n))
                y_new = np.zeros(cur_shape)

                # t (gmt)
                for col_num in self.col['t']:
                    epochs = x[:, col_num].astype('datetime64[s]').astype('int64')  # s

                    x_new[:, self.columns_map[col_num]] = epochs
                    day_ofweek, sec_ofday = np.vectorize(self._get_normalized_time)(epochs)  # sun, 0 - sat, 6

                    if self.normalization == 'minmax1r':
                        x_new[:, self.columns_map[col_num]+1] = (np.sin(2 * np.pi * sec_ofday) + 1) / 2
                        x_new[:, self.columns_map[col_num]+2] = (np.cos(2 * np.pi * sec_ofday) + 1) / 2
                    else:  # zscore normalization is not taken care of
                        x_new[:, self.columns_map[col_num]+1] = np.sin(2 * np.pi * sec_ofday)
                        x_new[:, self.columns_map[col_num]+2] = np.cos(2 * np.pi * sec_ofday)

                    # df = pd.DataFrame()  # import pandas as pd
                    # df['sine'] = np.sin(2 * np.pi * sec_ofday)
                    # df['cosine'] = np.cos(2 * np.pi * sec_ofday)
                    # df.plot.scatter('sine', 'cosine').set_aspect('equal')
                    # plt.show()  # import matplotlib.pyplot as plt

                # ips
                for col_num in self.col['ips']:
                    x_new[:, self.columns_map[col_num]] = np.vectorize(
                        self.x_map[self.col['ips'][0]].__getitem__)(x[:, col_num])

                # norm
                for col_num in self.col['norm']:
                    x_new[:, self.columns_map[col_num]] = self.normalize(x[:, col_num], col_num)

                # 1hot
                for col_num in self.col['1hot']:
                    unq_i = np.vectorize(self.x_map[col_num].__getitem__)(x[:, col_num])

                    cols_1hot = np.zeros((cur_shape, self.x_len[col_num]))
                    cols_1hot[np.arange(cur_shape), unq_i] = 1

                    x_new[:, self.columns_map[col_num]:self.columns_map[col_num] + self.x_len[col_num]] = cols_1hot

                # flg
                for col_num in self.col['flg']:
                    x_new[:, self.columns_map[col_num]:self.columns_map[col_num] + 6] = np.bitwise_not(
                        x[:, col_num].astype('S6').view('S1').reshape((cur_shape, -1)) == b'.') * 1

                # 8 bit binary (fwd, tos)
                for col_num in self.col['8bit']:
                    x_new[:, self.columns_map[col_num]:self.columns_map[col_num] + 8] = np.unpackbits(
                        x[:, col_num].astype('uint8')).reshape((cur_shape, -1))

                # 16 bit binary (ports)
                for col_num in self.col['16bit']:
                    for i, ele in enumerate(x[:, col_num]):  # detect ICMP type & code in protocol field
                        if not float(ele).is_integer():  # if ICMP code is 0, it will not be detected
                            icmp_field = ele.split('.')  # type, code
                            # order is reverse, ICMP code is shifted 8 bits to the left
                            x[i, col_num] = int(icmp_field[0]) + int(icmp_field[1]) * 256  # 8-bit code, 8-bit type

                    x_new[:, self.columns_map[col_num]:self.columns_map[col_num] + 16] = np.unpackbits(
                        x[:, col_num].astype('float32').astype('>i2').view('uint8')).reshape((cur_shape, -1))

                # others
                for i in self.unprocessed_i:
                    x_new[:, self.columns_map[i]] = x[:, i]

                array_x.append(x_new)

                for i, mapping in enumerate(self.y_map[:2]):  # class 2 & 3
                    y_new[:] = np.vectorize(mapping.__getitem__)(y[:, 0])
                    array_ys[i].append(y_new)

                for mapping in self.y_map[2:3]:  # class 5
                    y_new[:] = np.vectorize(mapping.__getitem__)(y[:, 1])
                    array_ys[2].append(y_new)

                for mapping in self.y_map[3:4]:  # class 9
                    y_new[:] = np.vectorize(mapping.__getitem__)(np.core.defchararray.add(
                        y[:, 0].astype(str), y[:, 1].astype(str)))
                    array_ys[3].append(y_new)

            train_fo.close()

        return True

    def transform_testset(self):
        """ starts preprocessing of training dataset(s) """
        pathlib.Path(self.io['output_dir']).mkdir(parents=True, exist_ok=True)

        for testset in self.testsets:
            print("[PreProcessing] [operation] Processing testset >", testset['name'])
            test_fo = tb.open_file(self.io['output_dir'] + "/" + testset['name'] + ".hd5", mode='w')

            array_x = test_fo.create_earray(test_fo.root, "x", tb.Float64Atom(),
                                            (0, self.features_n), "Feature Data")

            array_ys = []
            lbl_group = test_fo.create_group(test_fo.root, "y")
            for n in range(self.lbl_type_n):
                array_ys.append(test_fo.create_earray(lbl_group, "y"+str(n), tb.Int32Atom(),
                                                      (0,), "Label type "+str(n)))

            next_chunk = True
            while next_chunk:
                x, y, next_chunk = testset['reader'].next()
                cur_shape = self.io['read_chunk_size'] if next_chunk else x.shape[0]
                x_new = np.zeros((cur_shape, self.features_n))
                y_new = np.zeros(cur_shape)

                # t
                for col_num in self.col['t']:
                    epochs = x[:, col_num].astype('datetime64[s]').astype('int64')  # s

                    x_new[:, self.columns_map[col_num]] = epochs
                    day_ofweek, sec_ofday = np.vectorize(self._get_normalized_time)(epochs)  # sun, 0 - sat, 6

                    if self.normalization == 'minmax1r':
                        x_new[:, self.columns_map[col_num]+1] = (np.sin(2 * np.pi * sec_ofday) + 1) / 2
                        x_new[:, self.columns_map[col_num]+2] = (np.cos(2 * np.pi * sec_ofday) + 1) / 2
                    else:
                        x_new[:, self.columns_map[col_num]+1] = np.sin(2 * np.pi * sec_ofday)
                        x_new[:, self.columns_map[col_num]+2] = np.cos(2 * np.pi * sec_ofday)

                # ips
                for col_num in self.col['ips']:
                    unq_i = np.vectorize(self.x_map[self.col['ips'][0]].get, otypes=[np.object])(x[:, col_num])
                    unq_i[unq_i == None] = -1
                    x_new[:, self.columns_map[col_num]] = unq_i.astype(np.int)

                # norm
                for col_num in self.col['norm']:
                    x_new[:, self.columns_map[col_num]] = self.normalize(x[:, col_num], col_num)

                # 1hot
                for col_num in self.col['1hot']:
                    unq_i = np.vectorize(self.x_map[col_num].get, otypes=[np.object])(x[:, col_num])
                    unq_i[unq_i == None] = self.x_len[col_num] + 1
                    enc = OneHotEncoder(n_values=self.x_len[col_num], handle_unknown="ignore")
                    enc.fit(np.array(list(range(self.x_len[col_num]))).reshape(-1, 1))

                    x_new[:, self.columns_map[col_num]:self.columns_map[col_num]+self.x_len[col_num]] = enc.transform(
                        unq_i.reshape(-1, 1)).toarray()

                # flg
                for col_num in self.col['flg']:
                    x_new[:, self.columns_map[col_num]:self.columns_map[col_num] + 6] = np.bitwise_not(
                        x[:, col_num].astype('S6').view('S1').reshape((cur_shape, -1)) == b'.') * 1

                # 8 bit binary (fwd, tos)
                for col_num in self.col['8bit']:
                    x_new[:, self.columns_map[col_num]:self.columns_map[col_num] + 8] = np.unpackbits(
                        x[:, col_num].astype('uint8')).reshape((cur_shape, -1))

                # 16 bit binary (ports)
                for col_num in self.col['16bit']:
                    for i, ele in enumerate(x[:, col_num]):
                        if not float(ele).is_integer():
                            icmp_field = ele.split('.')
                            x[i, col_num] = int(icmp_field[0]) + int(icmp_field[1]) * 256

                    x_new[:, self.columns_map[col_num]:self.columns_map[col_num] + 16] = np.unpackbits(
                        x[:, col_num].astype('float32').astype('>i2').view('uint8')).reshape((cur_shape, -1))

                # others
                for i in self.unprocessed_i:
                    x_new[:, self.columns_map[i]] = x[:, i]

                array_x.append(x_new)

                for i, mapping in enumerate(self.y_map[:2]):  # 0: 2-class, 1: 3-class
                    y_new[:] = np.vectorize(mapping.__getitem__)(y[:, 0])
                    array_ys[i].append(y_new)

                for mapping in self.y_map[2:3]:  # 2: 5-class
                    y_new[:] = np.vectorize(mapping.__getitem__)(y[:, 1])
                    array_ys[2].append(y_new)

                for mapping in self.y_map[3:4]:  # 3: 9-class
                    y_new[:] = np.vectorize(mapping.__getitem__)(np.core.defchararray.add(
                        y[:, 0].astype(str), y[:, 1].astype(str)))
                    array_ys[3].append(y_new)

            test_fo.close()

        return True

    def _zscore_norm(self, x, col_num):
        """ Z-Standardization (zero Mean, unit Variance) """
        return (x - self.val[col_num]['v1']) / self.val[col_num]['v2']

    def _minmax2r_norm(self, x, col_num):
        """ Min-Max Normalization (range: -1 to 1) """
        return 2 * (x - self.val[col_num]['v1']) / (self.val[col_num]['v2'] - self.val[col_num]['v1']) - 1

    def _minmax1r_norm(self, x, col_num):
        """ Min-Max Normalization (range: 0 to 1) """
        return (x - self.val[col_num]['v1']) / (self.val[col_num]['v2'] - self.val[col_num]['v1'])

    def _none(*arg):
        pass

    '''
pp_config = {
    'io': {
        'read_chunk_size': 1000000,
        'train_sets': {
            'dir': 'E:/data/CIDDS-001/OpenStack',
            'labels': -4,
        },
        'test_sets': None,
        'output_dir': 'processed'
    },
    'normalization': 'minmax2r',  # zscore, minmax1r
    'pp': {
        't': 0,
        'ips': [3, 5],
        'pts': [4, 6],
        '1hot': [2],
        'flg': 10,  # .A.... -> 010000 (6)
        '8bit': [11],  # 4 -> 00000100 (8)
        'norm': [1, 7, 8, 9]
    }
}
    '''