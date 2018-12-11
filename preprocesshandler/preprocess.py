from inputhandler import input_reader
from sklearn.preprocessing import OneHotEncoder
from datetime import datetime
import numpy as np
import tables as tb
import math
import pathlib
import time


class PreProcessing:

    def __init__(self, configs):

        # get configs. information
        self.col = configs['pp']
        self.normalization = configs['normalization']
        self.lbl = configs['label']

        self.add_td = configs['add_td']
        self.td_scale = configs['td_scale']

        self.x_map = {}  # mappings for features (1hot & IP)
        self.y_map = []  # mappings for labels
        self.x_unq = {}  # sets for uniques
        self.y_unq = {}
        self.x_len = {}  # length for arrays init. (1hot)
        self.val = {}  # values for normalization

        # get datasets readers from specified directory
        try:
            self.io = configs['io_csv']
            self.trainsets = self._get_files(self.io['train_dir'], True)
            self.testsets = self._get_files(self.io['test_dir'], True)

            self.meta_migrate = False
            self.io['arff_output'] = None

        except KeyError:
            self.io = configs['io_hd5']
            self.trainsets = self._get_files(self.io['train_dir'], False)
            self.testsets = self._get_files(self.io['test_dir'], False)

            if self.trainsets is not None:
                self.io['read_chunk_size'] = self.io['read_chunk_size'] * self.trainsets[0]['reader'].sequence_n

            self.meta_migrate = True
            if self.io['arff_output'] != 1 and self.io['arff_output'] != 2:
                self.io['arff_output'] = None

        if self.io['arff_output'] == 1 or self.io['arff_output'] == 2:
            if self.io['arff_output'] == 1:
                print("[PreProcessing] Last instance will be saved as ARFF output")
            else:
                print("[PreProcessing] All instances will be saved as ARFF output")

            if not self.meta_migrate:
                print("[WARNING] [PreProcessing] ARFF output is not supported when using CSV input")

        if self.trainsets is None and self.testsets is None:
            raise FileNotFoundError("[Error] [PreProcessing] No input files found")

        # get number of label class for ARFF header
        if self.meta_migrate and self.io['arff_output']:
            self.y_arff = []
            if pathlib.Path(self.io['meta_path']).is_file():
                meta_fi = tb.open_file(self.io['meta_path'], mode='r')
                try:
                    for y_n in range(4):
                        y_node = meta_fi.get_node("/y" + str(y_n))
                        self.y_arff.append(y_node.shape[0])
                        print("[PreProcessing] /y" + str(y_n) + " loaded")
                except tb.exceptions.NoSuchNodeError:
                    pass
                meta_fi.close()
            else:
                raise FileNotFoundError("[Error] [PreProcessing] No meta file input")
        else:
            self.y_arff = []

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
        for col_num in (self.col['1hot'] + self.col['int'] +  # 1-hot columns
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

        # function assignment
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

        if self.meta_migrate:
            self.get_epoch = self._none
        else:
            if self.add_td is not None:  # ms + duration
                if self.io['is_epoch']:
                    self.get_epoch = lambda x, col_n: x[:, col_n] + (x[:, self.add_td] * self.td_scale)  # tested
                else:
                    self.get_epoch = lambda x, col_n: (x[:, col_n].astype('datetime64[ms]').astype('int64') / 1000
                                                       ) + (x[:, self.add_td] * self.td_scale)
            else:  # ms
                if self.io['is_epoch']:
                    self.get_epoch = lambda x, col_n: x[:, col_n]
                else:
                    self.get_epoch = lambda x, col_n: x[:, col_n].astype('datetime64[ms]').astype('int64') / 1000

    def _get_files(self, data_dir, is_csv):
        """ assign Reader for each dataset found """
        if data_dir is None:
            return []

        datasets = []
        file_count = 0

        data_path = pathlib.Path(data_dir)

        if data_path.is_dir():
            for child in data_path.iterdir():

                if pathlib.Path(child).is_file():
                    file_count += 1

                    datasets.append({
                        'name': child.stem,
                        'reader': input_reader.CsvReader(
                            str(child), label_loc=self.lbl['i'], dtypes=self.io['dtypes_in'],
                            parse_dates=self.io['dates'], read_chunk_size=self.io['read_chunk_size'],
                            delimiter=self.io['delimiter'], header=self.io['header']
                        ) if is_csv else input_reader.Hd5Reader(str(child), is_2d=True,
                                                                read_chunk_size=self.io['read_chunk_size'])
                    })

        elif data_path.is_file():
            file_count += 1

            datasets.append({
                'name': data_path.stem,
                'reader': input_reader.CsvReader(
                    data_dir, label_loc=self.lbl['i'], dtypes=self.io['dtypes_in'],
                    parse_dates=self.io['dates'], read_chunk_size=self.io['read_chunk_size'],
                    delimiter=self.io['delimiter'], header=self.io['header']
                ) if is_csv else input_reader.Hd5Reader(data_dir, is_2d=True,
                                                        read_chunk_size=self.io['read_chunk_size'])
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

                for col_num in (self.col['1hot'] + self.col['int']):  # 1-hot feature(s)
                    self.x_unq[col_num] |= set(x[:, col_num])

                for col_num in self.col['ips']:  # IPs
                    self.x_unq[self.col['ips'][0]] |= set(x[:, col_num])

                for col_num in self.val:  # Norm.
                    col = x[:, col_num]
                    self.get_norm_val(col, col_num)

                self.check_lbl_unq(y)

            instance_num -= (self.io['read_chunk_size'] - x.shape[0])
            self.instances += instance_num
            print("[PreProcessing] [metadata]", instance_num, "intance(s) iterated, time elapsed:",
                  time.strftime("%H:%M:%S", time.gmtime(time.time() - t)))

        # second iteration (obtain IPs from testset, NOTE: depends on usage, this block should be removed)
        for testset in self.testsets:
            t = time.time()
            print("[PreProcessing] [metadata] Processing >", testset['name'])

            x = None
            next_chunk = True
            instance_num = 0

            while next_chunk:
                x, _, next_chunk = testset['reader'].next()
                instance_num += self.io['read_chunk_size']

                for col_num in self.col['ips']:  # IPs
                    self.x_unq[self.col['ips'][0]] |= set(x[:, col_num])

            instance_num -= (self.io['read_chunk_size'] - x.shape[0])
            self.instances += instance_num
            print("[PreProcessing] [metadata]", instance_num, "intance(s) iterated, time elapsed:",
                  time.strftime("%H:%M:%S", time.gmtime(time.time() - t)))

        # finishing calculation of mean / output info.
        for col_num in self.val:
            if self.normalization == 'zscore':
                self.val[col_num]['v1'] /= self.instances

            elif self.normalization == 'minmax1r' or self.normalization == 'minmax2r':
                print("[PreProcessing] [metadata] Attribute", col_num, "[Min]", self.val[col_num]['v1'],
                      "[Max]", self.val[col_num]['v2'])

        # third iteration (calculation of Std Dev.)
        if self.normalization == 'zscore':
            self._calc_stddev()

        # labels map creation
        if self.lbl['i']:
            labels = []
            # NOTE: prone to error if lbl_normal is not the same length with col['lbl']
            for i, col_num in enumerate(self.lbl['i']):
                labels.append(list(self.y_unq[col_num]))
                print(labels)
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

        for col_num in self.col['t']:  # sin (day), cos (day), sin (week), sin (week)
            self.features_n += 3

            for i, item in enumerate(self.columns_map[col_num + 1:]):
                self.columns_map[i + col_num + 1] += 3

        for col_num in (self.col['1hot'] + self.col['int'] +
                        ([self.col['ips'][0]] if len(self.col['ips']) > 0 else [])):
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

        for col_num in (self.col['rm'] + self.col['ips']):  # column(s) to remove
            self.features_n -= 1
            self.columns_map[col_num] = None

            for i, item in enumerate(self.columns_map[col_num + 1:]):
                if item is not None:
                    self.columns_map[i + col_num + 1] -= 1

    def save_metadata(self, save_dir=None, name="/mappings"):

        if save_dir is None:
            save_dir = self.io['output_dir']
        save_path = save_dir + "/" + name + ".hd5"

        print("[PreProcessing] [metadata] Saving meta info. >", save_path)

        pathlib.Path(save_dir).mkdir(parents=True, exist_ok=True)
        meta_fo = tb.open_file(save_path, mode='w')

        # features' header initialization
        x_map = np.empty(self.features_n, dtype="S32")
        x_map[:] = ''
        meta_prev = {}  # previous meta data
        meta_prev_desc = {}

        if self.meta_migrate and pathlib.Path(self.io['meta_path']).is_file():
            meta_fi = tb.open_file(self.io['meta_path'], mode='r')

            meta_h5_iter = iter(meta_fi)
            next(meta_h5_iter)  # skip root node (/)

            # get data from meta file
            for node in meta_h5_iter:
                meta_prev[node._v_name] = node.read()
                meta_prev_desc[node._v_name] = node._g_gettitle()

            # save columns' name from previous meta
            for k, data in meta_prev.items():
                if k == "x":
                    for col_num, col in enumerate(self.columns_map):
                        if col is not None:
                            x_map[self.columns_map[col_num]] = data[col_num]
                else:
                    meta_fo.create_array(meta_fo.root, k, data, meta_prev_desc[k])

            meta_fi.close()

        for col, col_nums in self.col.items():
            if col == 't':
                cyclic_type = ["_sin", "_cos"]
                date_type = [" (day)", " (day)", " (month)", " (month)"]
                for i, col_num in enumerate(col_nums):
                    for r in range(4):
                        x_map[self.columns_map[col_num]+r] = col + "_" + str(i) + cyclic_type[r % 2] + date_type[r]
            elif col == '1hot':
                for i, col_num in enumerate(col_nums):
                    for j, unq in enumerate(self.x_unq[col_num]):
                        x_map[self.columns_map[col_num]+j] = col + "_" + str(i) + ": " + str(unq)
            elif col == 'rm' or col == 'ips':
                pass
            else:
                for i, col_num in enumerate(col_nums):
                    x_map[self.columns_map[col_num]] = col + "_" + str(i)

        meta_fo.create_array(meta_fo.root, "x", x_map, "Features Header")

        # IPs List
        if self.col['ips']:
            meta_fo.create_array(meta_fo.root, "ip", np.array(list(self.x_unq[self.col['ips'][0]])), "Indexed IPs")

        # Numericalized List
        for i, col_num in enumerate(self.col['int']):
            meta_fo.create_array(meta_fo.root, "int_" + str(i),
                                 np.array(list(self.x_unq[col_num])), "Numericalized Objects")

        # Labels Mapping
        for i, mapping in enumerate(self.y_map):
            y_map = np.empty(len(mapping), dtype="S32")

            for lbl, k in mapping.items():
                y_map[k] = lbl

            if i == 0:
                y_map = ["normal", "anomaly"]
            else:
                y_map[0] = "normal"

            meta_fo.create_array(meta_fo.root, "y"+str(i), y_map, "Labels Mapping")

        meta_fo.close()

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
    def _get_normalized_time(epoch):  # ms
        datetime_utc = datetime.utcfromtimestamp(epoch)

        return (datetime_utc.isoweekday() / 7), \
            ((datetime_utc.hour * 3600) + (datetime_utc.minute * 60) + datetime_utc.second +
             (datetime_utc.microsecond/1000000)) / 86400  # convert various time to (%s.%µs format)

    def transform_trainset(self):
        """ starts preprocessing of training dataset(s) """
        pathlib.Path(self.io['output_dir']).mkdir(parents=True, exist_ok=True)

        for trainset in self.trainsets:
            t = time.time()

            print("[PreProcessing] [operation] Processing trainset >", trainset['name'])

            output_name = self.io['output_dir'] + "/" + trainset['name']
            train_fo = tb.open_file(output_name + ".hd5", mode='w')

            array_x = train_fo.create_earray(train_fo.root, "x", tb.Float64Atom(shape=()),
                                             (0, self.trainsets[0]['reader'].sequence_n, self.features_n)
                                             if self.meta_migrate else (0, self.features_n), "Feature Data")

            array_t = train_fo.create_earray(train_fo.root, "t", tb.Float64Atom(shape=()),
                                             (0,) if self.meta_migrate else (0, len(self.col['t'])),
                                             "Time") if self.col['t'] or self.meta_migrate else None

            array_ip = train_fo.create_earray(train_fo.root, "ip", tb.Int64Atom(shape=()),
                                              (0, 2) if self.meta_migrate else (0, len(self.col['ips'])),
                                              "IP Addresses") if self.col['ips'] or self.meta_migrate else None

            array_ys = []
            lbl_group = train_fo.create_group(train_fo.root, "y")
            labels_n = len(self.trainsets[0]['reader'].misc[3:]) if self.meta_migrate else self.lbl_type_n
            for n in range(labels_n):
                array_ys.append(train_fo.create_earray(lbl_group, "y" + str(n), tb.Int32Atom(),
                                                       (0, self.trainsets[0]['reader'].sequence_n)
                                                       if self.meta_migrate and len(
                                                           self.trainsets[0]['reader'].ys_r[0][0]) > 1
                                                       else (0,), "Label type " + str(n)))

            # HD5/CSV Input
            if self.meta_migrate:
                array_seq = train_fo.create_earray(train_fo.root, "seq", tb.Int32Atom(),
                                                   (0,), "Dataset Sequence Length")
            else:
                array_seq = None

            # ARFF Output
            if not self.io['arff_output']:
                arffs_w = None
            else:
                # ARFF: add header and remove existing file (due to append mode writing on previous file if they exist)
                for y_n in range(labels_n):
                    with open(output_name + "_y" + str(y_n) + (
                            "_all.arff" if self.io['temporal_dim'] else "_last.arff"), 'wb') as arff_header:
                        header_contents = np.array(["@relation " + trainset['name'] + "_y" + str(y_n), ""])

                        # attributes' headers
                        if self.io['temporal_dim']:
                            for seq_n in range(trainset['reader'].sequence_n):
                                for n in range(self.features_n):
                                    header_contents = np.append(
                                        header_contents, ("@attribute 'T" + str(seq_n+1) + " " + str(n) + "' numeric"))
                            header_contents = np.append(
                                header_contents,
                                ("@attribute 'T" + str(trainset['reader'].sequence_n) +
                                 " y' {" + ','.join(str(y) for y in range(self.y_arff[y_n])) + "}")
                            )

                        else:
                            for n in range(self.features_n):
                                header_contents = np.append(header_contents, ("@attribute '" + str(n) + "' numeric"))
                            header_contents = np.append(
                                header_contents,
                                ("@attribute 'y' {" + ','.join(str(y) for y in range(self.y_arff[y_n])) + "}")
                            )

                        header_contents = np.append(header_contents, ("", "@data"))

                        np.savetxt(arff_header, header_contents[np.newaxis].T, fmt='%s')

                arffs_w = [open(output_name + "_y" + str(n) + (
                    "_all.arff" if self.io['temporal_dim'] else "_last.arff"), 'ab') for n in range(labels_n)]

            misc_shape = int(self.io['read_chunk_size'] / trainset['reader'].sequence_n) if self.meta_migrate else None

            next_chunk = True
            while next_chunk:
                x, misc, next_chunk = trainset['reader'].next()
                cur_shape = self.io['read_chunk_size'] if next_chunk else x.shape[0]
                x_new = np.zeros((cur_shape, self.features_n))

                ''' migrate labels from HD5 datasets and/or processing of labels for ARFF datasets '''
                if self.meta_migrate:
                    t_new = misc[0]
                    ip_new = misc[1]
                    array_seq.append(misc[2])

                    # initialization as arrays instead of writing to file every loop
                    if self.io['arff_output'] == 1:
                        ys_buffer_arff = [None] * labels_n
                    elif self.io['arff_output'] == 2:
                        ys_buffer_arff = [np.zeros((sum(misc[2]), 1)) for _ in range(labels_n)]
                    else:
                        ys_buffer_arff = None

                    if not next_chunk:
                        misc_shape = int(cur_shape / trainset['reader'].sequence_n)

                    for n, y_w in enumerate(array_ys):  # loop each class type
                        y_w.append(misc[3 + n])

                        if self.io['arff_output'] == 1:  # get last label and transform row to column
                            ys_buffer_arff[n] = misc[3 + n][np.arange(misc_shape), misc[2]-1][np.newaxis].T

                        elif self.io['arff_output'] == 2:
                            slice_i = 0  # pointer for saving to ys_buffer_arff
                            for i, seq in enumerate(misc[2]):  # loop each instance
                                ys_buffer_arff[n][slice_i:slice_i+seq] = misc[3 + n][i][:seq, np.newaxis]
                                slice_i += seq

                else:
                    ''' conversion of certain features and labels (time, int, ip, labels) '''
                    t_new = np.zeros((cur_shape, len(self.col['t'])))
                    ip_new = np.zeros((cur_shape, len(self.col['ips'])))
                    y_new = np.zeros(cur_shape)
                    ys_buffer_arff = []

                    # t (gmt)
                    for i, col_num in enumerate(self.col['t']):
                        epochs = self.get_epoch(x, col_num)

                        t_new[:, i] = epochs
                        day_ofweek, sec_ofday = np.vectorize(self._get_normalized_time)(epochs)  # sun, 0 - sat, 6

                        if self.normalization == 'minmax1r':
                            x_new[:, self.columns_map[col_num]] = (np.sin(2 * np.pi * sec_ofday) + 1) / 2
                            x_new[:, self.columns_map[col_num]+1] = (np.cos(2 * np.pi * sec_ofday) + 1) / 2
                            x_new[:, self.columns_map[col_num]+2] = (np.sin(2 * np.pi * day_ofweek) + 1) / 2
                            x_new[:, self.columns_map[col_num]+3] = (np.cos(2 * np.pi * day_ofweek) + 1) / 2

                        elif self.normalization == 'minmax2r':  # zscore normalization is not taken care of
                            x_new[:, self.columns_map[col_num]] = np.sin(2 * np.pi * sec_ofday)
                            x_new[:, self.columns_map[col_num]+1] = np.cos(2 * np.pi * sec_ofday)
                            x_new[:, self.columns_map[col_num]+2] = np.sin(2 * np.pi * day_ofweek)
                            x_new[:, self.columns_map[col_num]+3] = np.cos(2 * np.pi * day_ofweek)

                        elif self.normalization == 'zscore':
                            raise ValueError("Zscore normalization is not compatible with time features")

                        else:
                            raise ValueError("Values range not chosen for time features")

                        # df = pd.DataFrame()  # import pandas as pd
                        # df['sine'] = np.sin(2 * np.pi * sec_ofday)
                        # df['cosine'] = np.cos(2 * np.pi * sec_ofday)
                        # df.plot.scatter('sine', 'cosine').set_aspect('equal')
                        # plt.show()  # import matplotlib.pyplot as plt

                    # int
                    for i, col_num in enumerate(self.col['int']):
                        x_new[:, self.columns_map[col_num]] = np.vectorize(
                            self.x_map[col_num].__getitem__)(x[:, col_num])

                    # ips
                    for i, col_num in enumerate(self.col['ips']):
                        ip_new[:, i] = np.vectorize(
                            self.x_map[self.col['ips'][0]].__getitem__)(x[:, col_num])

                    # labels
                    for i, mapping in enumerate(self.y_map[:2]):  # class 2 & 3
                        y_new[:] = np.vectorize(mapping.__getitem__)(misc[:, 0])
                        array_ys[i].append(y_new)

                    for mapping in self.y_map[2:3]:  # class 5
                        y_new[:] = np.vectorize(mapping.__getitem__)(misc[:, 1])
                        array_ys[2].append(y_new)

                    for mapping in self.y_map[3:4]:  # class 9
                        y_new[:] = np.vectorize(mapping.__getitem__)(np.core.defchararray.add(
                            misc[:, 0].astype(str), misc[:, 1].astype(str)))
                        array_ys[3].append(y_new)

                ''' actual pre-processing of all features '''
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
                        try:
                            if not float(ele).is_integer():  # if ICMP code is 0, it will not be detected
                                icmp_field = ele.split('.')  # type, code
                                # order is reverse, ICMP code is shifted 8 bits to the left
                                x[i, col_num] = int(icmp_field[0]) + int(icmp_field[1]) * 256  # 8-bit code, 8-bit type
                        except ValueError:  # UNSW dataset (certain ICMP flow will result in hexadecimal port no.)
                            try:
                                x[i, col_num] = int(ele, 16)
                            except ValueError:
                                x[i, col_num] = 0

                            if x[i, col_num] > 65535:
                                x[i, col_num] = 65535

                    x_new[:, self.columns_map[col_num]:self.columns_map[col_num] + 16] = np.unpackbits(
                        x[:, col_num].astype('float32').astype('>i2').view('uint8')).reshape((cur_shape, -1))

                # other features
                for i in self.unprocessed_i:
                    x_new[:, self.columns_map[i]] = x[:, i]

                ''' write/save all features '''
                if self.meta_migrate:  # Saving 2D features from 2D-HD5
                    if self.io['arff_output']:

                        if self.io['arff_output'] == 1:  # many-to-one labeling

                            if self.io['temporal_dim']:
                                for y_n, arff_w in enumerate(arffs_w):
                                    # noinspection PyTypeChecker
                                    np.savetxt(
                                        arff_w, np.append(
                                            np.reshape(x_new, (int(cur_shape/trainset['reader'].sequence_n), -1)),
                                            (ys_buffer_arff[y_n]), axis=1),
                                        fmt="%.18e," * (self.features_n * trainset['reader'].sequence_n) + "%i")
                            else:
                                seqs = [trainset['reader'].sequence_n * i + (seq - 1) for i, seq in enumerate(misc[2])]
                                for y_n, arff_w in enumerate(arffs_w):
                                    # noinspection PyTypeChecker
                                    np.savetxt(
                                        arff_w, np.append(x_new[seqs], (ys_buffer_arff[y_n]), axis=1),
                                        fmt="%.18e," * self.features_n + "%i")

                        else:  # many-to-many labeling
                            for y_n, arff_w in enumerate(arffs_w):
                                x_buffer_arff = np.zeros((sum(misc[2]), self.features_n))
                                slice_i = 0  # index for x_buffer_arff
                                slice_j = 0  # index for x_new

                                print(x_buffer_arff.shape)

                                for i, seq in enumerate(misc[2]):  # loop each instance
                                    x_buffer_arff[slice_i:slice_i + seq] = x_new[slice_j:slice_j+seq]
                                    slice_i += seq
                                    slice_j += trainset['reader'].sequence_n

                                # noinspection PyTypeChecker
                                np.savetxt(
                                    arff_w, np.append(x_buffer_arff, (ys_buffer_arff[y_n]), axis=1),
                                    fmt="%.18e," * self.features_n + "%i")

                    array_x.append(x_new.reshape(int(cur_shape/trainset['reader'].sequence_n),
                                                 trainset['reader'].sequence_n, self.features_n))
                else:  # Saving 1D features from CSV into HD5
                    array_x.append(x_new)

                if self.col['ips'] or self.meta_migrate:
                    array_ip.append(ip_new)

                if self.col['t'] or self.meta_migrate:
                    array_t.append(t_new)

            if self.io['arff_output']:
                for arff_w in arffs_w:
                    arff_w.close()
            train_fo.close()

            print("[PreProcessing] [operation] time elapsed:",
                  time.strftime("%H:%M:%S", time.gmtime(time.time() - t)))

        return True

    def transform_testset(self):
        """ starts preprocessing of training dataset(s) """
        pathlib.Path(self.io['output_dir']).mkdir(parents=True, exist_ok=True)

        for testset in self.testsets:
            t = time.time()

            print("[PreProcessing] [operation] Processing testset >", testset['name'])

            output_name = self.io['output_dir'] + "/" + testset['name']
            test_fo = tb.open_file(output_name + ".hd5", mode='w')

            array_x = test_fo.create_earray(test_fo.root, "x", tb.Float64Atom(),
                                            (0, self.trainsets[0]['reader'].sequence_n, self.features_n)
                                            if self.meta_migrate else (0, self.features_n), "Feature Data")

            array_t = test_fo.create_earray(test_fo.root, "t", tb.Float64Atom(shape=()),
                                            (0,) if self.meta_migrate else (0, len(self.col['t'])), "Time")

            array_ip = test_fo.create_earray(test_fo.root, "ip", tb.Int64Atom(shape=()),
                                             (0, 2) if self.meta_migrate else (0, len(self.col['ips'])),
                                             "IP Addresses") if self.col['ips'] or self.meta_migrate else None

            array_ys = []
            lbl_group = test_fo.create_group(test_fo.root, "y")
            labels_n = len(self.trainsets[0]['reader'].misc[3:]) if self.meta_migrate else self.lbl_type_n
            for n in range(labels_n):
                array_ys.append(test_fo.create_earray(lbl_group, "y" + str(n), tb.Int32Atom(),
                                                      (0, self.trainsets[0]['reader'].sequence_n)
                                                      if self.meta_migrate and len(
                                                          self.trainsets[0]['reader'].ys_r[0][0]) > 1
                                                      else (0,), "Label type " + str(n)))

            # HD5/CSV Input
            if self.meta_migrate:
                array_seq = test_fo.create_earray(test_fo.root, "seq", tb.Int32Atom(),
                                                  (0,), "Dataset Sequence Length")
            else:
                array_seq = None

            # ARFF Output
            if not self.io['arff_output']:
                arffs_w = None
            else:
                # ARFF: add header and remove existing file (due to append mode writing on previous file if they exist)
                for y_n in range(labels_n):
                    with open(output_name + "_y" + str(y_n) + (
                            "_all.arff" if self.io['temporal_dim'] else "_last.arff"), 'wb') as arff_header:
                        header_contents = np.array(["@relation " + testset['name'] + "_y" + str(y_n), ""])

                        # attributes' headers
                        if self.io['temporal_dim']:
                            for seq_n in range(testset['reader'].sequence_n):
                                for n in range(self.features_n):
                                    header_contents = np.append(
                                        header_contents, ("@attribute 'T" + str(seq_n+1) + " " + str(n) + "' numeric"))
                            header_contents = np.append(
                                header_contents,
                                ("@attribute 'T" + str(testset['reader'].sequence_n) +
                                 " y' {" + ','.join(str(y) for y in range(self.y_arff[y_n])) + "}")
                            )

                        else:
                            for n in range(self.features_n):
                                header_contents = np.append(header_contents, ("@attribute '" + str(n) + "' numeric"))
                            header_contents = np.append(
                                header_contents,
                                ("@attribute 'y' {" + ','.join(str(y) for y in range(self.y_arff[y_n])) + "}")
                            )

                        header_contents = np.append(header_contents, ("", "@data"))

                        np.savetxt(arff_header, header_contents[np.newaxis].T, fmt='%s')

                arffs_w = [open(output_name + "_y" + str(n) + (
                    "_all.arff" if self.io['temporal_dim'] else "_last.arff"), 'ab') for n in range(labels_n)]

            misc_shape = int(self.io['read_chunk_size'] / testset['reader'].sequence_n) if self.meta_migrate else None

            next_chunk = True
            while next_chunk:
                x, misc, next_chunk = testset['reader'].next()
                cur_shape = self.io['read_chunk_size'] if next_chunk else x.shape[0]
                x_new = np.zeros((cur_shape, self.features_n))

                if self.meta_migrate:
                    t_new = misc[0]
                    ip_new = misc[1]
                    array_seq.append(misc[2])

                    # initialization as arrays instead of writing to file every loop
                    if self.io['arff_output'] == 1:
                        ys_buffer_arff = [None] * labels_n
                    elif self.io['arff_output'] == 2:
                        ys_buffer_arff = [np.zeros((sum(misc[2]), 1)) for _ in range(labels_n)]
                    else:
                        ys_buffer_arff = None

                    if not next_chunk:
                        misc_shape = int(cur_shape / testset['reader'].sequence_n)

                    for n, y_w in enumerate(array_ys):  # loop each class type
                        y_w.append(misc[3 + n])

                        if self.io['arff_output'] == 1:
                            ys_buffer_arff[n] = misc[3 + n][np.arange(misc_shape), misc[2]-1][np.newaxis].T

                        elif self.io['arff_output'] == 2:
                            slice_i = 0  # pointer for saving to ys_buffer_arff
                            for i, seq in enumerate(misc[2]):  # loop each instance
                                ys_buffer_arff[n][slice_i:slice_i + seq] = misc[3 + n][i][:seq, np.newaxis]
                                slice_i += seq

                else:
                    t_new = np.zeros((cur_shape, len(self.col['t'])))
                    ip_new = np.zeros((cur_shape, len(self.col['ips'])))
                    y_new = np.zeros(cur_shape)
                    ys_buffer_arff = []

                    # t
                    for i, col_num in enumerate(self.col['t']):
                        epochs = self.get_epoch(x, col_num)

                        t_new[:, i] = epochs
                        day_ofweek, sec_ofday = np.vectorize(self._get_normalized_time)(epochs)  # sun, 0 - sat, 6

                        if self.normalization == 'minmax1r':
                            x_new[:, self.columns_map[col_num] + 1] = (np.sin(2 * np.pi * sec_ofday) + 1) / 2
                            x_new[:, self.columns_map[col_num] + 2] = (np.cos(2 * np.pi * sec_ofday) + 1) / 2
                            x_new[:, self.columns_map[col_num] + 3] = (np.sin(2 * np.pi * day_ofweek) + 1) / 2
                            x_new[:, self.columns_map[col_num] + 4] = (np.cos(2 * np.pi * day_ofweek) + 1) / 2

                        elif self.normalization == 'minmax2r':  # zscore normalization is not taken care of
                            x_new[:, self.columns_map[col_num] + 1] = np.sin(2 * np.pi * sec_ofday)
                            x_new[:, self.columns_map[col_num] + 2] = np.cos(2 * np.pi * sec_ofday)
                            x_new[:, self.columns_map[col_num] + 3] = np.sin(2 * np.pi * day_ofweek)
                            x_new[:, self.columns_map[col_num] + 4] = np.cos(2 * np.pi * day_ofweek)

                        elif self.normalization == 'zscore':
                            raise ValueError("Zscore normalization is not compatible with time features")

                        else:
                            raise ValueError("Values range not chosen for time features")

                    # ips
                    for i, col_num in enumerate(self.col['ips']):
                        ip_new[:, i] = np.vectorize(
                            self.x_map[self.col['ips'][0]].__getitem__)(x[:, col_num])

                    # int
                    for i, col_num in enumerate(self.col['int']):
                        x_new[:, self.columns_map[col_num]] = np.vectorize(
                            self.x_map[col_num].__getitem__)(x[:, col_num])

                    # labels
                    for i, mapping in enumerate(self.y_map[:2]):  # 0: 2-class, 1: 3-class
                        y_new[:] = np.vectorize(mapping.__getitem__)(misc[:, 0])
                        array_ys[i].append(y_new)

                    for mapping in self.y_map[2:3]:  # 2: 5-class
                        y_new[:] = np.vectorize(mapping.__getitem__)(misc[:, 1])
                        array_ys[2].append(y_new)

                    for mapping in self.y_map[3:4]:  # 3: 9-class
                        y_new[:] = np.vectorize(mapping.__getitem__)(np.core.defchararray.add(
                            misc[:, 0].astype(str), misc[:, 1].astype(str)))
                        array_ys[3].append(y_new)

                    # # ips (when unknown IPs present, convert to -1)
                    # for col_num in self.col['ips']:
                    #     unq_i = np.vectorize(self.x_map[self.col['ips'][0]].get, otypes=[np.object])(x[:, col_num])
                    #     unq_i[unq_i == None] = -1
                    #     x_new[:, self.columns_map[col_num]] = unq_i.astype(np.int)

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
                        try:
                            if not float(ele).is_integer():
                                icmp_field = ele.split('.')
                                x[i, col_num] = int(icmp_field[0]) + int(icmp_field[1]) * 256
                        except ValueError:
                            try:
                                x[i, col_num] = int(ele, 16)
                            except ValueError:
                                x[i, col_num] = 0

                            if x[i, col_num] > 65535:
                                x[i, col_num] = 65535

                    x_new[:, self.columns_map[col_num]:self.columns_map[col_num] + 16] = np.unpackbits(
                        x[:, col_num].astype('float32').astype('>i2').view('uint8')).reshape((cur_shape, -1))

                # other features
                for i in self.unprocessed_i:
                    x_new[:, self.columns_map[i]] = x[:, i]

                if self.meta_migrate:  # Saving 2D features from 2D-HD5
                    if self.io['arff_output']:

                        if self.io['arff_output'] == 1:  # many-to-one labeling

                            if self.io['temporal_dim']:
                                for y_n, arff_w in enumerate(arffs_w):
                                    # noinspection PyTypeChecker
                                    np.savetxt(
                                        arff_w, np.append(
                                            np.reshape(x_new, (int(cur_shape / testset['reader'].sequence_n), -1)),
                                            (ys_buffer_arff[y_n]), axis=1),
                                        fmt="%.18e," * (self.features_n * testset['reader'].sequence_n) + "%i")
                            else:
                                seqs = [testset['reader'].sequence_n * i + (seq - 1) for i, seq in enumerate(misc[2])]
                                for y_n, arff_w in enumerate(arffs_w):
                                    # noinspection PyTypeChecker
                                    np.savetxt(
                                        arff_w, np.append(x_new[seqs], (ys_buffer_arff[y_n]), axis=1),
                                        fmt="%.18e," * self.features_n + "%i")

                        else:  # many-to-many labeling
                            for y_n, arff_w in enumerate(arffs_w):
                                x_buffer_arff = np.zeros((sum(misc[2]), self.features_n))
                                slice_i = 0  # index for x_buffer_arff
                                slice_j = 0  # index for x_new

                                for i, seq in enumerate(misc[2]):  # loop each instance
                                    x_buffer_arff[slice_i:slice_i + seq] = x_new[slice_j:slice_j + seq]
                                    slice_i += seq
                                    slice_j += testset['reader'].sequence_n

                                # noinspection PyTypeChecker
                                np.savetxt(
                                    arff_w, np.append(x_buffer_arff, (ys_buffer_arff[y_n]), axis=1),
                                    fmt="%.18e," * self.features_n + "%i")

                    array_x.append(x_new.reshape(int(cur_shape/testset['reader'].sequence_n),
                                                 testset['reader'].sequence_n, self.features_n))
                else:  # Saving 1D features from CSV into HD5
                    array_x.append(x_new)

                if self.col['ips'] or self.meta_migrate:
                    array_ip.append(ip_new)

                if self.col['t'] or self.meta_migrate:
                    array_t.append(t_new)

            if self.io['arff_output']:
                for arff_w in arffs_w:
                    arff_w.close()

            test_fo.close()

            print("[PreProcessing] [operation] time elapsed:",
                  time.strftime("%H:%M:%S", time.gmtime(time.time() - t)))

        return True

    def close(self):
        for trainset in self.trainsets:
            trainset['reader'].close()

        for testset in self.testsets:
            testset['reader'].close()

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
