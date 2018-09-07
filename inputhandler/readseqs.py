import pandas as pd
import numpy as np


# DEPRECIATED
class ReadSequenceData(object):
    ''' obtain the TextFileReader object and list of information for sequence reading '''

    def __init__(self, f_input, n_features, n_classes, seq_max=20, batch_size=20, f_labels=None, ele_max=10 ** 6):

        self.batch_id = 0
        self.iter_step = batch_size * seq_max  # number of instances for each iteration

        self.n_features = n_features
        self.n_classes = n_classes

        self.chunk_size = ele_max - (ele_max % self.iter_step)  # number of data instances to read simultaneously
        self.chunk_size_actual = 0  # number of actual chunk size read
        self.seq_max = seq_max  # number of time sequences considered
        self.batch_size = batch_size  # number of data instances used simultaneously in training

        # TextFileReader for features and labels file (w/o header)
        self.f_input = f_input
        self.f_labels = f_labels
        self.data_reader = pd.read_csv(f_input, iterator=True, chunksize=self.chunk_size)
        self.labels_reader = pd.read_csv(f_labels, iterator=True, chunksize=self.chunk_size,
                                         header=None) if f_labels else None

        # self.encoder = OneHotEncoder().fit([[n] for n in range(n_classes)])

        # print("-- Chunk size:", self.chunk_size)

        self._get_nextChunk()  # obtain chunk

    def _read_files(self):
        ''' read and save next data and labels chunk '''
        try:
            if self.labels_reader:
                self.chunk_data = self.data_reader.get_chunk().values
                self.chunk_labels = self.labels_reader.get_chunk().values  # seq to seq
                # self.chunk_labels = self.encoder.transform(self.labels_reader.get_chunk().values.reshape(-1,1)).toarray()
            else:
                df = self.data_reader.get_chunk().values
                self.chunk_data, self.chunk_labels = df.iloc[:, :-1], self.encoder.transform(
                    df.iloc[:, -1].reshape(-1, 1)).toarray()
            self.chunk_size_actual = len(self.chunk_labels)  # dynamic_rnn
            return True
        except Exception as e:
            # print(e)
            return False

    def _get_nextChunk(self):
        ''' if reach EOF, file is reloaded and return False '''
        self.batch_id = 0

        if self._read_files():
            return True
        else:
            self.data_reader = pd.read_csv(self.f_data, iterator=True, chunksize=self.chunk_size)
            self.labels_reader = pd.read_csv(self.f_labels, iterator=True, chunksize=self.chunk_size, header=None,
                                             dtype=np.int32) if self.f_labels else None

            self._read_files()  # no indication of whether the file is read succesfully
            return False

    def next(self):
        ''' return a batch of specified size (returning batch size * sequence length of data instance) '''
        batch_data = []  # (batch_size, seq_max, n_features)
        batch_labels = []  # (batch_size * seq_max, range(n_classes))
        batch_seq = []  # (batch_size)

        # get batch of data from current chunk
        batch_data = self.chunk_data[self.batch_id: self.batch_id + self.iter_step]
        batch_labels = self.chunk_labels[self.batch_id: self.batch_id + self.iter_step]

        current_shape = batch_labels.shape[0]

        # pad remaining rows (data which have shorter time step, occurs near EOF)
        batch_data = np.pad(batch_data, ((0, self.iter_step - current_shape), (0, 0)), 'constant', constant_values=0) \
            .reshape(-1, self.seq_max, self.n_features)  # static_rnn

        batch_labels = np.pad(batch_labels, ((0, self.iter_step - current_shape), (0, 0)), 'constant',
                              constant_values=0) \
            .reshape(-1, self.seq_max)  # seq to seq
        # .reshape(-1, self.seq_max ,self.n_classes) # for reshaping into 3D arrays (not needed for labels)

        # prepare ararys for recording number of sequences for each batch
        batch_seq = np.zeros(self.batch_size, dtype=np.int)
        n_filled = current_shape // self.seq_max

        batch_seq[:n_filled] = self.seq_max
        if n_filled < self.batch_size:
            batch_seq[n_filled] = current_shape % self.seq_max

        self.batch_id += self.iter_step

        # get next chunk if all instances are exhausted, return True if chunk exist, return False if this is the last one
        # if self.batch_id >= self.chunk_size: # static_rnn
        if self.batch_id >= self.chunk_size_actual:
            if not self._get_nextChunk():
                return batch_data, batch_labels, batch_seq, False
        return batch_data, batch_labels, batch_seq, True
