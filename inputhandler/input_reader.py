import pandas as pd
import numpy as np


class InputReader:
    def __init__(self, data_file, labels_file=None, label_loc=None,
                 dtypes=None, parse_dates=False, read_chunk_size=10**6):
        """
        label in last column: InputReader("E:/test.csv", label_loc=-1) # support multiple columns
        label in separate file: InputReader("E:/test_x.csv", labels_file="E:/test_y.csv")
        without label: InputReader("E:/test_x.csv")
        """

        self.data = None
        self.labels = None

        self.data_file = data_file
        self.labels_file = labels_file
        self.label_loc = label_loc
        self.dtypes = dtypes
        self.parse_dates = False if not parse_dates else parse_dates
        self.read_chunk_size = read_chunk_size

        if label_loc:
            self.next_input = self._get_data_lb_loc
            self._get_data_reader()
            self._get_data_lb_loc()

        elif labels_file:
            self.next_input = self._get_data_lb_file
            self._get_data_reader()
            self._get_labels_reader()
            self._get_data_lb_file()

        else:
            self.next_input = self._get_data
            self._get_data_reader()
            self._get_data()

        self.features_n = self.data.shape[1]

    def _get_data_reader(self):
        """ get reader object for data """
        self.data_reader = pd.read_csv(self.data_file, iterator=True, dtype=self.dtypes, error_bad_lines=False,
                                       header=None, parse_dates=self.parse_dates, chunksize=self.read_chunk_size,
                                       skipinitialspace=True, float_precision='round_trip')

    def _get_labels_reader(self):
        """ get reader object for labels """
        self.labels_reader = pd.read_csv(self.labels_file, iterator=True, dtype=np.object, error_bad_lines=False,
                                         header=None, chunksize=self.read_chunk_size,
                                         skipinitialspace=True, float_precision='round_trip') \
            if self.labels_file else None

    def _get_data_lb_loc(self):
        """ return chunk from reader (when labels are stored in the same data set) """
        try:
            chunk = self.data_reader.get_chunk().values
            self.data = chunk[:, :self.label_loc[0]]
            self.labels = chunk[:, self.label_loc]
            return True
        except StopIteration:
            self._get_data_reader()
            chunk = self.data_reader.get_chunk().values
            self.data = chunk[:, :self.label_loc[0]]
            self.labels = chunk[:, self.label_loc]
            return False

    def _get_data_lb_file(self):
        """ return chunk from reader (when labels are stored in separate file) """
        try:
            self.data = self.data_reader.get_chunk().values
            self.labels = self.labels_reader.get_chunk().values
            return True
        except StopIteration:
            self._get_data_reader()
            self._get_labels_reader()
            self.data = self.data_reader.get_chunk().values
            self.labels = self.labels_reader.get_chunk().values
            return False

    def _get_data(self):
        """ return data chunk from reader """
        try:
            self.data = self.data_reader.get_chunk().values
            return True
        except StopIteration:
            self._get_data_reader()
            self.data = self.data_reader.get_chunk().values
            return False

    def next(self):
        data = self.data
        labels = self.labels
        return data, labels, self.next_input()
