import pandas as pd
import numpy as np
import tables as tb


class CsvReader:
    def __init__(self, data_file, labels_file=None, label_loc=None,
                 dtypes=None, parse_dates=False, read_chunk_size=10**6, delimiter=',', header=None):
        """
        label in last column: InputReader("E:/test.csv", label_loc=-1) # support multiple columns
        label in separate file: InputReader("E:/test_x.csv", labels_file="E:/test_y.csv")
        without label: InputReader("E:/test_x.csv")

        # Example usage
        z = True
        while z:
            x, y, z = testInput.next()
            print("data:", x)
            print("labels:", y)
        """

        self.data = None
        self.labels = None

        self.data_file = data_file
        self.labels_file = labels_file
        self.label_loc = label_loc
        self.dtypes = dtypes
        self.parse_dates = False if not parse_dates else parse_dates
        self.read_chunk_size = read_chunk_size
        self.sep = delimiter
        self.header = 0 if header is True else None

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
        self.data_reader = pd.read_csv(self.data_file, sep=self.sep, iterator=True, dtype=self.dtypes,
                                       error_bad_lines=True, header=self.header, parse_dates=self.parse_dates,
                                       chunksize=self.read_chunk_size, skipinitialspace=True,
                                       float_precision='round_trip')

    def _get_labels_reader(self):
        """ get reader object for labels """
        self.labels_reader = pd.read_csv(self.labels_file, sep=self.sep, iterator=True, dtype=np.object,
                                         error_bad_lines=True, header=self.header, chunksize=self.read_chunk_size,
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


# Only for "2D" datasets: winsgt, ipsgt
class Hd5Reader:
    def __init__(self, data_file, is_2d=False, read_chunk_size=10**6):

        self.i = 0
        self.read_chunk_size = read_chunk_size
        self.is_2d = is_2d

        self.h5_r = tb.open_file(data_file, mode='r')
        self.x_r = self.h5_r.get_node('/x')

        try:
            self.t_r = self.h5_r.get_node('/t')
            self.ip_r = self.h5_r.get_node('/ip')
            self._get_data = self._get_2d_data if is_2d else self._get_1d_data

        except tb.exceptions.NoSuchNodeError:
            self.t_r = None
            self.ip_r = None
            self._get_data = self._get_basic_data

        self.seq_r = self.h5_r.get_node('/seq') if is_2d else None

        try:
            self.ys_r = [self.h5_r.get_node("/y", "y" + str(n)) for n in range(4)]
        except tb.exceptions.NoSuchNodeError:
            self.ys_r = [self.h5_r.get_node("/y", "y" + str(n)) for n in range(2)]

        self._get_data()

        self.features_n = self.data.shape[2] if is_2d else self.data.shape[1]
        self.sequence_n = self.data.shape[1] if is_2d else self.data.shape[0]

    def _get_basic_data(self):
        """ return data chunk from reader """
        j = self.read_chunk_size + self.i

        self.data = self.x_r[self.i: j]
        self.misc = (None,
                     None,) + tuple(
            y_r[self.i: j] for y_r in self.ys_r)

        if len(self.data):
            self.i += self.read_chunk_size
            return True

        else:
            self.i = 0
            self._get_data()
            return False

    def _get_1d_data(self):
        """ return data chunk from reader """
        j = self.read_chunk_size + self.i

        self.data = self.x_r[self.i: j]
        self.misc = (self.t_r[self.i: j],
                     self.ip_r[self.i: j],) + tuple(
            y_r[self.i: j] for y_r in self.ys_r)

        if len(self.data):
            self.i += self.read_chunk_size
            return True

        else:
            self.i = 0
            self._get_data()
            return False

    def _get_2d_data(self):
        """ return data chunk from reader """
        j = self.read_chunk_size + self.i

        self.data = self.x_r[self.i: j]
        self.misc = (self.t_r[self.i: j],
                     self.ip_r[self.i: j],
                     self.seq_r[self.i: j],) + tuple(
            y_r[self.i: j] for y_r in self.ys_r)

        if len(self.data):
            self.i += self.read_chunk_size
            return True

        else:
            self.i = 0
            self._get_data()
            return False

    def next(self):
        data = self.data.reshape(-1, self.features_n)
        misc = self.misc
        return data, misc, self._get_data()

    def close(self):
        self.h5_r.close()

