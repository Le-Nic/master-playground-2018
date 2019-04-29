from inputhandler import input_reader
import numpy as np
import tables as tb
import pathlib
import time


# DEPRECIATED: update format, and add time
class TimeSegregation(object):
    """
    NOTE: beware the usage of 3-class (y1) and 9-class (y3) labels when the bidirectional parameter is True

    NOTABLE BEHAVIOR:
    I) loops are based on the flows, instead the loop off of time itself, for example: every 1s time.

    # flows_buffer format
     {
        'ip_pair_1': {
            'src': 127.0.0.1  # source IP
            'i': [],  # indexes of flows being concatenated
            'n': 1,  # number of flows concatenated
            'fs': 0000000000,  # flow time first seen for tracking timeout (time of first concatenated flow)
            'fe': 0000000001,  # flow time last seen for tracking time window (time of last concatenated flow)
            'x': [[...], [...], ...],  # concatenated features
            'y': ()  # tuples of length 2/4 labels
        },
        'ip_pair_2': { ... },
        'ip_pair_3': { ... }
     }
    """

    def __init__(self, configs, time_window=10, time_out=60, sequence_max=500,
                 bidirectional=True, single_output=True):
        self.io = configs
        self.time_window = time_window  # max. duration for aggregating different flows under the same set
        self.time_out = time_out  # max. duration the aggregated flows stayed in the buffer (1 = 1s)
        self.sequence_max = sequence_max
        self.bidirectional = bidirectional  # segregate flows by allowing unordered/ordered IP pairs
        self.single_output = single_output

        # variables initialization
        self.datasets = self._get_files(self.io['input_dir'])

        if self.bidirectional:
            self.exec_timesegt = self._timesegt_bi
            print("[TimeSegt] Bidirectional")
        else:
            self.exec_timesegt = self._timesegt_uni
            print("[TimeSegt] Unidirectional")

        print("[TimeSegt] Sequence Max:", self.sequence_max)
        print("[TimeSegt] Window:", self.time_window)
        print("[TimeSegt] Timeout:", self.time_out)

        # get labels length
        h5_r = tb.open_file(self.datasets[0]['path'], mode='r')
        try:
            self.labels_len = len([h5_r.get_node("/y", "y" + str(n)) for n in range(4)])
        except tb.exceptions.NoSuchNodeError:
            self.labels_len = len([h5_r.get_node("/y", "y" + str(n)) for n in range(2)])
        h5_r.close()

        # get features length
        try:
            features_len = int(self.io['features_len'])
            self.features_len = (features_len + 1) if bidirectional else features_len
        except ValueError:
            meta_h5 = tb.open_file(self.io['features_len'], mode='r')
            self.features_len = len(meta_h5.get_node("/x").read())

            if bidirectional:
                self.features_len += 1

                meta_h5_iter = iter(meta_h5)
                next(meta_h5_iter)  # skip root node (/)

                # get data from meta file
                meta_data = {}
                meta_desc = {}
                for node in meta_h5_iter:
                    meta_data[node._v_name] = node.read()
                    meta_desc[node._v_name] = node._g_gettitle()
                meta_h5.close()

                # create new meta file with extra feature column (is_source_ip)
                meta_output_name = str(pathlib.Path(self.io['features_len']).parent) + "/" + \
                    self.io['meta_output_name'] + "_timesgt" + str(self.sequence_max) + \
                    "t" + str(self.time_window) + str(self.time_out) + ".hd5"

                meta_h5 = tb.open_file(meta_output_name, mode='w')

                for k, data in meta_data.items():
                    if k == "x":
                        meta_h5.create_array(meta_h5.root, k, np.append(data, "is_src"), meta_desc[k])
                    else:
                        meta_h5.create_array(meta_h5.root, k, data, meta_desc[k])

                print("[WinSegt] Meta file saved >", meta_output_name)
                meta_h5.close()

    @staticmethod
    def _get_files(self, data_dir):
        """ assign Reader for each dataset found """
        if data_dir is None:
            return None

        datasets = []
        file_count = 0

        data_path = pathlib.Path(data_dir)

        if data_path.is_dir():
            for child in data_path.iterdir():
                if pathlib.Path(child).is_file():
                    datasets.append({
                        'name': child.stem,
                        'path': str(child),
                        'reader': input_reader.Hd5Reader(
                            str(child), is_2d=False, read_chunk_size=(1 if self.ip_segt else self.stride))
                    })
                    file_count += 1

        elif data_path.is_file():
            file_count += 1
            datasets.append({
                'name': data_path.stem,
                'path': data_dir,
                'reader': input_reader.Hd5Reader(
                    data_dir, is_2d=False, read_chunk_size=(1 if self.ip_segt else self.stride))
            })

        print("[TimeSegt]", file_count, "file(s) found in >", data_dir)
        return datasets

    def _get_writers(self, h5_w):
        x_w = h5_w.create_earray(h5_w.root, "x", tb.Float64Atom(),
                                 (0, self.sequence_max, self.features_len), "Feature Data (Time Segregated)")

        t_w = h5_w.create_earray(h5_w.root, "t", tb.Float64Atom(), (0, self.sequence_max), "Time")
        ip_w = h5_w.create_earray(h5_w.root, "ip", tb.Int32Atom(), (0, self.sequence_max, 2), "IP Addresses")
        seq_w = h5_w.create_earray(h5_w.root, "seq", tb.Int32Atom(), (0,), "Dataset Sequence Length")

        ys_group = h5_w.create_group(h5_w.root, "y")
        ys_w = [h5_w.create_earray(ys_group, "y" + str(n), tb.Int32Atom(), (0, self.sequence_max),
                                   "Label type " + str(n) + " (Time Segregated)")
                for n in range(self.labels_len)]

        return x_w, t_w, ip_w, seq_w, ys_w

    def time_segregate(self):
        """ starts time ip segregation of dataset(s) """

        pathlib.Path(self.io['output_dir']).mkdir(parents=True, exist_ok=True)

        # IO Write
        if self.single_output:
            h5_w = tb.open_file(self.io['output_dir'] + "/" + self.datasets[0]['name'] + "_timesgt" +
                                str(self.sequence_max) + "t" + str(self.time_window) +
                                str(self.time_out) + ".hd5", mode='w')
            x_w, t_w, ip_w, seq_w, ys_w = self._get_writers(h5_w)

        for dataset in self.datasets:
            time_elapsed = time.time()
            print("[TimeSegt] Processing >", dataset['name'])

            if not self.single_output:
                h5_w = tb.open_file(self.io['output_dir'] + "/" + dataset['name'] + "_timesgt" +
                                    str(self.sequence_max) + "t" + str(self.time_window) +
                                    str(self.time_out) + ".hd5", mode='w')
                x_w, t_w, ip_w, seq_w, ys_w = self._get_writers(h5_w)

            flow_n, dataset_n, seqlen_count = self.exec_timesegt(x_w, t_w, ip_w, seq_w, ys_w)

            print("[TimeSegt]", flow_n, "flows processed, time elapsed:",
                  time.strftime("%H:%M:%S", time.gmtime(time.time() - time_elapsed)))
            print("[TimeSegt]", dataset_n, "datasets generated")
            print("[TimeSegt] Sequences count:", seqlen_count)

            if not self.single_output:
                h5_w.close()

        if self.single_output:
            h5_w.close()

        return True

    # bidirectional ip
    def _timesegt_bi(self, h5_r, x_w, t_w, ip_w, seq_w, ys_w):

        flows_buffer = {}

        flow_n = 0  # flow count tracker
        dataset_n = 0  # segregated dataset count tracker
        seqlen_count = {}  # sequence length repetition tracker

        if h5_r.t_r.shape[0] == 0 or h5_r.ip_r.shape[0] == 0:
            print("[WinSegt] No IP Address / time")
            return 0

        next_chunk = True

        while next_chunk:
            x, misc, next_chunk = h5_r.next()

            t = misc[0]
            ip = misc[1][0]  # even stride is 1, array is in 2-dimension
            ys = misc[2:]

            # check expired flows for exporting
            expired_k = [k for k in flows_buffer if (
                    (t - flows_buffer[k]['t'][0] > self.time_window) and flows_buffer[k]['s'] > 0)]

            for k in expired_k:
                expired_seq = 0
                for past_t in flows_buffer[k]['t']:
                    if t - past_t > self.time_window:
                        expired_seq += 1
                flows_buffer[k]['i'][:-expired_seq] = flows_buffer[k]['i'][expired_seq:]
                flows_buffer[k]['t'][:-expired_seq] = flows_buffer[k]['t'][expired_seq:]
                flows_buffer[k]['x'][:-expired_seq] = flows_buffer[k]['x'][expired_seq:]
                for n, y in enumerate(ys):
                    flows_buffer[k]['y'][n][:-expired_seq] = flows_buffer[k]['y'][n][expired_seq:]
                flows_buffer[k]['s'] -= expired_seq

            # save new flow into buffer
            ip_pair = ip[0] * ip[1] + ((np.absolute(ip[0] - ip[1]) - 1) ** 2 / 4)  # unordered pairing function

            try:  # try append to existing entry
                flows_buffer[ip_pair]['x'][flows_buffer[ip_pair]['s']][:-1] = x  # trigger for new entry
                if flows_buffer[ip_pair]['i'][0][0] == ip[0]:
                    flows_buffer[ip_pair]['x'][flows_buffer[ip_pair]['s']][-1] = 1
                else:
                    flows_buffer[ip_pair]['x'][flows_buffer[ip_pair]['s']][-1] = 0

                for n, y_n in enumerate(y):
                    flows_buffer[ip_pair]['y'][n][flows_buffer[ip_pair]['n']] = y_n

                # only work for "2 labels type" datasets
                if flows_buffer[ip_pair]['y'][-1][flows_buffer[ip_pair]['n']] != y[-1]:
                    print("[Warning] Conflict labels for flow#:", flows_buffer[ip_pair]['i'])
                    print(flows_buffer[ip_pair]['y'][-1][flows_buffer[ip_pair]['n']], "<->", y[-1])

                flows_buffer[ip_pair]['fe'] = t
                flows_buffer[ip_pair]['n'] += 1

            except KeyError:  # add new entry
                flows_buffer[ip_pair] = {
                    'i': np.zeros((self.sequence_max, 2)),
                    's': 1,
                    't': np.zeros(self.sequence_max),
                    'x': np.zeros((self.sequence_max, self.features_len)),
                    'y': [np.zeros(self.sequence_max) for _ in range(self.labels_len)]
                }
                flows_buffer[ip_pair]['i'][0] = [ip[0], ip[1]]
                flows_buffer[ip_pair]['t'][0] = t
                flows_buffer[ip_pair]['x'][0][:-1] = x
                flows_buffer[ip_pair]['x'][0][-1] = 1
                for n, y in enumerate(ys):
                    flows_buffer[ip_pair]['y'][n][0] = y

            # # insert ip segregated dataset
            # x_w.append([flow_ip_seg['x']])
            # t_w.append(flow_ip_seg['fs'])
            # ip_w.append([flow_ip_seg['ips']])
            # seq_w.append([flow_ip_seg['n']])
            # for n, y_w in enumerate(ys_w):
            #     y_w.append([flow_ip_seg['y'][n]])

            flow_n += 1
            print(flow_n, end='\r')

        # export remaining flows
        for k in flows_buffer:
            dataset_n += 1

            try:
                seqlen_count[flows_buffer[k]['n']] += 1
            except KeyError:
                seqlen_count[flows_buffer[k]['n']] = 1

            # insert ip segregated dataset
            x_w.append([flows_buffer[k]['x']])
            t_w.append(flows_buffer[k]['fs'])
            ip_w.append([flows_buffer[k]['ips']])
            seq_w.append([flows_buffer[k]['n']])
            for n, y_w in enumerate(ys_w):
                y_w.append([flows_buffer[k]['y'][n]])

        return flow_n, dataset_n, seqlen_count

    # DEPRECIATED (add t in last buffer append, and y multiclass label
    def _timesegt_uni(self, h5_r, x_w, t_w, ip_w, seq_w, ys_w):

        x_r = h5_r.get_node("/x")
        t_r = h5_r.get_node("/t")
        ip_r = h5_r.get_node("/ip")
        ys_r = [h5_r.get_node("/y", "y" + str(n)) for n in range(self.labels_len)]

        flows_buffer = {}

        flow_n = 0  # flow count tracker
        dataset_n = 0  # segregated dataset count tracker
        flow_total_n = len(x_r)  # total initial flows (asserting flow_total_n = flow_n)
        seqlen_count = {}  # sequence length repetition tracker

        for x, t, ip, y in zip(x_r.iterrows(), t_r.iterrows(), ip_r.iterrows(),
                               zip(*[y_r.iterrows() for y_r in ys_r])):
            flow_n += 1
            print(flow_n, end='\r')

            # check expired flows for exporting
            expired_k = [k for k in flows_buffer if
                         ((t - flows_buffer[k]['fs'] > self.time_out or
                           t - flows_buffer[k]['fe'] > self.time_window) and
                          flows_buffer[k]['n'] > 0) or
                         flows_buffer[k]['n'] == self.sequence_max]

            for k in expired_k:
                dataset_n += 1

                flow_ip_seg = flows_buffer.pop(k)

                # track sequences length
                try:
                    seqlen_count[flow_ip_seg['n']] += 1
                except KeyError:
                    seqlen_count[flow_ip_seg['n']] = 1

                # insert ip segregated dataset
                x_w.append([flow_ip_seg['x']])
                t_w.append(flow_ip_seg['fs'])
                ip_w.append([flow_ip_seg['ips']])
                seq_w.append([flow_ip_seg['n']])
                for n, y_w in enumerate(ys_w):
                    y_w.append([flow_ip_seg['y'][n]])

            # save new flow into buffer
            ip_pair = (ip[0] + ip[1]) * (ip[0] + ip[1] + 1) / 2 + ip[1]  # ordered pairing function

            try:  # try append to existing entry
                flows_buffer[ip_pair]['x'][flows_buffer[ip_pair]['n']] = x  # trigger for new entry

                flows_buffer[ip_pair]['fe'] = t
                flows_buffer[ip_pair]['n'] += 1
                flows_buffer[ip_pair]['i'].append(flow_n)

                if flows_buffer[ip_pair]['y'] != y:
                    if not self.bidirectional or self.labels_len != 4:
                        print("[Warning] Conflict labels for flow#:", flows_buffer[ip_pair]['i'])
                        print(flows_buffer[ip_pair]['y'], "<->", y)

                        flows_buffer[ip_pair]['y'] = y

            except KeyError:  # add new entry
                flows_buffer[ip_pair] = {
                    'ips': [ip[0], ip[1]],
                    'i': [flow_n],
                    'n': 1,
                    'fs': t,
                    'fe': t,
                    'x': np.zeros((self.sequence_max, self.features_len)),
                    'y': y
                }
                flows_buffer[ip_pair]['x'][0] = x

        # export remaining flows
        for k in flows_buffer:
            dataset_n += 1

            try:
                seqlen_count[flows_buffer[k]['n']] += 1
            except KeyError:
                seqlen_count[flows_buffer[k]['n']] = 1

            # insert ip segregated dataset
            x_w.append([flows_buffer[k]['x']])
            ip_w.append([flows_buffer[k]['ips']])
            seq_w.append([flows_buffer[k]['n']])
            for n, y_w in enumerate(ys_w):
                y_w.append([flows_buffer[k]['y'][n]])

        assert flow_total_n == flow_n, "Number of flows processed not tally, expected " + \
                                       str(flow_total_n) + " flows"

        return flow_n, dataset_n, seqlen_count
