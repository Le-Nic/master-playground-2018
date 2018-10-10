import numpy as np
import tables as tb
import pathlib
import time


class IpSegregation(object):
    """
    NOTE: 3-class (y1) and 9-class (y3) labels should not be used when the bidirectional parameter is True. The
    expected behavior after seggregating IP Flows would be: each "dataset" has only one distinct label (many-to-one).

    NOTABLE BEHAVIOR:
    I) loops are based on the flows, instead the loop of, for example: every 1s time.

    # flows_buffer format
     {
        'ip_pair_1': {
            'src': 127.0.0.1  # source IP
            'i': [],  # indexes of flows being concatenated
            'n': 1,  # number of flows concatenated
            'fs': 0000000000000,  # flow time first seen for tracking timeout (time of first concatenated flow)
            'fe': 0000000000001,  # flow time last seen for tracking time window (time of last concatenated flow)
            'x': [[...], [...], ...],  # concatenated features
            'y': ()  # tuples of length 2/4 labels
        },
        'ip_pair_2': { ... },
        'ip_pair_3': { ... }
     }
    """

    def __init__(self, configs, time_window=10000, time_out=60000, sequence_max=500, bidirectional=True):
        self.io = configs
        self.time_window = time_window  # max. duration for aggregating different flows under the same set
        self.time_out = time_out  # max. duration the aggregated flows stayed in the buffer (ms: 1000 = 1s)
        self.sequence_max = sequence_max
        self.bidirectional = bidirectional  # segregate flows by allowing unordered/ordered IP pairs

        # variables initialization
        self.datasets = self._get_files(self.io['input_dir'])

        if self.bidirectional:
            self.exec_ipsegt = self._ipsegt_bi
            print("[IPSegt] Bidirectional")
        else:
            self.exec_ipsegt = self._ipsegt_uni
            print("[IPSegt] Unidirectional")

        print("[IPSegt] Sequence Max:", self.sequence_max)
        print("[IPSegt] Window:", self.time_window)
        print("[IPSegt] Timeout:", self.time_out)

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
                meta_h5 = tb.open_file(self.io['features_len'] + "_ipsgt" + str(self.sequence_max), mode='w')
                for k, data in meta_data.items():
                    if k == "x":
                        meta_h5.create_array(meta_h5.root, k, np.append(data, "is_src"), meta_desc[k])
                    else:
                        meta_h5.create_array(meta_h5.root, k, data, meta_desc[k])
                meta_h5.close()

    @staticmethod
    def _get_files(data_dir):
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
                        'path': str(child)
                    })
                    file_count += 1

        elif data_path.is_file():
            file_count += 1
            datasets.append({
                'name': data_path.stem,
                'path': data_dir
            })

        print("[IPSegt]", file_count, "file(s) found in >", data_dir)
        return datasets

    def ip_segregate(self):
        """ starts ip segregation of dataset(s) """
        for dataset in self.datasets:
            t = time.time()
            print("[IPSegt] Processing >", dataset['name'])

            # IO Read & Write
            h5_r = tb.open_file(dataset['path'], mode='r')
            h5_w = tb.open_file(self.io['output_dir'] + "/" + dataset['name'] + "_ipsgt" +
                                str(self.sequence_max) + ".hd5", mode='w')

            flow_n, dataset_n, seqlen_count = self.exec_ipsegt(h5_r, h5_w)

            print("[IPSegt]", flow_n, "flows processed, time elapsed:",
                  time.strftime("%H:%M:%S", time.gmtime(time.time() - t)))

            print("[IPSegt]", dataset_n, "datasets generated")
            print("[IPSegt] Sequences count:", seqlen_count)

            h5_r.close()
            h5_w.close()

        return True

    def _ipsegt_bi(self, h5_r, h5_w):

        x_r = h5_r.get_node("/x")
        t_r = h5_r.get_node("/t")
        ip_r = h5_r.get_node("/ip")

        try:
            ys_r = [h5_r.get_node("/y", "y" + str(n)) for n in range(4)]
        except tb.exceptions.NoSuchNodeError:
            ys_r = [h5_r.get_node("/y", "y" + str(n)) for n in range(2)]

        x_w = h5_w.create_earray(h5_w.root, "x", tb.Float64Atom(),
                                 (0, self.sequence_max, self.features_len), "Feature Data (IP Segregated)")
        # x_w = h5_w.create_vlarray(h5_w.root, "x", tb.ObjectAtom(), "Feature Data (IP Segregated)")  # ragged

        ip_w = h5_w.create_earray(h5_w.root, "ip", tb.Int32Atom(), (0, 2), "IP Addresses")
        seq_w = h5_w.create_earray(h5_w.root, "seq", tb.Int32Atom(), (0,), "Dataset Sequence Length")

        y_group = h5_w.create_group(h5_w.root, "y")
        ys_w = [h5_w.create_earray(y_group, "y" + str(n), tb.Int32Atom(), (0,),
                                   "Label type " + str(n) + " (IP Segregated)")
                for n in range(len(ys_r))]

        flows_buffer = {}

        flow_n = 0  # flow count tracker
        dataset_n = 0  # segregated dataset count tracker

        flow_total_n = len(x_r)  # total initial flows (asserting flow_total_n = flow_n)
        y_len = len(ys_r)  # total label type

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
                ip_w.append([flow_ip_seg['ips']])
                seq_w.append([flow_ip_seg['n']])
                for n, y_w in enumerate(ys_w):
                    y_w.append([flow_ip_seg['y'][n]])

            # save new flow into buffer
            ip_pair = ip[0] * ip[1] + ((np.absolute(ip[0] - ip[1]) - 1) ** 2 / 4)  # unordered pairing function

            try:  # try append to existing entry
                flows_buffer[ip_pair]['x'][flows_buffer[ip_pair]['n']][:-1] = x  # trigger for new entry
                if flows_buffer[ip_pair]['ips'][0] == ip[0]:
                    flows_buffer[ip_pair]['x'][flows_buffer[ip_pair]['n']][-1] = 1
                else:
                    flows_buffer[ip_pair]['x'][flows_buffer[ip_pair]['n']][-1] = 0

                flows_buffer[ip_pair]['fe'] = t
                flows_buffer[ip_pair]['n'] += 1
                flows_buffer[ip_pair]['i'].append(flow_n)

                if flows_buffer[ip_pair]['y'] != y:
                    if not self.bidirectional or y_len != 4:
                        print("[Warning] Conflict labels for flow#:", flows_buffer[ip_pair]['i'])
                        print(flows_buffer[ip_pair]['y'], "<->", y)

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
                flows_buffer[ip_pair]['x'][0][:-1] = x
                flows_buffer[ip_pair]['x'][0][-1] = 1

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

    def _ipsegt_uni(self, h5_r, h5_w):

        x_r = h5_r.get_node("/x")
        t_r = h5_r.get_node("/t")
        ip_r = h5_r.get_node("/ip")

        try:
            ys_r = [h5_r.get_node("/y", "y" + str(n)) for n in range(4)]
        except tb.exceptions.NoSuchNodeError:
            ys_r = [h5_r.get_node("/y", "y" + str(n)) for n in range(2)]

        x_w = h5_w.create_earray(h5_w.root, "x", tb.Float64Atom(),
                                 (0, self.sequence_max, self.features_len), "Feature Data (IP Segregated)")
        # x_w = h5_w.create_vlarray(h5_w.root, "x", tb.ObjectAtom(), "Feature Data (IP Segregated)")  # ragged

        ip_w = h5_w.create_earray(h5_w.root, "ip", tb.Int32Atom(), (0, 2), "IP Addresses")
        seq_w = h5_w.create_earray(h5_w.root, "seq", tb.Int32Atom(), (0,), "Dataset Sequence Length")

        y_group = h5_w.create_group(h5_w.root, "y")
        ys_w = [h5_w.create_earray(y_group, "y" + str(n), tb.Int32Atom(), (0,),
                                   "Label type " + str(n) + " (IP Segregated)")
                for n in range(len(ys_r))]

        flows_buffer = {}

        flow_n = 0  # flow count tracker
        dataset_n = 0  # segregated dataset count tracker

        flow_total_n = len(x_r)  # total initial flows (asserting flow_total_n = flow_n)
        y_len = len(ys_r)  # total label type

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
                    if not self.bidirectional or y_len != 4:
                        print("[Warning] Conflict labels for flow#:", flows_buffer[ip_pair]['i'])
                        print(flows_buffer[ip_pair]['y'], "<->", y)

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
