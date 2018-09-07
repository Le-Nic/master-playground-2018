import numpy as np
import tables as tb
import pathlib
import time


class FlowSegregation(object):
    """
    NOTE: 3-class (y1) and 9-class (y3) labels should not be used when the bidirectional parameter is True. The
    expected behavior after seggregating IP Flows would be: each "dataset" has only one distinct label (many-to-one).

    # flows_buffer format
     {
        'ip_pair_1': {
            'i': [],  # indexes of flows being concatenated
            'n': 1,  # number of flows concatenated
            'ts': 0000000000000,  # time first seen (for tracking timeout)
            'te': 0000000000001,  # time last seen (for tracking time window)
            'x': [[...], [...], ...],  # concatenated features
            'y': ()  # tuples of length 2/4 labels
        },
        'ip_pair_2': { ... },
        'ip_pair_3': { ... }
     }
    """

    def __init__(self, configs, time_window=10000, time_out=60000, flow_te=True, bidirectional=True):
        self.io = configs
        self.time_window = time_window  # max. duration for aggregating different flows under the same set
        self.time_out = time_out  # max. duration the aggregated flows stayed in the buffer (ms: 1000 = 1s)
        self.flow_te = flow_te  # dataset type (te / ts)
        self.bidirectional = bidirectional  # segregate flows by allowing unordered/ordered IP pairs

        # variables initialization
        self.datasets = self._get_files(self.io['input_dir'])
        self.pairing_func = self._unordered_pairing_func if bidirectional else self._ordered_pairing_func
        self.flows_buffer = {}

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
                file_count += 1

                datasets.append({
                    'name': child.stem,
                    'path': str(child)
                }) if pathlib.Path(child).is_file() else None

        elif data_path.is_file():
            file_count += 1
            datasets.append({
                'name': data_path.stem,
                'path': data_dir
            })

        print("[FlowSegt]", file_count, "file(s) found in >", data_dir)
        return datasets

    @staticmethod
    def _unordered_pairing_func(ip_1, ip_2):
        return ip_1 * ip_2 + ((np.absolute(ip_1 - ip_2) - 1) ** 2 / 4)

    @staticmethod
    def _ordered_pairing_func(ip_1, ip_2):
        return (ip_1 + ip_2) * (ip_1 + ip_2 + 1) / 2 + ip_2

    def ip_segregate(self):
        """ starts ip segregation of dataset(s) """
        for dataset in self.datasets:
            t = time.time()
            print("[FlowSegt] Processing >", dataset['name'])

            # IO Read
            h5_r = tb.open_file(dataset['path'], mode='r')
            x_r = h5_r.get_node("/x")

            try:
                ys_r = [h5_r.get_node("/y", "y" + str(n)) for n in range(4)]
            except TypeError:
                ys_r = [h5_r.get_node("/y", "y" + str(n)) for n in range(2)]

            # IO Write
            h5_w = tb.open_file(self.io['output_dir'] + "/" + dataset['name'] + "_ipsgt.hd5", mode='w')
            x_w = h5_w.create_vlarray(h5_w.root, "x", tb.ObjectAtom(), "Feature Data (IP Segregated)")

            y_group = h5_w.create_group(h5_w.root, "y")
            ys_w = [h5_w.create_earray(y_group, "y" + str(n), tb.Int32Atom(), (0,),
                                       "Label type " + str(n) + " (IP Segregated)")
                    for n in range(len(ys_r))]

            flow_n = 0  # flow count tracker
            dataset_n = 0  # segregated dataset count tracker

            flow_total_n = len(x_r)  # total initial flows (asserting flow_total_n = flow_n)
            y_len = len(ys_r)  # total label type

            seqlen_count = {}  # sequence length repetition tracker
            dataset_maxlen = 1  # maximum sequence length tracker
            dataset_maxlen_i = 0  # index of maximum sequence length tracker

            for x, y in zip(x_r.iterrows(), zip(*[y_r.iterrows() for y_r in ys_r])):
                flow_n += 1
                time_now = x[0]

                print("flow:", flow_n, end='\r')

                # check expired flows for exporting
                expired_k = [k for k in self.flows_buffer if
                             (time_now - self.flows_buffer[k]['ts'] > self.time_out) or
                             (time_now - self.flows_buffer[k]['te']) > self.time_window]

                for k in expired_k:
                    dataset_n += 1
                    flow_ip_seg = self.flows_buffer.pop(k)

                    # track sequences length
                    try:
                        seqlen_count[flow_ip_seg['n']] += 1
                    except KeyError:
                        seqlen_count[flow_ip_seg['n']] = 1

                    if flow_ip_seg['n'] > dataset_maxlen:
                        dataset_maxlen = flow_ip_seg['n']
                        dataset_maxlen_i = dataset_n

                    # insert ip segregated dataset
                    x_w.append(flow_ip_seg['x'])
                    for n, y_w in enumerate(ys_w):
                        y_w.append([flow_ip_seg['y'][n]])

                # save new flow into buffer
                ip_pair = self.pairing_func(x[self.io['ip_1']], x[self.io['ip_2']])

                if ip_pair not in self.flows_buffer:  # add new entry
                    self.flows_buffer[ip_pair] = {'i': [flow_n], 'n': 1, 'ts': time_now, 'te': time_now, 'x': [x], 'y': y}

                else:  # append to existing entry
                    self.flows_buffer[ip_pair]['i'].append(flow_n)
                    self.flows_buffer[ip_pair]['n'] += 1
                    self.flows_buffer[ip_pair]['te'] = time_now
                    self.flows_buffer[ip_pair]['x'].append(x)

                    if self.flows_buffer[ip_pair]['y'] != y:
                        if not self.bidirectional or y_len != 4:
                            print("[Warning] Conflict labels for flow#:", self.flows_buffer[ip_pair]['i'])
                            print(self.flows_buffer[ip_pair]['y'], "<->", y)
            h5_r.close()
            h5_w.close()

            print("[FlowSegt]", flow_n, "flows processed, time elapsed:", time.time() - t)
            assert flow_total_n == flow_n, "Number of flows processed not tally, expected " + \
                                           str(flow_total_n) + " flows"

            print("[FlowSegt]", dataset_n, "datasets generated")
            print("[FlowSegt] Max. sequence length: " + str(dataset_maxlen) +
                  ", flow #" + str(dataset_maxlen_i))
            print("[FlowSegt] Sequences count:", seqlen_count)

        return True
