from inputhandler import input_reader
import numpy as np
import tables as tb
import pathlib
import time


class HierarchicalSegregation(object):
    """ Rearrange data into 4D for hierarchical model """

    def __init__(self, configs, host_sequence=None):
        self.io = configs

        self.exec_winsegt = self._hierc_segt

        self.datasets = self._get_files(self.io['input_dir'])
        self.netw_sequence = self.datasets[0]['reader'].sequence_n
        self.labels_len = len(self.datasets[0]['reader'].ys_r)

        self.host_sequence = self.netw_sequence if host_sequence is None else host_sequence
        print("[HiercSegt] Network Sequence:", self.netw_sequence)
        print("[HiercSegt] Host Sequence:", self.host_sequence)

        # get features length
        try:
            features_len = int(self.io['features_len'])
            self.features_len = features_len
        except ValueError:
            meta_h5 = tb.open_file(self.io['features_len'], mode='r')
            self.features_len = len(meta_h5.get_node("/x").read())

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
                        'path': str(child),
                        'reader': input_reader.Hd5Reader(str(child), is_2d=True, read_chunk_size=1)
                    })
                    file_count += 1

        elif data_path.is_file():
            file_count += 1
            datasets.append({
                'name': data_path.stem,
                'path': data_dir,
                'reader': input_reader.Hd5Reader(data_dir, is_2d=True, read_chunk_size=1)
            })

        print("[HiercSegt]", file_count, "file(s) found in >", data_dir)
        return datasets

    def _get_writers(self, h5_w):
        x_w = h5_w.create_earray(
            h5_w.root, "x", tb.Float64Atom(),
            (0, self.netw_sequence, self.host_sequence, self.features_len), "Feature Data (Window Segregated)"
        )
        t_w = h5_w.create_earray(
            h5_w.root, "t", tb.Float64Atom(),
            (0, self.netw_sequence), "Time"
        )
        ip_w = h5_w.create_earray(
            h5_w.root, "ip", tb.Int32Atom(),
            (0, self.netw_sequence, 2), "IP Addresses"
        )
        netw_seq_w = h5_w.create_earray(
            h5_w.root, "netw_seq", tb.Int32Atom(),
            (0,), "Network-level Sequence Length"
        )
        host_seq_w = h5_w.create_earray(
            h5_w.root, "host_seq", tb.Int32Atom(),
            (0, self.netw_sequence), "Host-level Sequence Length"
        )

        ys_group = h5_w.create_group(h5_w.root, "y")
        ys_w = [h5_w.create_earray(ys_group, "y" + str(n), tb.Int32Atom(), (0, self.netw_sequence),
                                   "Label type " + str(n) + " (Window Segregated)")
                for n in range(self.labels_len)]

        return x_w, t_w, ip_w, netw_seq_w, host_seq_w, ys_w

    def hierc_segregate(self):
        """ starts window segregation of dataset(s) """

        pathlib.Path(self.io['output_dir']).mkdir(parents=True, exist_ok=True)

        for dataset in self.datasets:
            time_elapsed = time.time()
            print("[HiercSegt] Processing >", dataset['name'])

            h5_w = tb.open_file(self.io['output_dir'] + "/" + dataset['name'] + ".hd5", mode='w')
            x_w, t_w, ip_w, netw_seq_w, host_seq_w, ys_w = self._get_writers(h5_w)

            flow_n = self.exec_winsegt(dataset['reader'], x_w, t_w, ip_w, netw_seq_w, host_seq_w, ys_w)

            print("[HiercSegt]", flow_n, "flows generated, time elapsed:",
                  time.strftime("%H:%M:%S", time.gmtime(time.time() - time_elapsed)))

            h5_w.close()

        return True

    def _hierc_segt(self, h5_r, x_w, t_w, ip_w, netw_seq_w, host_seq_w, ys_w):
        flow_n = 0  # keep track how many flows are generated
        ip_flows = {}

        next_chunk = True
        while next_chunk:
            x, misc, next_chunk = h5_r.next()

            t = misc[0]
            ip = misc[1][0]
            netw_seq = misc[2][0]
            ys = misc[3:]

            x_new = np.zeros((self.netw_sequence, self.host_sequence, self.features_len))
            host_seq = np.zeros(self.netw_sequence)

            for s in range(netw_seq):
                ip_k = ip[s][0] * ip[s][1] + ((np.absolute(ip[s][0] - ip[s][1]) - 1) ** 2 / 4)  # unordered

                try:  # update host-level flows
                    if ip_flows[ip_k]['s'] >= self.host_sequence:
                        ip_flows[ip_k]['x'][:-1] = ip_flows[ip_k]['x'][1:]
                        ip_flows[ip_k]['x'][-1] = x[s]
                    else:
                        ip_flows[ip_k]['x'][ip_flows[ip_k]['s']] = x[s]
                        ip_flows[ip_k]['s'] += 1
                except KeyError:
                    ip_flows[ip_k] = {
                        'x': np.zeros((self.host_sequence, self.features_len)),
                        's': 1
                    }
                    ip_flows[ip_k]['x'][0] = x[s]

                x_new[s, :] = ip_flows[ip_k]['x']
                host_seq[s] = ip_flows[ip_k]['s']

            x_w.append([x_new])
            t_w.append(t)
            ip_w.append([ip])
            netw_seq_w.append([netw_seq])
            host_seq_w.append([host_seq])
            for i in range(self.labels_len):
                ys_w[i].append(ys[i])

            flow_n += 1
            print(flow_n, end='\r')

        return flow_n

    def close(self):
        for dataset in self.datasets:
            dataset['reader'].close()
