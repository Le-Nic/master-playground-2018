from inputhandler import input_reader
import numpy as np
import tables as tb
import pathlib
import time


class IPBatchSegregation(object):
    """ Why? To avoid flows from same IP in the same mini.batch  """

    def __init__(self, configs, batch_n=1):
        self.io = configs

        self.batch_n = batch_n
        print("[IPBatchSegt] m.Batch created:", batch_n)

        self.exec_winsegt = self._ipbatch_segt

        self.datasets = self._get_files(self.io['input_dir'])
        self.sequence_max = self.datasets[0]['reader'].sequence_n
        self.labels_len = len(self.datasets[0]['reader'].ys_r)

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

        print("[FlowSegt]", file_count, "file(s) found in >", data_dir)
        return datasets

    def _get_writers(self, h5_w):
        x_w = h5_w.create_earray(h5_w.root, "x", tb.Float64Atom(),
                                 (0, self.sequence_max, self.features_len), "Feature Data (Window Segregated)")

        t_w = h5_w.create_earray(h5_w.root, "t", tb.Float64Atom(), (0, self.sequence_max), "Time")
        ip_w = h5_w.create_earray(h5_w.root, "ip", tb.Int32Atom(), (0, self.sequence_max, 2), "IP Addresses")
        seq_w = h5_w.create_earray(h5_w.root, "seq", tb.Int32Atom(), (0,), "Dataset Sequence Length")

        ys_group = h5_w.create_group(h5_w.root, "y")
        ys_w = [h5_w.create_earray(ys_group, "y" + str(n), tb.Int32Atom(), (0, self.sequence_max),
                                   "Label type " + str(n) + " (Window Segregated)")
                for n in range(self.labels_len)]

        return x_w, t_w, ip_w, seq_w, ys_w

    def ip_batch_segregate(self):
        """ starts window segregation of dataset(s) """

        pathlib.Path(self.io['output_dir']).mkdir(parents=True, exist_ok=True)

        for dataset in self.datasets:
            time_elapsed = time.time()
            print("[IPBatchSegt] Processing >", dataset['name'])

            h5_w = tb.open_file(self.io['output_dir'] + "/" + dataset['name'] + ".hd5", mode='w')
            x_w, t_w, ip_w, seq_w, ys_w = self._get_writers(h5_w)

            flow_n = self.exec_winsegt(dataset['reader'], x_w, t_w, ip_w, seq_w, ys_w)

            print("[IPBatchSegt]", flow_n, "flows generated, time elapsed:",
                  time.strftime("%H:%M:%S", time.gmtime(time.time() - time_elapsed)))

            h5_w.close()

        return True

    def _ipbatch_segt(self, h5_r, x_w, t_w, ip_w, seq_w, ys_w):
        flow_n = 0  # keep track how many flows are generated
        chunk_size = 0  # keep track no. of chunk size so as to append it
        ready_chunk = []  # buffer for storing chunks ready to be appended
        reserves = []  # buffer for storing excees record which share the same IP addresses

        next_chunk = True
        while next_chunk:
            x, misc, next_chunk = h5_r.next()

            t = misc[0]
            ip = misc[1][0]
            seq = misc[2]
            ys = misc[3:]

            ip_k = ip[-1][0] * ip[-1][1] + ((np.absolute(ip[-1][0] - ip[-1][1]) - 1) ** 2 / 4)  # unordered

            if not any(item.get('k', None) == ip_k for item in ready_chunk):  # add if ip not found in next_chunk
                ready_chunk.append({'k': ip_k, 'x': x, 't': t, 'i': ip, 's': seq, 'y': ys})
                chunk_size += 1

                if chunk_size == self.batch_n:  # buffer is full (ready to be appended)
                    for instance in ready_chunk:
                        x_w.append([instance['x']])
                        t_w.append(instance['t'])
                        ip_w.append([instance['i']])
                        seq_w.append(instance['s'])
                        for i in range(self.labels_len):
                            ys_w[i].append(instance['y'][i])

                    flow_n += self.batch_n
                    chunk_size = 0
                    ready_chunk = []

                    for instance in reserves[:]:  # add to data from reserves to next_chunk
                        if not any(item.get('k', None) == instance['k'] for item in ready_chunk):
                            ready_chunk.append(instance)
                            reserves.remove(instance)
                            chunk_size += 1

                    print(flow_n, end='\r')
                    if chunk_size >= self.batch_n:
                        print("[IPBatchSegt] ALERT, ABNORMAL BEHAVIOR")
            else:
                reserves.append({'k': ip_k, 'x': x, 't': t, 'i': ip, 's': seq, 'y': ys})

        # append all remaining from reserve and add 0 for paddings(?)

        return flow_n

    def close(self):
        for dataset in self.datasets:
            dataset['reader'].close()
