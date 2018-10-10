import numpy as np
import tables as tb
import pathlib
import time


class WindowSegregation(object):

    def __init__(self, configs, sequence_max=32, ip_segt=False, single_output=False):
        self.io = configs
        self.sequence_max = sequence_max
        self.ip_segt = ip_segt
        self.single_output = single_output

        # variables initialization
        self.datasets = self._get_files(self.io['input_dir'])
        self.labels_len = None

        if self.ip_segt:
            self.exec_winsegt = self._winsegt_ip
            print("[WinSegt] IP Segregated")
        else:
            self.exec_winsegt = self._winsegt
            print("[WinSegt] Multi-Grained")

        print("[WinSegt] Sequence Max:", self.sequence_max)

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
            self.features_len = features_len
        except ValueError:
            meta_h5 = tb.open_file(self.io['features_len'], mode='r')
            self.features_len = len(meta_h5.get_node("/x").read())

            if self.ip_segt:
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
                meta_h5 = tb.open_file(self.io['features_len'] + "_winsgt" + str(self.sequence_max), mode='w')
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

        print("[FlowSegt]", file_count, "file(s) found in >", data_dir)
        return datasets

    def _get_writers(self, h5_w):
        x_w = h5_w.create_earray(h5_w.root, "x", tb.Float64Atom(),
                                 (0, self.sequence_max, self.features_len), "Feature Data (Window Segregated)")

        t_w = h5_w.create_earray(h5_w.root, "t", tb.Float64Atom(), (0,), "Time")
        ip_w = h5_w.create_earray(h5_w.root, "ip", tb.Int32Atom(), (0, 2), "IP Addresses")
        seq_w = h5_w.create_earray(h5_w.root, "seq", tb.Int32Atom(), (0,), "Dataset Sequence Length")

        ys_group = h5_w.create_group(h5_w.root, "y")
        ys_w = [h5_w.create_earray(ys_group, "y" + str(n), tb.Int32Atom(), (0,),
                                   "Label type " + str(n) + " (Window Segregated)")
                for n in range(self.labels_len)]

        return x_w, t_w, ip_w, seq_w, ys_w

    def window_segregate(self):
        """ starts window segregation of dataset(s) """

        # IO Write
        h5_w = tb.open_file(self.io['output_dir'] + "/" + self.datasets[0]['name'] + "winsgt" +
                            str(self.sequence_max) + ".hd5", mode='w')
        x_w, t_w, ip_w, seq_w, ys_w = self._get_writers(h5_w)

        for dataset in self.datasets:
            time_elapsed = time.time()
            print("[WinSegt] Processing >", dataset['name'])

            # IO Read
            h5_r = tb.open_file(dataset['path'], mode='r')

            if not self.single_output:
                h5_w = tb.open_file(self.io['output_dir'] + "/" + dataset['name'] + "_winsgt" +
                                    str(self.sequence_max) + ".hd5", mode='w')
                x_w, t_w, ip_w, seq_w, ys_w = self._get_writers(h5_w)

            flow_n = self.exec_winsegt(h5_r, x_w, t_w, ip_w, seq_w, ys_w)

            print("[WinSegt]", flow_n, "flows processed, time elapsed:",
                  time.strftime("%H:%M:%S", time.gmtime(time.time() - time_elapsed)))

            h5_r.close()
            if not self.single_output:
                h5_w.close()

        return True

    # bidirectional ip
    def _winsegt_ip(self, h5_r, x_w, t_w, ip_w, seq_w, ys_w):

        x_r = h5_r.get_node("/x")
        t_r = h5_r.get_node("/t")
        ip_r = h5_r.get_node("/ip")

        try:
            ys_r = [h5_r.get_node("/y", "y" + str(n)) for n in range(4)]
        except tb.exceptions.NoSuchNodeError:
            ys_r = [h5_r.get_node("/y", "y" + str(n)) for n in range(2)]

        window_buffer = {}

        flow_n = 0  # flow count tracker
        flow_total_n = len(x_r)  # total initial flows (asserting flow_total_n = flow_n)

        for x, t, ip, y in zip(x_r.iterrows(), t_r.iterrows(), ip_r.iterrows(),
                               zip(*[y_r.iterrows() for y_r in ys_r])):
            flow_n += 1
            print(flow_n, end='\r')

            ip_pair = ip[0] * ip[1] + ((np.absolute(ip[0] - ip[1]) - 1) ** 2 / 4)  # unordered pairing function

            try:
                if window_buffer[ip_pair]['n'] >= self.sequence_max:  # shift existing rows up & replace the last one
                    window_buffer[ip_pair]['x'][:-1] = window_buffer[ip_pair]['x'][1:]
                    window_buffer[ip_pair]['x'][-1][:-1] = x

                    if window_buffer[ip_pair]['ips'][0] == ip[0]:
                        window_buffer[ip_pair]['x'][-1][-1] = 1
                    else:
                        window_buffer[ip_pair]['x'][-1][-1] = 0

                else:  # broadcast to n'th row
                    window_buffer[ip_pair]['x'][window_buffer[ip_pair]['n']][:-1] = x

                    if window_buffer[ip_pair]['ips'][0] == ip[0]:
                        window_buffer[ip_pair]['x'][window_buffer[ip_pair]['n']][-1] = 1
                    else:
                        window_buffer[ip_pair]['x'][window_buffer[ip_pair]['n']][-1] = 0

                    window_buffer[ip_pair]['n'] += 1

            except KeyError:
                window_buffer[ip_pair] = {
                    'ips': [ip[0], ip[1]],
                    'n': 1,
                    'x': np.zeros((self.sequence_max, self.features_len)),
                }
                window_buffer[ip_pair]['x'][0][:-1] = x
                window_buffer[ip_pair]['x'][0][-1] = 1

            # insert window-ip segregated dataset
            x_w.append([window_buffer[ip_pair]['x']])
            t_w.append(t)
            ip_w.append([ip])
            seq_w.append([window_buffer[ip_pair]['n']])
            for n, y_w in enumerate(ys_w):
                y_w.append([y[n]])

        assert flow_total_n == flow_n, "Number of flows processed not tally, expected " + \
                                       str(flow_total_n) + " flows"

        return flow_n

    def _winsegt(self, h5_r, x_w, t_w, ip_w, seq_w, ys_w):

        x_r = h5_r.get_node("/x")
        t_r = h5_r.get_node("/t")
        ip_r = h5_r.get_node("/ip")

        try:
            ys_r = [h5_r.get_node("/y", "y" + str(n)) for n in range(4)]
        except tb.exceptions.NoSuchNodeError:
            ys_r = [h5_r.get_node("/y", "y" + str(n)) for n in range(2)]

        flow_n = 0  # flow count tracker
        flow_total_n = len(x_r)  # total initial flows (asserting flow_total_n = flow_n)
        window_buffer = np.zeros((self.sequence_max, self.features_len))

        for x, t, ip, y in zip(x_r.iterrows(), t_r.iterrows(), ip_r.iterrows(),
                               zip(*[y_r.iterrows() for y_r in ys_r])):
            flow_n += 1
            print(flow_n, end='\r')

            window_buffer[:-1] = window_buffer[1:]
            window_buffer[-1:] = x

            # insert window segregated dataset
            x_w.append([window_buffer])
            t_w.append(t)
            ip_w.append([ip])
            for n, y_w in enumerate(ys_w):
                y_w.append([y[n]])

            if flow_n >= self.sequence_max:
                seq_w.append([self.sequence_max])
            else:
                seq_w.append([flow_n])

        assert flow_total_n == flow_n, "Number of flows processed not tally, expected " + \
                                       str(flow_total_n) + " flows"

        return flow_n
