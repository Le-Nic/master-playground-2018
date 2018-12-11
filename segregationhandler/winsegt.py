from inputhandler import input_reader
import numpy as np
import tables as tb
import pathlib
import time


class WindowSegregation(object):

    def __init__(self, configs, sequence_max=32, ip_segt=False, stride=1,
                 single_output=False, const_sequence=False):
        self.io = configs
        self.sequence_max = sequence_max
        self.ip_segt = ip_segt
        self.single_output = single_output
        self.const_sequence = const_sequence

        # variables initialization
        if self.ip_segt:
            self.stride = 1  # strides are only used when ip_segt is set to False
            self.exec_winsegt = self._winsegt_ip
            print("[WinSegt] IP Segregated")
        else:
            self.stride = stride  # stride=1 : "multi-grained", stride=sequence_max : "normal"
            self.exec_winsegt = self._winsegt
            print("[WinSegt] Stride:", self.stride)

        print("[WinSegt] Window Size:", self.sequence_max)

        self.datasets = self._get_files(self.io['input_dir'])
        self.labels_len = len(self.datasets[0]['reader'].ys_r)

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
                meta_output_name = str(pathlib.Path(self.io['features_len']).parent) + "/" + \
                    self.io['meta_output_name'] + "_winsgt" + str(self.sequence_max) + \
                    "s" + str(self.stride) + ("_ip.hd5" if self.ip_segt else ".hd5")

                meta_h5 = tb.open_file(meta_output_name, mode='w')

                for k, data in meta_data.items():
                    if k == "x":
                        meta_h5.create_array(meta_h5.root, k, np.append(data, "is_src"), meta_desc[k])
                    else:
                        meta_h5.create_array(meta_h5.root, k, data, meta_desc[k])

                print("[WinSegt] Meta file saved >", meta_output_name)
            meta_h5.close()

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
                        'reader': input_reader.Hd5Reader(str(child), is_2d=False, read_chunk_size=self.stride)
                    })
                    file_count += 1

        elif data_path.is_file():
            file_count += 1
            datasets.append({
                'name': data_path.stem,
                'path': data_dir,
                'reader': input_reader.Hd5Reader(data_dir, is_2d=False, read_chunk_size=self.stride)
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
        ys_w = [h5_w.create_earray(ys_group, "y" + str(n), tb.Int32Atom(), (0, self.sequence_max),
                                   "Label type " + str(n) + " (Window Segregated)")
                for n in range(self.labels_len)]

        return x_w, t_w, ip_w, seq_w, ys_w

    def window_segregate(self):
        """ starts window segregation of dataset(s) """

        pathlib.Path(self.io['output_dir']).mkdir(parents=True, exist_ok=True)

        # IO Write
        if self.single_output:
            h5_w = tb.open_file(self.io['output_dir'] + "/" + self.datasets[0]['name'] + "_winsgt" +
                                str(self.sequence_max) + "s" + str(self.stride) +
                                ("_ip.hd5" if self.ip_segt else ".hd5"), mode='w')
            x_w, t_w, ip_w, seq_w, ys_w = self._get_writers(h5_w)

        for dataset in self.datasets:
            time_elapsed = time.time()
            print("[WinSegt] Processing >", dataset['name'])

            if not self.single_output:
                h5_w = tb.open_file(self.io['output_dir'] + "/" + dataset['name'] + "_winsgt" +
                                    str(self.sequence_max) + "s" + str(self.stride) +
                                    ("_ip.hd5" if self.ip_segt else ".hd5"), mode='w')
                x_w, t_w, ip_w, seq_w, ys_w = self._get_writers(h5_w)

            flow_n = self.exec_winsegt(dataset['reader'], x_w, t_w, ip_w, seq_w, ys_w)

            print("[WinSegt]", flow_n, "flows generated, time elapsed:",
                  time.strftime("%H:%M:%S", time.gmtime(time.time() - time_elapsed)))

            # h5_r.close()
            if not self.single_output:
                h5_w.close()

        if self.single_output:
            h5_w.close()

        return True

    def close(self):
        for dataset in self.datasets:
            dataset['reader'].close()

    # bidirectional ip
    # def _winsegt_ip(self, h5_r, x_w, t_w, ip_w, seq_w, ys_w):
    #
    #     window_buffer = {}
    #
    #     flow_n = 0  # flow count tracker
    #     flow_total_n = h5_r.x_r.shape[0]  # total initial flows (asserting flow_total_n = flow_n)
    #
    #     next_chunk = True
    #     while next_chunk:
    #         x, misc, next_chunk = h5_r.next()
    #
    #         t = misc[0][0]
    #         ip = misc[1][0]
    #         ys = misc[2:]
    #
    #         ip_pair = ip[0] * ip[1] + ((np.absolute(ip[0] - ip[1]) - 1) ** 2 / 4)  # unordered pairing function
    #
    #         try:
    #             # shift existing rows up & replace the last one if the the chunk is full
    #             # if not, place the new data on top of the chunk starting from index 0
    #             if window_buffer[ip_pair]['n'] >= self.sequence_max:
    #                 window_buffer[ip_pair]['x'][:-1] = window_buffer[ip_pair]['x'][1:]
    #                 window_buffer[ip_pair]['x'][-1][:-1] = x
    #
    #                 if window_buffer[ip_pair]['ips'][0] == ip[0]:
    #                     window_buffer[ip_pair]['x'][-1][-1] = 1
    #                 else:
    #                     window_buffer[ip_pair]['x'][-1][-1] = 0
    #
    #                 for n, y in enumerate(ys):
    #                     window_buffer[ip_pair]['ys'][n][:-1] = window_buffer[ip_pair]['ys'][n][1:]
    #                     window_buffer[ip_pair]['ys'][n][-1] = y[0]
    #
    #             else:  # broadcast to n'th row
    #                 window_buffer[ip_pair]['x'][window_buffer[ip_pair]['n']][:-1] = x
    #
    #                 if window_buffer[ip_pair]['ips'][0] == ip[0]:
    #                     window_buffer[ip_pair]['x'][window_buffer[ip_pair]['n']][-1] = 1
    #                 else:
    #                     window_buffer[ip_pair]['x'][window_buffer[ip_pair]['n']][-1] = 0
    #
    #                 for n, y in enumerate(ys):
    #                     window_buffer[ip_pair]['ys'][n][window_buffer[ip_pair]['n']] = y[0]
    #
    #                 window_buffer[ip_pair]['n'] += 1
    #
    #         except KeyError:
    #             window_buffer[ip_pair] = {
    #                 'ips': [ip[0], ip[1]],
    #                 'n': 1,
    #                 'x': np.zeros((self.sequence_max, self.features_len)),
    #                 'ys': [np.zeros(self.sequence_max) for _ in range(self.labels_len)]
    #             }
    #             window_buffer[ip_pair]['x'][0][:-1] = x
    #             window_buffer[ip_pair]['x'][0][-1] = 1
    #             for n, y in enumerate(ys):
    #                 window_buffer[ip_pair]['ys'][n][0] = y[0]
    #
    #         # insert window-ip segregated dataset
    #         if not self.const_sequence or (
    #                 self.const_sequence and window_buffer[ip_pair]['n'] == self.sequence_max
    #         ):
    #             x_w.append([window_buffer[ip_pair]['x']])
    #             t_w.append(t)
    #             ip_w.append([ip])
    #             seq_w.append([window_buffer[ip_pair]['n']])
    #             for n, y_w in enumerate(ys_w):
    #                 y_w.append([window_buffer[ip_pair]['ys'][n]])
    #
    #             flow_n += 1
    #             print(flow_n, end='\r')
    #
    #     if self.const_sequence:
    #         print("[WinSegt]", flow_total_n, "flows processed")
    #     else:
    #         assert flow_total_n == flow_n, "Number of flows processed not tally, expected " + \
    #                                    str(flow_total_n) + " flows"
    #
    #     return flow_n

    def _winsegt_ip(self, h5_r, x_w, t_w, ip_w, seq_w, ys_w):

        window_buffer = {}

        flow_n = 0  # flow count tracker

        if h5_r.t_r.shape[0] == 0 or h5_r.ip_r.shape[0] == 0:
            print("[WinSegt] No IP Address / time")
            return 0

        next_chunk = True

        while next_chunk:
            x, misc, next_chunk = h5_r.next()

            t = misc[0]
            ip = misc[1]
            ys = misc[2:]

            ip_pair = ip[0] * ip[1] + ((np.absolute(ip[0] - ip[1]) - 1) ** 2 / 4)  # unordered pairing function

            try:
                # shift existing rows up & replace the last one if the the chunk is full
                # if not, place the new data on top of the chunk starting from index 0
                if window_buffer[ip_pair]['n'] >= self.sequence_max:
                    window_buffer[ip_pair]['x'][:-1] = window_buffer[ip_pair]['x'][1:]
                    window_buffer[ip_pair]['x'][-1][:-1] = x

                    if window_buffer[ip_pair]['ips'][0] == ip[0]:
                        window_buffer[ip_pair]['x'][-1][-1] = 1
                    else:
                        window_buffer[ip_pair]['x'][-1][-1] = 0

                    for n, y in enumerate(ys):
                        window_buffer[ip_pair]['ys'][n][:-1] = window_buffer[ip_pair]['ys'][n][1:]
                        window_buffer[ip_pair]['ys'][n][-1] = y[0]

                else:  # broadcast to n'th row
                    window_buffer[ip_pair]['x'][window_buffer[ip_pair]['n']][:-1] = x

                    if window_buffer[ip_pair]['ips'][0] == ip[0]:
                        window_buffer[ip_pair]['x'][window_buffer[ip_pair]['n']][-1] = 1
                    else:
                        window_buffer[ip_pair]['x'][window_buffer[ip_pair]['n']][-1] = 0

                    for n, y in enumerate(ys):
                        window_buffer[ip_pair]['ys'][n][window_buffer[ip_pair]['n']] = y[0]

                    window_buffer[ip_pair]['n'] += 1

            except KeyError:
                window_buffer[ip_pair] = {
                    'ips': [ip[0], ip[1]],
                    'n': 1,
                    'x': np.zeros((self.sequence_max, self.features_len)),
                    'ys': [np.zeros(self.sequence_max) for _ in range(self.labels_len)]
                }
                window_buffer[ip_pair]['x'][0][:-1] = x
                window_buffer[ip_pair]['x'][0][-1] = 1
                for n, y in enumerate(ys):
                    window_buffer[ip_pair]['ys'][n][0] = y[0]

            # insert window-ip segregated dataset
            if not self.const_sequence or (
                    self.const_sequence and window_buffer[ip_pair]['n'] == self.sequence_max
            ):
                x_w.append([window_buffer[ip_pair]['x']])
                t_w.append(t)
                ip_w.append([ip])
                seq_w.append([window_buffer[ip_pair]['n']])
                for n, y_w in enumerate(ys_w):
                    y_w.append([window_buffer[ip_pair]['ys'][n]])

                flow_n += 1
                print(flow_n, end='\r')

        return flow_n

    def _winsegt(self, h5_r, x_w, t_w, ip_w, seq_w, ys_w):

        flow_n = 0  # flow count tracker
        flow_total_n = h5_r.x_r.shape[0]  # total initial flows (asserting flow_total_n = flow_n)
        x_buffer = np.zeros((self.sequence_max, self.features_len))
        ys_buffer = [np.zeros(self.sequence_max) for _ in range(self.labels_len)]
        stride = self.stride

        if h5_r.t_r.shape[0] == 0 or h5_r.ip_r.shape[0] == 0:
            extra_contents = False
        else:
            extra_contents = True

        next_chunk = True

        if self.const_sequence:  # for constant sequence length (works for any stride from 1 to sequence_max)
            while next_chunk:
                x, misc, next_chunk = h5_r.next()

                t = misc[0]
                ip = misc[1]
                ys = misc[2:]

                if next_chunk:
                    x_buffer[:-stride] = x_buffer[stride:]  # x is moved to the top
                    x_buffer[-stride:] = x  # bottom is replaced with new x
                    flow_n += stride

                    for n in range(self.labels_len):
                        ys_buffer[n][:-stride] = ys_buffer[n][stride:]
                        ys_buffer[n][-stride:] = ys[n]

                    # cases when stride less than sequence_max
                    if flow_n >= self.sequence_max:  # write only when buffer is filled
                        x_w.append([x_buffer])
                        seq_w.append([self.sequence_max])

                        for n in range(self.labels_len):
                            ys_w[n].append([ys_buffer[n]])

                        if extra_contents:  # when stride < sequence_max, only last <stride> contents are written
                            t_w.append(t)
                            ip_w.append(ip)

                # cases when stride is more than 1
                else:  # when processing last chunk
                    cur_shape = x.shape[0]

                    if cur_shape == stride:  # when last chunk has sufficient length (codes below are duplicated)
                        x_buffer[:-stride] = x_buffer[stride:]
                        x_buffer[-stride:] = x
                        flow_n += stride

                        for n in range(self.labels_len):
                            ys_buffer[n][:-stride] = ys_buffer[n][stride:]
                            ys_buffer[n][-stride:] = ys[n]

                        if flow_n >= self.sequence_max:
                            x_w.append([x_buffer])
                            seq_w.append([self.sequence_max])

                            for n in range(self.labels_len):
                                ys_w[n].append([ys_buffer[n]])

                            if extra_contents:
                                t_w.append(t)
                                ip_w.append([ip])

        else:  # dynamic sequence length (updated, but not tested)
            while next_chunk:
                x, misc, next_chunk = h5_r.next()

                t = misc[0]
                ip = misc[1]
                ys = misc[2:]

                if next_chunk:
                    # cases when stride less than sequence_max
                    if flow_n >= self.sequence_max:  # write only when buffer is filled

                        x_buffer[:-stride] = x_buffer[stride:]  # x is moved to the top
                        x_buffer[-stride:] = x  # bottom is replaced with new x
                        flow_n += stride

                        for n in range(self.labels_len):
                            ys_buffer[n][:-stride] = ys_buffer[n][stride:]
                            ys_buffer[n][-stride:] = ys[n]

                        x_w.append([x_buffer])
                        seq_w.append([self.sequence_max])

                        for n in range(self.labels_len):
                            ys_w[n].append([ys_buffer[n]])

                        if extra_contents:  # when stride < sequence_max, only last <stride> contents are written
                            t_w.append(t)
                            ip_w.append(ip)

                    # contents are prepended instead of appending them
                    else:  # processing first few chunks when buffer is not filled

                        # when contents exceed (excess contents are ignored and code is not implemented)
                        if (flow_n + stride) > self.sequence_max:  # occurs when stride > 1 and < sequence_max
                            print("[WinSegt] WARNING: CONTENTS ARE NOT APPENDED CORRECTLY")

                        else:

                            x_buffer[flow_n:stride + flow_n] = x
                            flow_n += stride

                            for n in range(self.labels_len):
                                ys_buffer[n][flow_n:stride + flow_n] = ys[n]

                            if flow_n >= self.sequence_max:
                                x_w.append([x_buffer])
                                seq_w.append([flow_n])

                                for n in range(self.labels_len):
                                    ys_w[n].append([ys_buffer[n]])

                                if extra_contents:
                                    t_w.append(t)
                                    ip_w.append([ip])

                else:
                    cur_shape = x.shape[0]

                    if cur_shape == stride:  # when stride is compatible with last chunk
                        x_buffer[:-stride] = x_buffer[stride:]
                        x_buffer[-stride:] = x
                        flow_n += stride

                        for n in range(self.labels_len):
                            ys_buffer[n][:-stride] = ys_buffer[n][stride:]
                            ys_buffer[n][-stride:] = ys[n]

                        x_w.append([x_buffer])
                        for n in range(self.labels_len):
                            ys_w[n].append([ys_buffer[n]])

                    else:  # only happens when stride is more than 1 and last chunk is incompatible
                        x_zeros_buffer = np.zeros((self.sequence_max, self.features_len))  # zeros filler
                        ys_zeros_buffer = [np.zeros(self.sequence_max) for _ in range(self.labels_len)]

                        x_zeros_buffer[:-stride] = x_buffer[stride:]
                        x_zeros_buffer[-stride:] = x  # leftovers = zeros
                        flow_n += stride

                        for n in range(self.labels_len):
                            ys_zeros_buffer[n][:-stride] = ys_buffer[n][stride:]
                            ys_zeros_buffer[n][-stride:] = ys[n]

                        x_w.append([x_zeros_buffer])
                        for n in range(self.labels_len):
                            ys_w[n].append([ys_zeros_buffer[n]])

                    seq_w.append([cur_shape])

                    if extra_contents:  # when stride < sequence_max, only last <stride> contents are written
                        t_w.append(t)
                        ip_w.append(ip)

                print(flow_n, end='\r')

        print(flow_total_n, "total flows")

        # assert flow_total_n == flow_n, "Number of flows processed not tally, expected " + \
        #                                str(flow_total_n) + " flows"

        return flow_n
