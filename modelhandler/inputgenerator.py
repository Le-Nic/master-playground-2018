import tables as tb


class Generator:
    # https://stackoverflow.com/questions/48309631/tensorflow-tf-data-dataset-reading-large-hdf5-files
    def __init__(self, filepath, class_type, is_stateful_ip, is_m1, is_4d=False):
        self.file_path = filepath
        self.class_type = '/y/y' + str(class_type)
        self.is_stateful_ip = is_stateful_ip
        self.is_m1 = is_m1
        self.is_4d = is_4d
        self.dataset_n = None

    def __call__(self):

        # # instances: 3487772, Epoch 1: 05:06m, Epoch 2: 05:09m

        h5_r = tb.open_file(self.file_path, mode='r')
        x_r = h5_r.get_node('/x')
        y_r = h5_r.get_node(self.class_type)
        # y_extra_r = h5_r.get_node('/y/y1')  # additional y_extra
        ip_r = h5_r.get_node('/ip')

        if self.is_4d:  # hierarchical model
            netw_seq_r = h5_r.get_node('/netw_seq')
            host_seq_r = h5_r.get_node('/host_seq')

            for data, label, netw_sequence, host_sequence in zip(
                    x_r.iterrows(), y_r.iterrows(), netw_seq_r.iterrows(), host_seq_r.iterrows()):
                yield (data, label[netw_sequence-1], netw_sequence, host_sequence)

        else:
            seq_r = h5_r.get_node('/seq')
            assert x_r.shape[0] == y_r.shape[0] == seq_r.shape[0]

            if self.is_m1:
                if self.is_stateful_ip:
                    for data, label, sequence, ips in zip(
                            x_r.iterrows(), y_r.iterrows(), seq_r.iterrows(), ip_r.iterrows()):
                        yield (data, label[sequence-1], sequence, ips)
                else:
                    # for data, label, sequence, label_extra in zip(  # additional y_extra
                    #         x_r.iterrows(), y_r.iterrows(), seq_r.iterrows(), y_extra_r.iterrows()
                    # ):  # additional y_extra
                    for data, label, sequence in zip(x_r.iterrows(), y_r.iterrows(), seq_r.iterrows()):
                        yield (data, label[sequence-1],  sequence)
            else:
                if self.is_stateful_ip:
                    for data, label, sequence, ips in zip(
                            x_r.iterrows(), y_r.iterrows(), seq_r.iterrows(), ip_r.iterrows()):
                        yield (data, label, sequence, ips)
                else:
                    for data, label, sequence in zip(x_r.iterrows(), y_r.iterrows(), seq_r.iterrows()):
                        yield (data, label, sequence)

        h5_r.close()

        # # instances: 3487772, Epoch 1: 16:46m, Epoch 2: 17:03m
        # with h5py.File(self.file_path, 'r') as hf:
        #     for data, label, sequence in zip(hf['x'], hf[self.class_type], hf['seq']):
        #         yield (data, label, sequence)

    def get_instances(self):
        if self.dataset_n is None:
            h5_r = tb.open_file(self.file_path, mode='r')
            self.dataset_n = h5_r.get_node('/x').shape[0]
            h5_r.close()

        return self.dataset_n
