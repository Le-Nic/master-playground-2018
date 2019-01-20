from modelhandler.lstmodel import *
from modelhandler.inputgenerator import Generator

from sklearn.metrics import confusion_matrix
import tensorflow as tf
import numpy as np
import tables as tb
import os
import pathlib
import time
import logging

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class ModelTrainer(object):
    def __init__(self, configs, dataset_meta, checkpoint_dir=None, saver_dir=None):
        
        tf.reset_default_graph()
        
        self.seed_value = 147

        self.hyperparams = configs['hyperparameters']
        self.class_type = configs['class_type']
        self.stateful_ip = configs['stateful_ip']
        self.m1_labels = configs['m1_labels']
        self.batch_n_test = configs['batch_n_test']

        if self.stateful_ip:
            self._train = self._train_ip_func
            self._save_output = self._output_ip_func
            self._validate = self._validate_ip_func
        else:
            self._train = self._train_func
            self._save_output = self._output_func
            self._validate = self._validate_func

        self.checkpoint_dir = checkpoint_dir + "/" if checkpoint_dir is not None else None
        self.saver_dir = saver_dir + "/" if saver_dir is not None else ""
        self.model_name = "_s" + str(self.hyperparams['netw_sequence']) + \
                          "u" + str(self.hyperparams['units_n']) + \
                          "b" + str(self.hyperparams['batch_n']) + \
                          "l" + str(self.hyperparams['layers_n']) + \
                          "d" + str(int(self.hyperparams['dropout_r']*100)) + \
                          "_y" + str(self.class_type)
        self.save_output = configs['save_output'] if configs['save_output'] else None

        if self.stateful_ip:
            print("[MT Config.] IP Stateful behaviour")
        print("[MT Config.]", "M:1" if self.m1_labels else "M:N", "labeling strategy")
        print("[MT Config.] Dev Batch size:", self.batch_n_test)
        print("[MT Config.] Sequence:", self.hyperparams['netw_sequence'])
        print("[MT Config.] Batch size:", self.hyperparams['batch_n'])
        print("[MT Config.] Epochs:", self.hyperparams['epochs_n'])
        print("[MT Config.] Hidden units:", self.hyperparams['units_n'])
        print("[MT Config.] Layer(s):", self.hyperparams['layers_n'])
        print("[MT Config.] Dropout rate:", self.hyperparams['dropout_r'])

        # get Labels Mapping and Features Length
        try:
            self.y_dict = dataset_meta['y_dict']
            self.features_len = dataset_meta['features_len']

        except TypeError:
            meta_h5 = tb.open_file(dataset_meta, mode='r')
            self.y_dict = dict(enumerate(meta_h5.get_node("/y" + str(self.class_type)).read()))
            self.features_len = len(meta_h5.get_node("/x").read())

            print("[ModelTrainer] Features length:", str(self.features_len))
            print("[ModelTrainer] Labels:", ', '.join(["{0} - {1}".format(k, v.decode("utf-8"))
                                                       for k, v in self.y_dict.items()]))

            meta_h5.close()

        self.model_train = None
        self.model_dev = None
        self.model_test = None

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
                        'gen': Generator(str(child), self.class_type, self.stateful_ip, self.m1_labels)
                    })
                    file_count += 1

        elif data_path.is_file():
            file_count += 1
            datasets.append({
                'name': data_path.stem,
                'path': data_dir,
                'gen': Generator(data_dir, self.class_type, self.stateful_ip, self.m1_labels)
            })

        print("[ModelTrainer]", file_count, "dataset(s) found in >", data_dir)
        return datasets

    def _model_init(self, batch_n, is_training):

        if self.stateful_ip:
            return LSTModel({
                'features': tf.placeholder(tf.float32, name="features", shape=[
                    batch_n, self.hyperparams['netw_sequence'], self.features_len]),
                'labels': tf.placeholder(tf.int32, name="labels", shape=batch_n if self.m1_labels else [
                    batch_n, self.hyperparams['netw_sequence']]),
                'sequences': tf.placeholder(tf.int32, name="sequences", shape=batch_n),
                'states': tf.placeholder(tf.float32, shape=[
                    self.hyperparams['layers_n'], 2, batch_n, self.hyperparams['units_n']])
            }, self.hyperparams, self.features_len, len(self.y_dict), self.m1_labels, self.seed_value, is_training)

        else:
            return LSTModel({
                'features': tf.placeholder(tf.float32, name="features", shape=[
                    batch_n, self.hyperparams['netw_sequence'], self.features_len]),
                'labels': tf.placeholder(tf.int32, name="labels", shape=batch_n if self.m1_labels else [
                    batch_n, self.hyperparams['netw_sequence']]),
                'sequences': tf.placeholder(tf.int32, name="sequences", shape=batch_n)
            }, self.hyperparams, self.features_len, len(self.y_dict), self.m1_labels, self.seed_value, is_training)

    def _dataset_prep(self, generator, batch_size):

        dataset = tf.data.Dataset.from_generator(
            generator,
            output_types=(tf.float32, tf.int32, tf.int32, tf.int32) if self.stateful_ip else (
                tf.float32, tf.int32, tf.int32),
            output_shapes=(
                tf.TensorShape([self.hyperparams['netw_sequence'], self.features_len]),  # x
                tf.TensorShape([] if self.m1_labels else [self.hyperparams['netw_sequence']]),  # y
                tf.TensorShape([]),  # seq
                tf.TensorShape([self.hyperparams['netw_sequence'], 2])  # ip
            ) if self.stateful_ip else (
                tf.TensorShape([self.hyperparams['netw_sequence'], self.features_len]),  # x
                tf.TensorShape([] if self.m1_labels else [self.hyperparams['netw_sequence']]),  # y
                tf.TensorShape([]),  # seq
            )
        )

        if self.stateful_ip:
            dataset = dataset.map(lambda x, y, seq, ip: (
                x, y, seq, tf.cast(
                    (ip[:, 0] * ip[:, 1]), tf.int32) + tf.cast(((tf.abs(ip[:, 0] - ip[:, 1]) - 1) ** 2) / 4, tf.int32)
            ))

        dataset = dataset.batch(batch_size, drop_remainder=True)
        # dataset = dataset.padded_batch(self.hyperparams['batch_n'], padded_shapes=[])
        dataset = dataset.prefetch(buffer_size=batch_size)

        return dataset.make_one_shot_iterator().get_next()

    def _validate_ip_func(self, sess, devsets):

        for devset in devsets:
            next_element = self._dataset_prep(devset['gen'], self.batch_n_test)
            t = time.time()

            predictions = []
            ground_truth = []
            acc_dev = 0
            step_dev = 0
            ip_states = {}

            try:
                while True:
                    dataset_batched = sess.run(next_element)
                    features_batched, label_batched, seq_batched, ips_batched = dataset_batched

                    state_current = np.zeros((self.hyperparams['layers_n'], 2,
                                              self.batch_n_test, self.hyperparams['units_n']))

                    for m_batch, ip in enumerate(ips_batched[:, -1]):
                        get_state = ip_states.get(ip, [(.0, .0) for _ in range(self.hyperparams['layers_n'])])
                        for i, state in enumerate(state_current):
                            state_current[i][0] = get_state[i][0]
                            state_current[i][1] = get_state[i][1]

                    outputs, truth, pred, acc = sess.run(
                        self.model_dev.error,  # error prior backpropagation
                        feed_dict={
                            self.model_dev.features_placeholder: features_batched,
                            self.model_dev.label_placeholder: label_batched,
                            self.model_dev.seq_placeholder: seq_batched,
                            self.model_dev.state_placeholder: state_current
                        }
                    )
                    # saving states based on IP (earlier state from same IP will be replaced)
                    for m_batch, ip in enumerate(ips_batched[:, -1]):
                        ip_states[ip] = [(state[0][m_batch], state[1][m_batch]) for state in outputs[1]]

                    predictions.extend(pred)
                    ground_truth.extend(truth)

                    acc_dev += acc
                    step_dev += 1

            except tf.errors.OutOfRangeError:
                pass

            acc = round(acc_dev / step_dev, 9)
            cm = confusion_matrix(ground_truth, predictions, labels=[label for label in range(len(self.y_dict))])

            logging.info("[ModelTrainer] acc: %.9f  time: " % acc +
                         time.strftime("%H:%M:%S", time.gmtime(time.time() - t)))
            for row in cm:
                logging.info(" ".join(str(col) for col in row))

        return True

    def _validate_func(self, sess, devsets):

        for devset in devsets:
            next_element = self._dataset_prep(devset['gen'], self.batch_n_test)
            t = time.time()

            predictions = []
            ground_truth = []
            acc_dev = 0
            step_dev = 0

            try:
                while True:
                    features_batched, label_batched, seq_batched = sess.run(next_element)

                    _, truth, pred, acc = sess.run(
                        self.model_dev.error,
                        feed_dict={
                            self.model_dev.features_placeholder: features_batched,
                            self.model_dev.label_placeholder: label_batched,
                            self.model_dev.seq_placeholder: seq_batched
                        }
                    )
                    predictions.extend(pred)
                    ground_truth.extend(truth)

                    acc_dev += acc
                    step_dev += 1

            except tf.errors.OutOfRangeError:
                pass

            acc = round(acc_dev / step_dev, 9)
            cm = confusion_matrix(ground_truth, predictions, labels=[label for label in range(len(self.y_dict))])

            logging.info("[ModelTrainer] acc: %.9f  time: " % acc +
                         time.strftime("%H:%M:%S", time.gmtime(time.time() - t)))
            for row in cm:
                logging.info(" ".join(str(col) for col in row))

        return True

    def validate(self, train_dir, dev_dir):
        np.random.seed = self.seed_value
        tf.set_random_seed(self.seed_value)

        trainsets = self._get_files(train_dir)
        devsets = self._get_files(dev_dir)

        self.model_dev = self._model_init(self.batch_n_test, False)
        saver = tf.train.Saver()

        with tf.Session() as sess:
            if self.checkpoint_dir:
                saver.restore(sess, self.checkpoint_dir + trainsets[0]['name'] + self.model_name)
                print("[ModelTrainer] model restored for validation >",
                      self.checkpoint_dir + trainsets[0]['name'] + self.model_name)

                self._validate(sess, devsets)

    def _output_ip_func(self, sess, saver, my_model, trainset, batch_n, log_output=False, devset=None):

        saver.restore(sess, self.saver_dir + trainset['name'] + self.model_name)
        logging.info("[ModelTrainer] restored model > " + trainset['name'] + self.model_name)

        output_name = trainset['name'] if devset is None else devset['name']
        output_path = self.save_output + ("/train/" if devset is None else "/dev/")
        pathlib.Path(output_path).mkdir(parents=True, exist_ok=True)

        # ARFF header creation
        # arffs_w_last = [open(output_path + output_name + self.model_name + "_" + str(layer_n) + "_last.arff", 'wb')
        #                 for layer_n in range(1, self.hyperparams['layers_n']+1)]
        arff_w_last = open(output_path + output_name + self.model_name + "_" + self.hyperparams['layers_n'] +
                           "_last.arff", 'wb')

        header_last = np.array(["@relation " + output_name + self.model_name + "_last", ""])

        for n in range(self.hyperparams['units_n']):
            header_last = np.append(header_last, ("@attribute '" + str(n) + "' numeric"))

        header_label = "@attribute 'y' {" + ','.join(
            str(y) for y in range(len(self.y_dict))) + "}"

        header_last = np.append(header_last, header_label)
        header_last = np.append(header_last, ("", "@data"))

        # for layer_n, arff_w_last in enumerate(arffs_w_last, 1):
        #     np.savetxt(arff_w_last, header_last[np.newaxis].T, fmt='%s')
        #     arff_w_last.close()
        #     arffs_w_last[layer_n-1] = open(
        #         output_path + output_name + self.model_name + "_" + str(layer_n) + "_last.arff", 'ab')
        np.savetxt(arff_w_last, header_last[np.newaxis].T, fmt='%s')

        predictions = []
        ground_truth = []
        acc_dev = 0
        step_dev = 0

        next_element = self._dataset_prep(trainset['gen'] if devset is None else devset['gen'], batch_n)
        t = time.time()

        try:
            ip_states = {}
            while True:
                features_batched, label_batched, seq_batched, ips_batched = sess.run(next_element)

                state_current = np.zeros((self.hyperparams['layers_n'], 2,
                                          batch_n, self.hyperparams['units_n']))

                for m_batch, ip in enumerate(ips_batched[:, -1]):
                    get_state = ip_states.get(ip, [(.0, .0) for _ in range(self.hyperparams['layers_n'])])

                    for i, state in enumerate(state_current):
                        state_current[i][0] = get_state[i][0]
                        state_current[i][1] = get_state[i][1]
                        # print(state_current[i][0])

                (output, state), truth, pred, acc = sess.run(  # error prior backpropagation
                    my_model.error,
                    feed_dict={
                        my_model.features_placeholder: features_batched,
                        my_model.label_placeholder: label_batched,
                        my_model.seq_placeholder: seq_batched,
                        my_model.state_placeholder: state_current
                    }
                )

                for m_batch, ip in enumerate(ips_batched[:, -1]):
                    ip_states[ip] = [(s[0][m_batch], s[1][m_batch]) for s in state]

                predictions.extend(pred)
                ground_truth.extend(truth)

                acc_dev += acc
                step_dev += 1

                if self.m1_labels:

                    # for i, arff_w_last in enumerate(arffs_w_last):
                    #     # noinspection PyTypeChecker
                    #     np.savetxt(  # output for last sequence
                    #         arff_w_last, np.append(outputs[1][i].h, truth[np.newaxis].T, axis=1),
                    #         fmt="%.18e," * (self.hyperparams['units_n']) + "%i")  # last layer: output[..., -1, :]
                    # noinspection PyTypeChecker
                    np.savetxt(
                        arff_w_last, np.append(
                            np.concatenate([s.h for s in state], axis=1), truth[np.newaxis].T, axis=1),
                        fmt="%.18e," * (self.hyperparams['units_n']) + "%i")

                else:  # only outputs from last layer are saved, unable to acces all sequences in intermediate layers
                    # noinspection PyTypeChecker
                    np.savetxt(  # output for all sequences at last layer, each of them a different ground truth
                        arff_w_last, np.append(
                            np.reshape(output, (batch_n * self.hyperparams['netw_sequence'], -1)),
                            truth[np.newaxis].T, axis=1), fmt="%.18e," * self.hyperparams['units_n'] + "%i")

        except tf.errors.OutOfRangeError:
            pass

        # for arff_w_last in arffs_w_last:
        #     arff_w_last.close()
        arff_w_last.close()

        logging.info("[ModelTrainer] vectors > " + output_path + output_name + self.model_name)

        if log_output:
            acc = round(acc_dev / step_dev, 9)
            cm = confusion_matrix(ground_truth, predictions, labels=[label for label in range(len(self.y_dict))])

            logging.info("[ModelTrainer] acc: %.9f  time: " % acc +
                         time.strftime("%H:%M:%S", time.gmtime(time.time() - t)))
            for row in cm:
                logging.info(" ".join(str(col) for col in row))

    def _output_func(self, sess, saver, my_model, trainset, batch_n, log_output=False, devset=None):

        saver.restore(sess, self.saver_dir + trainset['name'] + self.model_name)
        if log_output:
            logging.info("[ModelTrainer] restored model > " + trainset['name'] + self.model_name)
        else:
            print("[ModelTrainer] restored model > " + trainset['name'] + self.model_name)

        output_name = trainset['name'] if devset is None else devset['name']
        output_path = self.save_output + ("/train/" if devset is None else "/dev/")
        pathlib.Path(output_path).mkdir(parents=True, exist_ok=True)

        # ARFF header creation
        # arffs_w_last = [open(output_path + output_name + self.model_name + "_" + str(layer_n) + "_last.arff", 'wb')
        #                 for layer_n in range(1, self.hyperparams['layers_n']+1)]
        arff_w_last = open(output_path + output_name + self.model_name + "_" + str(self.hyperparams['layers_n']) +
                           "_last.arff", 'wb')

        header_last = np.array(["@relation " + output_name + self.model_name + "_last", ""])

        for layer_n in range(1, self.hyperparams['layers_n']+1):
            for n in range(self.hyperparams['units_n']):
                header_last = np.append(header_last, ("@attribute '" + str(layer_n) + "_" + str(n) + "' numeric"))

        header_label = "@attribute 'y' {" + ','.join(
            str(y) for y in range(len(self.y_dict))) + "}"

        header_last = np.append(header_last, header_label)
        header_last = np.append(header_last, ("", "@data"))

        # for layer_n, arff_w_last in enumerate(arffs_w_last, 1):
        #     np.savetxt(arff_w_last, header_last[np.newaxis].T, fmt='%s')
        #     arff_w_last.close()
        #     arffs_w_last[layer_n-1] = open(
        #         output_path + output_name + self.model_name + "_" + str(layer_n) + "_last.arff", 'ab')
        np.savetxt(arff_w_last, header_last[np.newaxis].T, fmt='%s')

        predictions = []
        ground_truth = []
        acc_dev = 0
        step_dev = 0

        next_element = self._dataset_prep(trainset['gen'] if devset is None else devset['gen'], batch_n)
        t = time.time()

        try:
            while True:
                features_batched, label_batched, seq_batched = sess.run(next_element)

                (output, state), truth, pred, acc = sess.run(  # error prior backpropagation
                    my_model.error,
                    feed_dict={
                        my_model.features_placeholder: features_batched,
                        my_model.label_placeholder: label_batched,
                        my_model.seq_placeholder: seq_batched
                    }
                )

                predictions.extend(pred)
                ground_truth.extend(truth)

                acc_dev += acc
                step_dev += 1

                if self.m1_labels:

                    # for i, arff_w_last in enumerate(arffs_w_last):
                        # noinspection PyTypeChecker
                        # np.savetxt(  # output for last sequence
                        #     arff_w_last, np.append(state[i].h, truth[np.newaxis].T, axis=1),
                        #     fmt="%.18e," * (self.hyperparams['units_n']) + "%i")  # last layer: output[..., -1, :]
                    # noinspection PyTypeChecker
                    np.savetxt(
                        arff_w_last, np.append(
                            np.concatenate([s.h for s in state], axis=1), truth[np.newaxis].T, axis=1),
                        fmt="%.18e," * (self.hyperparams['units_n']) * self.hyperparams['layers_n'] + "%i")

                else:  # only outputs from last layer are saved, unable to acces all sequences in intermediate layers
                    # noinspection PyTypeChecker
                    np.savetxt(  # output for all sequences at last layer, each of them a different ground truth
                        arff_w_last, np.append(
                            np.reshape(output, (batch_n * self.hyperparams['netw_sequence'], -1)),
                            truth[np.newaxis].T, axis=1), fmt="%.18e," * self.hyperparams['units_n'] + "%i")

        except tf.errors.OutOfRangeError:
            pass

        # for arff_w_last in arffs_w_last:
        #     arff_w_last.close()
        arff_w_last.close()

        if log_output:
            logging.info("[ModelTrainer] vectors > " + output_path + output_name + self.model_name)
        else:
            print("[ModelTrainer] vectors > " + output_path + output_name + self.model_name)

        acc = round(acc_dev / step_dev, 9)
        cm = confusion_matrix(ground_truth, predictions, labels=[label for label in range(len(self.y_dict))])

        if log_output:
            logging.info("[ModelTrainer] acc: %.9f  time: " % acc +
                         time.strftime("%H:%M:%S", time.gmtime(time.time() - t)))
            for row in cm:
                logging.info(" ".join(str(col) for col in row))
        else:
            print("[ModelTrainer] acc: %.9f  time: " % acc + time.strftime("%H:%M:%S", time.gmtime(time.time() - t)))
            for row in cm:
                print(" ".join(str(col) for col in row))

    def _train_ip_func(self, sess, next_element):

        loss_train = 0
        acc_train = 0
        step_train = 0
        ip_states = {}

        try:
            while True:
                dataset_batched = sess.run(next_element)
                features_batched, label_batched, seq_batched, ips_batched = dataset_batched

                state_current = np.zeros((self.hyperparams['layers_n'], 2,
                                          self.hyperparams['batch_n'], self.hyperparams['units_n']))

                for m_batch, ip in enumerate(ips_batched[:, -1]):
                    get_state = ip_states.get(ip, [(.0, .0) for _ in range(self.hyperparams['layers_n'])])
                    for i, state in enumerate(state_current):
                        state_current[i][0] = get_state[i][0]
                        state_current[i][1] = get_state[i][1]

                (_, loss), (outputs, _, _, acc) = sess.run(
                    [self.model_train.optimize, self.model_train.error],  # error prior backpropagation
                    feed_dict={
                        self.model_train.features_placeholder: features_batched,
                        self.model_train.label_placeholder: label_batched,
                        self.model_train.seq_placeholder: seq_batched,
                        self.model_train.state_placeholder: state_current
                    }
                )

                # saving states based on IP (earlier state from same IP will be replaced)
                for m_batch, ip in enumerate(ips_batched[:, -1]):
                    ip_states[ip] = [(state[0][m_batch], state[1][m_batch]) for state in outputs[1]]

                loss_train += loss
                acc_train += acc
                step_train += 1

        except tf.errors.OutOfRangeError:
            return (loss_train / step_train), (acc_train / step_train)

    def _train_func(self, sess, next_element):
        loss_train = 0
        acc_train = 0
        step_train = 0

        try:
            while True:
                features_batched, label_batched, seq_batched = sess.run(next_element)
                (_, loss), (_, _, _, acc) = sess.run(  # error prior backpropagation
                    [self.model_train.optimize, self.model_train.error],
                    feed_dict={
                        self.model_train.features_placeholder: features_batched,
                        self.model_train.label_placeholder: label_batched,
                        self.model_train.seq_placeholder: seq_batched
                    }
                )
                loss_train += loss
                acc_train += acc
                step_train += 1

        except tf.errors.OutOfRangeError:
            return (loss_train / step_train), (acc_train / step_train)

    def train(self, train_dir, dev_dir):
        np.random.seed = self.seed_value
        tf.set_random_seed(self.seed_value)

        trainsets = self._get_files(train_dir)
        devsets = self._get_files(dev_dir)
        pathlib.Path(self.saver_dir).mkdir(parents=True, exist_ok=True)

        epoch_current = 1
        prev_loss_dev = 999999999.
        tolerance = self.hyperparams['e.stopping'] if self.hyperparams['e.stopping'] else 0

        # ========== Model Restoration Check ==========
        if self.checkpoint_dir:
            if os.path.isfile(self.checkpoint_dir + trainsets[0]['name'] + self.model_name + ".log"):
                with open(self.checkpoint_dir + trainsets[0]['name'] + self.model_name + ".log", "r") as log_r:

                    saved_count = 0
                    for line in log_r:
                        log_output = line.split()

                        if log_output[0] == "[ModelTrainer]":
                            if log_output[1] == "vectors":  # indication for saved vectors
                                saved_count += 1
                            elif log_output[-1] == "saved":
                                prev_loss_dev = float(log_output[4])
                                epoch_current = int(log_output[2]) + 1

                    if saved_count >= 1:
                        return True
                    if prev_loss_dev >= 999999999.:
                        self.checkpoint_dir = None

            else:
                self.checkpoint_dir = None
        # ========== Model Restoration Check ==========

        # Logger Setup
        log_file = logging.FileHandler(self.saver_dir + trainsets[0]['name'] + self.model_name + ".log")
        log_console = logging.StreamHandler()
        logging.basicConfig(level=logging.INFO, format="%(message)s", handlers=[log_file, log_console])

        logging.info("---- ---- ---- ----")

        for trainset in trainsets:
            logging.info("[ModelTrainer] Train Set instances: " + str(trainset['gen'].get_instances()))
        for devset in devsets:
            logging.info("[ModelTrainer] Dev Set instances: " + str(devset['gen'].get_instances()))

            if devset['gen'].get_instances() % self.batch_n_test != 0:
                logging.info("[ModelTrainer] [WARNING] Validation batch size (" +
                             str(self.batch_n_test) + ") is not tally")

        # Model Initialization
        self.model_train = self._model_init(self.hyperparams['batch_n'], True)
        self.model_dev = self._model_init(self.batch_n_test, False)

        saver = tf.train.Saver()

        with tf.Session() as sess:

            # Model & Weights Restoration (if necessary)
            if self.checkpoint_dir or prev_loss_dev < 999999999.:
                saver.restore(sess, self.checkpoint_dir + trainsets[0]['name'] + self.model_name)
                logging.info("[ModelTrainer] model restored,  epoch: " + str(epoch_current-1))

                self._validate(sess, devsets)
            else:
                sess.run(tf.global_variables_initializer())

            # ========== Model Training ==========
            for epoch_n in range(epoch_current, self.hyperparams['epochs_n']+1):
                t = time.time()
                loss_epoch, acc_epoch = self._train(
                    sess, self._dataset_prep(trainsets[0]['gen'], self.hyperparams['batch_n'])
                ) if self.stateful_ip else self._train(
                    sess, self._dataset_prep(trainsets[0]['gen'], self.hyperparams['batch_n'])
                )

                # ========== Saver & Early Stopping ==========
                if self.saver_dir and (not self.hyperparams['e.stopping'] or (
                        self.hyperparams['e.stopping'] and
                        epoch_n >= 50 and  # look at weights only after 19th epochs
                        loss_epoch < prev_loss_dev and loss_epoch < 0.1)):

                    saver.save(sess, self.saver_dir + trainsets[0]['name'] + self.model_name)
                    logging.info("[ModelTrainer] epoch: " + str(epoch_n) +
                                 "  loss: %.9f" % round(loss_epoch, 9) +
                                 "  acc: %.9f" % round(acc_epoch, 9) +
                                 "  time: " + time.strftime("%H:%M:%S", time.gmtime(time.time() - t)) + "  saved")
                    prev_loss_dev = loss_epoch
                    tolerance = self.hyperparams['e.stopping'] if self.hyperparams['e.stopping'] else 0

                elif not self.saver_dir or self.hyperparams['e.stopping']:
                    if epoch_n >= 50 and loss_epoch < 0.1:  # look at weights only after 19th epochs
                        tolerance -= 1

                    logging.info("[ModelTrainer] epoch: " + str(epoch_n) +
                                 "  loss: %.9f" % round(loss_epoch, 9) +
                                 "  acc: %.9f" % round(acc_epoch, 9) +
                                 "  time: " + time.strftime("%H:%M:%S", time.gmtime(time.time() - t)))

                    if self.hyperparams['e.stopping'] and tolerance <= 0:
                        logging.info("[ModelTrainer] e.stop,  epoch: " + str(epoch_n - self.hyperparams['e.stopping']))

                        if self.save_output:
                            self._save_output(sess, saver, self.model_train, trainsets[0],
                                              self.hyperparams['batch_n'], False)
                            self._save_output(sess, saver, self.model_dev, trainsets[0],
                                              self.batch_n_test, True, devsets[0])
                        break
                # ========== Saver & Early Stopping ==========

                # calculate validation acc every n'th epoch
                if not (epoch_n % self.hyperparams['calc_dev']):
                    self._validate(sess, devsets)

            # max. epoch reached, save output from last checkpoint
            if self.hyperparams['e.stopping'] and tolerance > 0:
                logging.info("[ModelTrainer] maximum epoch, restoring best model")

                if self.save_output:
                    self._save_output(sess, saver, self.model_train, trainsets[0],
                                      self.hyperparams['batch_n'], False)
                    self._save_output(sess, saver, self.model_dev, trainsets[0],
                                      self.batch_n_test, True, devsets[0])
            # ========== Model Training ==========

        logging.getLogger().removeHandler(log_file)
        logging.getLogger().removeHandler(log_console)

        return True

    def gen_output(self, train_dir, dev_dir, output_trainset=False):
        """ transform dev set from saved model """

        np.random.seed = self.seed_value
        tf.set_random_seed(self.seed_value)

        trainsets = self._get_files(train_dir)
        devsets = self._get_files(dev_dir)

        model_train = self._model_init(
            self.hyperparams['batch_n'], True) if self.model_train is None else self.model_train
        model_dev = self._model_init(self.batch_n_test, False) if self.model_dev is None else self.model_dev

        saver = tf.train.Saver()

        with tf.Session() as sess:
            if self.checkpoint_dir:
                saver.restore(sess, self.checkpoint_dir + trainsets[0]['name'] + self.model_name)

                if output_trainset:
                    self._save_output(sess, saver, model_train, trainsets[0],
                                      self.hyperparams['batch_n'], False)

                self._save_output(sess, saver, model_dev, trainsets[0],
                                  self.batch_n_test, False, devsets[0])
            else:
                print("[ModelTrainer] checkpoint directory not given")
