from modelhandler.hierarchical.hierarchicalmodel import *
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
        self.batch_n_dev = configs['batch_n_dev']

        self._train = self._train_func
        self._validate = self._validate_func
        self._save_output = self._output_func

        self.checkpoint_dir = checkpoint_dir + "/" if checkpoint_dir is not None else None
        self.saver_dir = saver_dir + "/" if saver_dir is not None else ""
        self.model_name = "_HS" + str(self.hyperparams['host_sequence']) + \
                          "NO" + str(self.hyperparams['netw_output_n']) + \
                          "HO" + str(self.hyperparams['host_output_n']) + \
                          "U" + str(self.hyperparams['units_n']) + \
                          "B" + str(self.hyperparams['batch_n']) + \
                          "L" + str(self.hyperparams['layers_n']) + \
                          "D" + str(int(self.hyperparams['dropout_r'] * 100)) + \
                          "LR" + "{:.0e}".format(self.hyperparams['learning_r']) + \
                          "_y" + str(self.class_type)

        self.save_output = configs['save_output'] if configs['save_output'] else None

        print("[HMT Config.]", "M:1" if self.m1_labels else "M:N", "labeling strategy")
        print("[HMT Config.] Test set Batch size:", self.batch_n_test)
        print("[HMT Config.] Dev set Batch size:", self.batch_n_dev)
        print("[HMT Config.] Network-level Sequence:", self.hyperparams['netw_sequence'])
        print("[HMT Config.] Host-level Sequence:", self.hyperparams['host_sequence'])
        print("[HMT Config.] Batch size:", self.hyperparams['batch_n'])
        print("[HMT Config.] Epochs:", self.hyperparams['epochs_n'])
        print("[HMT Config.] Hidden units:", self.hyperparams['units_n'])
        print("[HMT Config.] Layer(s):", self.hyperparams['layers_n'])
        print("[HMT Config.] Dropout rate:", self.hyperparams['dropout_r'])
        print("[HMT Config.] Learning rate:", self.hyperparams['learning_r'])

        # get Labels Mapping and Features Length
        try:
            self.y_dict = dataset_meta['y_dict']
            self.features_len = dataset_meta['features_len']

        except TypeError:
            meta_h5 = tb.open_file(dataset_meta, mode='r')
            self.y_dict = dict(enumerate(meta_h5.get_node("/y" + str(self.class_type)).read()))
            self.features_len = len(meta_h5.get_node("/x").read())

            print("[HModelTrainer] Features length:", str(self.features_len))
            print("[HModelTrainer] Labels:", ', '.join(["{0} - {1}".format(k, v.decode("utf-8"))
                                                       for k, v in self.y_dict.items()]))

            self.model_train = None
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
                        'gen': Generator(str(child), self.class_type, False, self.m1_labels, True)
                    })
                    file_count += 1

        elif data_path.is_file():
            file_count += 1
            datasets.append({
                'name': data_path.stem,
                'path': data_dir,
                'gen': Generator(data_dir, self.class_type, False, self.m1_labels, True)
            })

        print("[HModelTrainer]", file_count, "dataset(s) found in >", data_dir)
        return datasets

    def _model_init(self, batch_n, is_training):
        return LSTModel({
            'features': tf.placeholder(tf.float32, name="features", shape=[
                batch_n, self.hyperparams['netw_sequence'], self.hyperparams['host_sequence'], self.features_len]),
            'labels': tf.placeholder(tf.int32, name="labels", shape=batch_n if self.m1_labels else [
                batch_n, self.hyperparams['netw_sequence']]),
            'netw_sequence': tf.placeholder(tf.int32, name="netw_sequence", shape=batch_n),
            'host_sequence': tf.placeholder(tf.int32, name="host_sequence", shape=[
                batch_n, self.hyperparams['netw_sequence']])
        }, self.hyperparams, batch_n, self.features_len, len(self.y_dict), self.m1_labels, self.seed_value, is_training)

    def _dataset_prep(self, generator, batch_size):

        dataset = tf.data.Dataset.from_generator(
            generator,
            output_types=(tf.float32, tf.int32, tf.int32, tf.int32),
            output_shapes=(
                tf.TensorShape(
                    [self.hyperparams['netw_sequence'], self.hyperparams['host_sequence'], self.features_len]),  # x
                tf.TensorShape([] if self.m1_labels else [self.hyperparams['sequence_max_n']]),  # y
                tf.TensorShape([]),  # network-level seq
                tf.TensorShape([self.hyperparams['netw_sequence']])  # host-level seq
            )
        )

        dataset = dataset.batch(batch_size, drop_remainder=True)
        # dataset = dataset.padded_batch(self.hyperparams['batch_n'], padded_shapes=[])
        dataset = dataset.prefetch(buffer_size=batch_size)

        return dataset.make_one_shot_iterator().get_next()

    def _validate_func(self, sess, model, testsets, batch_n, log_output):
        final_loss = 0

        for testset in testsets:
            next_element = self._dataset_prep(testset['gen'], batch_n)
            t = time.time()

            predictions = []
            ground_truth = []
            acc_test = 0
            loss_test = 0
            step_test = 0

            try:
                while True:
                    features_batched, label_batched, netw_seq_batched, host_seq_batched = sess.run(next_element)

                    _, truth, pred, loss, acc = sess.run(
                        model.error,
                        feed_dict={
                            model.features_placeholder: features_batched,
                            model.label_placeholder: label_batched,
                            model.netw_seq_placeholder: netw_seq_batched,
                            model.host_seq_placeholder: host_seq_batched
                        }
                    )

                    predictions.extend(pred)
                    ground_truth.extend(truth)

                    acc_test += acc
                    loss_test += loss
                    step_test += 1

            except tf.errors.OutOfRangeError:
                pass

            final_acc = round(acc_test / step_test, 9)
            final_loss = loss_test / step_test
            cm = confusion_matrix(ground_truth, predictions, labels=[label for label in range(len(self.y_dict))])

            if log_output:
                for row in cm:
                    logging.info(" ".join(str(col) for col in row))
                logging.info("[HModelTrainer] validation loss: %.9f" % round(
                    final_loss, 9) + "  acc: %.9f  time: " % final_acc + time.strftime(
                    "%H:%M:%S", time.gmtime(time.time() - t)))
            else:
                for row in cm:
                    print(" ".join(str(col) for col in row))
                print("[HModelTrainer] validation loss: %.9f" % round(
                    final_loss, 9) + "  acc: %.9f  time: " % final_acc + time.strftime(
                    "%H:%M:%S", time.gmtime(time.time() - t)))

        return final_loss

    def validate(self, train_dir, test_dir):
        np.random.seed(self.seed_value)
        tf.set_random_seed(self.seed_value)

        trainsets = self._get_files(train_dir)
        testsets = self._get_files(test_dir)

        model_test = self._model_init(self.batch_n_test, False)
        saver = tf.train.Saver()

        with tf.Session() as sess:
            if self.checkpoint_dir:
                saver.restore(sess, self.checkpoint_dir + trainsets[0]['name'] + self.model_name)
                print("[HModelTrainer] model restored for validation >",
                      self.checkpoint_dir + trainsets[0]['name'] + self.model_name)

                self._validate(sess, model_test, testsets, self.batch_n_test, False)

    def _output_func(self, sess, model, dataset, batch_n, is_trainset=True, log_output=False):
        test_name = "train" if is_trainset else "test"
        output_name = dataset['name']
        output_path = self.save_output + ("/train/" if is_trainset else "/dev/")  # gen Dev / Test
        pathlib.Path(output_path).mkdir(parents=True, exist_ok=True)

        # ARFF header creation
        arff_w_last = open(output_path + output_name + self.model_name + "_" + str(self.hyperparams['layers_n']) +
                           "_last.arff", 'wb')

        header_last = np.array(["@relation " + output_name + self.model_name + "_last", ""])

        for n in range(1, self.features_len+1):  # original data
            header_last = np.append(header_last, ("@attribute '" + str(n) + "' numeric"))

        for layer_n in range(1, self.hyperparams['layers_n']+1):
            for n in range(self.hyperparams['netw_output_n']):
                header_last = np.append(header_last, ("@attribute '" + str(layer_n) + "_" + str(n) + "' numeric"))
        header_label = "@attribute 'y' {" + ','.join(
            str(y) for y in range(len(self.y_dict))) + "}"
        header_last = np.append(header_last, header_label)
        header_last = np.append(header_last, ("", "@data"))

        np.savetxt(arff_w_last, header_last[np.newaxis].T, fmt='%s')

        next_element = self._dataset_prep(dataset['gen'], batch_n)
        t = time.time()

        predictions = []
        ground_truth = []
        acc_test = 0
        loss_test = 0
        step_test = 0

        try:
            while True:
                features_batched, label_batched, netw_seq_batched = sess.run(next_element)
                # features_batched, label_batched, netw_seq_batched, label_extra = sess.run(next_element)  # y1

                output, truth, pred, loss, acc = sess.run(
                    model.error,
                    feed_dict={
                        model.features_placeholder: features_batched,
                        model.label_placeholder: label_batched,
                        model.netw_seq_placeholder: netw_seq_batched
                    }
                )

                predictions.extend(pred)
                ground_truth.extend(truth)

                acc_test += acc
                loss_test += loss
                step_test += 1

                # noinspection PyTypeChecker
                np.savetxt(
                    arff_w_last, np.append(
                        np.append(
                            features_batched[np.arange(features_batched.shape[0]), netw_seq_batched - 1],
                            output, axis=1
                        ), truth[np.newaxis].T, axis=1
                    ),
                    #     np.concatenate([s.h for s in state], axis=1), label_extra[np.newaxis].T, axis=1),  # y1
                    fmt="%.18e," * (self.hyperparams['units_n']) * self.hyperparams['layers_n'] +
                        "%.18e," * self.features_len + "%i")

        except tf.errors.OutOfRangeError:
            pass

        arff_w_last.close()

        if log_output:
            logging.info("[ModelTrainer] vectors > " + output_path + output_name + self.model_name)
        else:
            print("[ModelTrainer] vectors > " + output_path + output_name + self.model_name)

        final_acc = round(acc_test / step_test, 9)
        final_loss = loss_test / step_test
        cm = confusion_matrix(ground_truth, predictions, labels=[label for label in range(len(self.y_dict))])

        if log_output:
            for row in cm:
                logging.info(" ".join(str(col) for col in row))
            logging.info("[ModelTrainer] " + test_name + " loss: %.9f" % round(
                final_loss, 9) + "  acc: %.9f  time: " % final_acc + time.strftime(
                "%H:%M:%S", time.gmtime(time.time() - t)))
        else:
            for row in cm:
                print(" ".join(str(col) for col in row))
            print("[ModelTrainer] " + test_name + " loss: %.9f" % round(
                final_loss, 9) + "  acc: %.9f  time: " % final_acc + time.strftime(
                "%H:%M:%S", time.gmtime(time.time() - t)))

        return final_loss

    def _train_func(self, sess, next_element, summary_writer):
        loss_train = 0
        acc_train = 0
        step_train = 0

        try:
            while True:
                features_batched, label_batched, netw_seq_batched, host_seq_batched = sess.run(next_element)
                (_, step_summaries, step), (_, _, _, loss, acc) = sess.run(  # error prior backpropagation
                    [self.model_train.optimize, self.model_train.error],
                    feed_dict={
                        self.model_train.features_placeholder: features_batched,
                        self.model_train.label_placeholder: label_batched,
                        self.model_train.netw_seq_placeholder: netw_seq_batched,
                        self.model_train.host_seq_placeholder: host_seq_batched
                    }
                )
                summary_writer.add_summary(step_summaries, global_step=step)

                loss_train += loss
                acc_train += acc
                step_train += 1

        except tf.errors.OutOfRangeError:
            return (loss_train / step_train), (acc_train / step_train)

    def train(self, train_dir, test_dir, dev_dir=None):
        np.random.seed(self.seed_value)
        tf.set_random_seed(self.seed_value)

        trainsets = self._get_files(train_dir)
        devsets = self._get_files(dev_dir)
        testsets = self._get_files(test_dir)
        pathlib.Path(self.saver_dir).mkdir(parents=True, exist_ok=True)

        epoch_current = 1
        dev_loss_prev = 999999999.
        tolerance = self.hyperparams['e.stopping'] if self.hyperparams['e.stopping'] else 0

        # ========== Model Restoration Check ==========
        if self.checkpoint_dir:
            if os.path.isfile(self.checkpoint_dir + trainsets[0]['name'] + self.model_name + ".log"):
                with open(self.checkpoint_dir + trainsets[0]['name'] + self.model_name + ".log", "r") as log_r:

                    dev_loss_checkpoint = 999999999.
                    for line in log_r:
                        log_output = line.split()

                        if log_output[0] == "[HModelTrainer]":
                            if log_output[1] == "validation":
                                dev_loss_checkpoint = float(log_output[3])
                            elif log_output[-1] == "saved":
                                dev_loss_prev = dev_loss_checkpoint
                                epoch_current = int(log_output[2]) + 1
                            elif log_output[1] == "e.stop,":
                                return True

                    if dev_loss_prev >= 999999999.:
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
        for testset in testsets:
            logging.info("[ModelTrainer] Test Set instances: " + str(testset['gen'].get_instances()))
            if testset['gen'].get_instances() % self.batch_n_test != 0:
                logging.info("[ModelTrainer] [WARNING] Testing batch size (" +
                             str(self.batch_n_test) + ") is not tally")
        for devset in devsets:
            logging.info("[ModelTrainer] Test Set instances: " + str(devset['gen'].get_instances()))
            if devset['gen'].get_instances() % self.batch_n_dev != 0:
                logging.info("[ModelTrainer] [WARNING] Validation batch size (" +
                             str(self.batch_n_dev) + ") is not tally")

        with tf.Session() as sess:
            # Model Initialization
            self.model_train = self._model_init(self.hyperparams['batch_n'], True)

            # TF Summary writer
            writer_train = tf.summary.FileWriter(
                (self.saver_dir + trainsets[0]['name'] + self.model_name + "/train"), sess.graph
            )
            writer_dev = tf.summary.FileWriter(
                (self.saver_dir + trainsets[0]['name'] + self.model_name + "/val"), sess.graph
            )

            # Model Initialization (dev and test)
            model_dev = self._model_init(self.batch_n_dev, False)
            model_test = self._model_init(self.batch_n_test, False)

            # Model & Weights Restoration (if necessary)
            saver = tf.train.Saver()

            # Model & Weights Restoration (if necessary)
            if self.checkpoint_dir or dev_loss_prev < 999999999.:
                saver.restore(sess, self.checkpoint_dir + trainsets[0]['name'] + self.model_name)
                logging.info("[HModelTrainer] model restored,  epoch: " + str(epoch_current - 1))

                self._validate(sess, model_test, testsets, self.batch_n_test, True)
            else:
                sess.run(tf.global_variables_initializer())

            # ========== Model Training ==========
            for epoch_n in range(epoch_current, self.hyperparams['epochs_n'] + 1):
                t = time.time()
                train_loss, acc_epoch = self._train(
                    sess, self._dataset_prep(trainsets[0]['gen'], self.hyperparams['batch_n']), writer_train)
                t_train = time.time() - t

                writer_train.add_summary(
                    tf.Summary(value=[tf.Summary.Value(tag="loss", simple_value=train_loss)]),
                    global_step=epoch_n
                )

                dev_loss = self._validate(sess, model_dev, devsets, self.batch_n_dev, True)
                writer_dev.add_summary(
                    tf.Summary(value=[tf.Summary.Value(tag="loss", simple_value=dev_loss)]),
                    global_step=epoch_n
                )

                # ========== Saver & Early Stopping ==========
                if self.saver_dir and (not self.hyperparams['e.stopping'] or (
                        self.hyperparams['e.stopping'] and dev_loss < dev_loss_prev)):
                        # and epoch_n >= 10  # look at weights only after 19th epochs

                    saver.save(sess, self.saver_dir + trainsets[0]['name'] + self.model_name)
                    logging.info("[HModelTrainer] epoch: " + str(epoch_n) + "  loss: %.9f" % round(
                        train_loss, 9) + "  acc: %.9f" % round(acc_epoch, 9) + "  time: " + time.strftime(
                        "%H:%M:%S", time.gmtime(t_train)) + "  saved")
                    dev_loss_prev = dev_loss
                    tolerance = self.hyperparams['e.stopping'] if self.hyperparams['e.stopping'] else 0

                elif not self.saver_dir or self.hyperparams['e.stopping']:
                    tolerance -= 1

                    logging.info("[HModelTrainer] epoch: " + str(epoch_n) + "  loss: %.9f" % round(
                        train_loss, 9) + "  acc: %.9f" % round(acc_epoch, 9) + "  time: " + time.strftime(
                        "%H:%M:%S", time.gmtime(t_train)))

                    if self.hyperparams['e.stopping'] and tolerance <= 0:
                        # evaluate last model
                        saver.save(sess, self.saver_dir + trainsets[0]['name'] + self.model_name + "_lastepoch")
                        self._validate(sess, model_test, testsets, self.batch_n_test, True)

                        # evaluate best model
                        logging.info("[HModelTrainer] e.stop,  epoch: " + str(epoch_n - self.hyperparams['e.stopping']))

                        saver.restore(sess, self.saver_dir + trainsets[0]['name'] + self.model_name)
                        logging.info("[ModelTrainer] restored model > " + trainsets[0]['name'] + self.model_name)

                        self._validate(sess, model_test, testsets, self.batch_n_test, True)
                        break

                # if not (epoch_n % self.hyperparams['calc_test']):
                #     test_loss = self._validate(sess, self._model_init(
                #         self.batch_n_test, False), testsets, self.batch_n_test, True)
                #     writer_test.add_summary(
                #         tf.Summary(value=[tf.Summary.Value(tag="loss", simple_value=test_loss)]),
                #         global_step=epoch_n
                #     )
                # ========== Saver & Early Stopping ==========

            writer_train.close()
            writer_dev.close()

            # max. epoch reached, save output from last checkpoint
            if self.hyperparams['e.stopping'] and tolerance > 0:
                logging.info("[HModelTrainer] maximum epoch")

                saver.restore(sess, self.saver_dir + trainsets[0]['name'] + self.model_name)
                logging.info("[ModelTrainer] restored model > " + trainsets[0]['name'] + self.model_name)

                self._validate(sess, model_test, testsets, self.batch_n_test, True)
            # ========== Model Training ==========

        logging.getLogger().removeHandler(log_file)
        logging.getLogger().removeHandler(log_console)

        return True
