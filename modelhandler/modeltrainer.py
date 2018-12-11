from modelhandler.lstmodel import *
# from modelhandler.ntmodel import *
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
        self.m1_labels = configs['m1_labels']
        self.batch_n_test = configs['batch_n_test']

        self.checkpoint_dir = checkpoint_dir + "/" if checkpoint_dir is not None else None
        self.saver_dir = saver_dir + "/" if saver_dir is not None else ""
        self.model_name = "_s" + str(self.hyperparams['sequence_max_n']) + \
                          "u" + str(self.hyperparams['units_n']) + \
                          "b" + str(self.hyperparams['batch_n']) + \
                          "l" + str(self.hyperparams['layers_n']) + \
                          "d" + str(int(self.hyperparams['dropout_r']*100)) + \
                          "_y" + str(self.class_type)
        self.save_output = True

        print("[MT Config.]", "M:1" if self.m1_labels else "M:N", "labeling strategy")
        print("[MT Config.] Sequence:", self.hyperparams['sequence_max_n'])
        print("[MT Config.] Batch size:", self.hyperparams['batch_n'])
        print("[MT Config.] Epochs:", self.hyperparams['epochs_n'])
        print("[MT Config.] Hidden units:", self.hyperparams['units_n'])
        print("[MT Config.] Layer(s):", self.hyperparams['layers_n'])
        print("[MT Config.] Dropout rate:", self.hyperparams['dropout_r'])
        print("[MT Config.] Learning rate:", self.hyperparams['learning_r'])

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
                        'gen': Generator(str(child), self.class_type, self.m1_labels)
                    })
                    file_count += 1

        elif data_path.is_file():
            file_count += 1
            datasets.append({
                'name': data_path.stem,
                'path': data_dir,
                'gen': Generator(data_dir, self.class_type, self.m1_labels)
            })

        print("[ModelTrainer]", file_count, "dataset(s) found in >", data_dir)
        return datasets

    def _model_init(self, batch_n, is_training):

        return LSTModel({
            'features': tf.placeholder(tf.float32, name="features",
                                       shape=[batch_n, self.hyperparams['sequence_max_n'], self.features_len]),
            'labels': tf.placeholder(tf.int32, name="labels",
                                     shape=batch_n if self.m1_labels else [
                                         batch_n, self.hyperparams['sequence_max_n']]),
            'sequences': tf.placeholder(tf.int32, name="sequences", shape=batch_n)
            # 'states': tf.placeholder(
            #     tf.float32,
            #     shape=[self.hyperparams['layers_n'], 2, self.hyperparams['batch_n'], self.hyperparams['units_n']]
            # ),  # passing state to next batch
        }, self.hyperparams, self.features_len, len(self.y_dict), self.m1_labels, self.seed_value, is_training)

    def _dataset_prep(self, generator, batch_size):

        dataset = tf.data.Dataset.from_generator(
            generator,
            output_types=(tf.float32, tf.int32, tf.int32),
            output_shapes=(
                tf.TensorShape([self.hyperparams['sequence_max_n'], self.features_len]),
                tf.TensorShape([] if self.m1_labels else [self.hyperparams['sequence_max_n']]),
                tf.TensorShape([])
            )
        )
        dataset = dataset.batch(batch_size, drop_remainder=True)
        # dataset = dataset.padded_batch(self.hyperparams['batch_n'], padded_shapes=[])
        dataset = dataset.prefetch(buffer_size=batch_size)

        return dataset.make_one_shot_iterator().get_next()

    def _save_output(self, sess, saver, my_model, trainset, batch_n, devset=None):

        saver.restore(sess, self.saver_dir + trainset['name'] + self.model_name)
        logging.info("[ModelTrainer] restored model > " + trainset['name'] + self.model_name)

        output_name = trainset['name'] if devset is None else devset['name']
        output_path = "F:/data/UNSW_splits/5_output/train/" if devset is None else "F:/data/UNSW_splits/5_output/dev/"
        pathlib.Path(output_path).mkdir(parents=True, exist_ok=True)

        # ARFF header creation

        if self.m1_labels:  # save all outputs (valid for when m1_labels = True)
            arff_w_all = open(output_path + output_name + self.model_name + "_all.arff", 'wb')
        else:
            arff_w_all = None

        arff_w_last = open(output_path + output_name + self.model_name + "_last.arff", 'wb')

        header_all = np.array(["@relation " + output_name + self.model_name + "_all", ""])
        header_last = np.array(["@relation " + output_name + self.model_name + "_last", ""])

        for seq_n in range(self.hyperparams['sequence_max_n']):
            for n in range(self.hyperparams['units_n']):
                header_all = np.append(
                    header_all, ("@attribute 'T" + str(seq_n+1) + " " + str(n) + "' numeric"))

        for n in range(self.hyperparams['units_n']):
            header_last = np.append(header_last, ("@attribute '" + str(n) + "' numeric"))

        header_label = "@attribute 'y' {" + ','.join(
            str(y) for y in range(len(self.y_dict))) + "}"

        header_all = np.append(header_all, header_label)
        header_all = np.append(header_all, ("", "@data"))

        header_last = np.append(header_last, header_label)
        header_last = np.append(header_last, ("", "@data"))

        if self.m1_labels:
            np.savetxt(arff_w_all, header_all[np.newaxis].T, fmt='%s')
            arff_w_all.close()
            arff_w_all = open(output_path + output_name + self.model_name + "_all.arff", 'ab')

        np.savetxt(arff_w_last, header_last[np.newaxis].T, fmt='%s')
        arff_w_last.close()
        arff_w_last = open(output_path + output_name + self.model_name + "_last.arff", 'ab')

        predictions = []
        ground_truth = []
        acc_dev = 0
        step_dev = 0

        next_element = self._dataset_prep(trainset['gen'] if devset is None else devset['gen'], batch_n)

        try:
            while True:
                features_batched, label_batched, seq_batched = sess.run(next_element)

                (output, truth, pred, acc) = sess.run(  # error prior backpropagation
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
                    # noinspection PyTypeChecker
                    np.savetxt(  # output for all sequences
                        arff_w_all, np.append(
                            np.reshape(output, (batch_n, -1)), truth[np.newaxis].T, axis=1),
                        fmt="%.18e," * (self.hyperparams['units_n'] * self.hyperparams['sequence_max_n']) + "%i")

                    # noinspection PyTypeChecker
                    np.savetxt(  # output for last sequence
                        arff_w_last, np.append(output[..., -1, :], truth[np.newaxis].T, axis=1),
                        fmt="%.18e," * (self.hyperparams['units_n']) + "%i")
                else:
                    # noinspection PyTypeChecker
                    np.savetxt(  # output for all sequences
                        arff_w_last, np.append(
                            np.reshape(output, (batch_n * self.hyperparams['sequence_max_n'], -1)),
                            truth[np.newaxis].T, axis=1), fmt="%.18e," * self.hyperparams['units_n'] + "%i")

        except tf.errors.OutOfRangeError:
            pass

        arff_w_last.close()
        if self.m1_labels:
            arff_w_all.close()

        logging.info("[ModelTrainer] vectors > " + output_path + output_name + self.model_name)
        return confusion_matrix(ground_truth, predictions, labels=[
            label for label in range(len(self.y_dict))]), round(acc_dev / step_dev, 6)

    def _validate(self, sess, devsets):
        predictions = []
        ground_truth = []
        acc_dev = 0
        step_dev = 0

        for devset in devsets:
            next_element = self._dataset_prep(devset['gen'], self.batch_n_test)

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

        return confusion_matrix(ground_truth, predictions, labels=[
            label for label in range(len(self.y_dict))]), round(acc_dev / step_dev, 6)

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

                t = time.time()
                cm, acc = self._validate(sess, devsets)

                print("[ModelTrainer] acc: %.6f  time: " % acc +
                      time.strftime("%H:%M:%S", time.gmtime(time.time() - t)))
                for row in cm:
                    print(" ".join(str(col) for col in row))

    def train(self, train_dir, dev_dir):
        np.random.seed = self.seed_value
        tf.set_random_seed(self.seed_value)

        trainsets = self._get_files(train_dir)
        devsets = self._get_files(dev_dir)
        pathlib.Path(self.saver_dir).mkdir(parents=True, exist_ok=True)

        epoch_current = 1
        prev_acc_dev = .0
        rollback_counter = 4  # patient for e.stop

        # check if retraining is needed
        if self.checkpoint_dir:
            if os.path.isfile(self.checkpoint_dir + trainsets[0]['name'] + self.model_name + ".log"):
                with open(self.checkpoint_dir + trainsets[0]['name'] + self.model_name + ".log", "r") as log_r:

                    saved_count = 0
                    last_epoch = -1
                    for line in log_r:
                        log_output = line.split()
                        if log_output[0] == "[ModelTrainer]":
                            if log_output[1] == "vectors":  # indication for saved vectors
                                saved_count += 1
                            elif log_output[1] == "acc:":
                                if float(log_output[2]) >= prev_acc_dev:
                                    prev_acc_dev = float(log_output[2])
                                    epoch_current = last_epoch + 1
                            elif log_output[1] == "epoch:":
                                last_epoch = int(log_output[2])

                    if saved_count >= 2:
                        return True
                    if prev_acc_dev <= .0:
                        self.checkpoint_dir = None

            else:
                self.checkpoint_dir = None

        log_file = logging.FileHandler(self.saver_dir + trainsets[0]['name'] + self.model_name + ".log")
        log_console = logging.StreamHandler()
        logging.basicConfig(level=logging.INFO, format="%(message)s", handlers=[log_file, log_console])

        logging.info("---- ---- ---- ----")

        for trainset in trainsets:
            logging.info("[ModelTrainer] Train Set instances: " + str(trainset['gen'].get_instances()))
        for devset in devsets:
            logging.info("[ModelTrainer] Dev Set instances: " + str(devset['gen'].get_instances()))

        self.model_train = self._model_init(self.hyperparams['batch_n'], True)
        self.model_dev = self._model_init(self.batch_n_test, False)

        saver = tf.train.Saver()

        with tf.Session() as sess:
            # summary_writer = tf.summary.FileWriter('./logs', graph_def=sess.graph_def)  # tensorboard

            # h5_r = tb.open_file(trainset['path'], mode='r')
            # features = h5_r.get_node("/x")
            # labels = h5_r.get_node("/y/y" + str(self.class_type))
            # seqs = h5_r.get_node("/seq")
            # assert features.shape[0] == labels.shape[0] == seqs.shape[0]
            #
            # with tf.device('/cpu:0'):
            #     iterator = self._dataset_prep().make_initializable_iterator()
            #     next_element = iterator.get_next()
            #
            #     sess.run(iterator.initializer, feed_dict={  # (re)initialize iterator's state
            #         m.features_placeholder: features,
            #         m.labels_placeholder: labels,
            #         m.seqs_placeholder: seqs
            #     })

            # h5_r.close()

            if self.checkpoint_dir or prev_acc_dev > .0:  # retore best weights and epoch
                saver.restore(sess, self.checkpoint_dir + trainsets[0]['name'] + self.model_name)
                # with open(self.checkpoint_dir + "epoch", "r") as f:
                #     epoch_current = int(f.read()) + 1

                logging.info("[ModelTrainer] model restored,  epoch: " + str(epoch_current-1))
                t = time.time()
                cm, acc = self._validate(sess, devsets)

                logging.info("[ModelTrainer] acc: %.6f  time: " % acc +
                             time.strftime("%H:%M:%S", time.gmtime(time.time() - t)))
                for row in cm:
                    logging.info(" ".join(str(col) for col in row))

            else:
                sess.run(tf.global_variables_initializer())

            for epoch_n in range(epoch_current, self.hyperparams['epochs_n']+1):
                t = time.time()
                loss_train = 0
                acc_train = 0
                step_train = 0

                # merged_summary = tf.summary.merge_all()  # tensorboard

                # state_current = np.zeros((self.hyperparams['layers_n'], 2,
                #                           self.hyperparams['batch_n'], self.hyperparams['units_n']))
                next_element = self._dataset_prep(trainsets[0]['gen'], self.hyperparams['batch_n'])

                try:
                    while True:
                        features_batched, label_batched, seq_batched = sess.run(next_element)
                        # features_batched, label_batched = sess.run(next_element)

                        (_, loss, labels), (output, truth, _, acc) = sess.run(  # error prior backpropagation
                            [self.model_train.optimize, self.model_train.error],
                            feed_dict={
                                self.model_train.features_placeholder: features_batched,
                                self.model_train.label_placeholder: label_batched,
                                self.model_train.seq_placeholder: seq_batched
                                # self.model.state_placeholder: state_current
                            }
                        )

                        # summary_writer.add_summary(summary_str, step_train)  # tensorboard

                        # # retrieve trainable variables
                        # trainable_vars_dict = {}
                        # for k in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
                        #     trainable_vars_dict[k.name] = sess.run(k)
                        # lstm_weight_vals = trainable_vars_dict[
                        #     "prediction/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel:0"]
                        # w_i, w_C, w_f, w_o = np.split(lstm_weight_vals, 4, axis=1)

                        loss_train += loss
                        acc_train += acc
                        step_train += 1

                except tf.errors.OutOfRangeError:
                    pass

                logging.info("[ModelTrainer] epoch: " + str(epoch_n) +
                             "  loss: %.6f" % round(loss_train / step_train, 6) +
                             "  acc: %.6f" % round(acc_train / step_train, 6) +
                             "  time: " + time.strftime("%H:%M:%S", time.gmtime(time.time() - t)))

                if not (epoch_n % 3):  # validation  every 3rd epoch

                    t = time.time()
                    cm, acc = self._validate(sess, devsets)
                    logging.info("[ModelTrainer] acc: %.6f  time: " % acc +
                                 time.strftime("%H:%M:%S", time.gmtime(time.time() - t)))
                    for row in cm:
                        logging.info(" ".join(str(col) for col in row))

                    if acc > prev_acc_dev:  # save/replace model if there's improvement
                        prev_acc_dev = acc
                        rollback_counter = 4  # patient for e.stop

                        # epoch_n = sess.run(self.model_train.add_global_step)
                        if self.saver_dir:
                            saver.save(sess, self.saver_dir + trainsets[0]['name'] + self.model_name)
                            with open(self.saver_dir + "epoch", 'w+') as f:
                                f.write(str(epoch_n))

                    else:
                        rollback_counter -= 1

                        if rollback_counter <= 0:  # early stopping criteria met: save output from last checkpoint
                            logging.info("[ModelTrainer] e.stop,  epoch: " + str(epoch_n - 12))  # patient for e.stop

                            if self.save_output:
                                t = time.time()
                                _, _ = self._save_output(sess, saver, self.model_train,
                                                         trainsets[0], self.hyperparams['batch_n'])
                                cm, acc = self._save_output(sess, saver, self.model_dev,
                                                            trainsets[0], self.batch_n_test, devsets[0])
                                logging.info("[ModelTrainer] acc: %.6f  time: " % acc +
                                             time.strftime("%H:%M:%S", time.gmtime(time.time() - t)))
                                for row in cm:
                                    logging.info(" ".join(str(col) for col in row))
                            break

            if rollback_counter > 0:  # max. epoch reached, save output from last checkpoint
                logging.info("[ModelTrainer] maximum epoch, restoring best model")

                if self.save_output:
                    t = time.time()

                    _, _ = self._save_output(sess, saver, self.model_train,
                                             trainsets[0], self.hyperparams['batch_n'])
                    cm, acc = self._save_output(sess, saver, self.model_dev,
                                                trainsets[0], self.batch_n_test, devsets[0])
                    logging.info("[ModelTrainer] acc: %.6f  time: " % acc +
                                 time.strftime("%H:%M:%S", time.gmtime(time.time() - t)))
                    for row in cm:
                        logging.info(" ".join(str(col) for col in row))

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

                t = time.time()
                if output_trainset:
                    _, _ = self._save_output(sess, saver, model_train, trainsets[0], self.hyperparams['batch_n'])

                cm, acc = self._save_output(sess, saver, model_dev, trainsets[0], self.batch_n_test, devsets[0])

                logging.info("[ModelTrainer] acc: %.6f  time: " % acc +
                             time.strftime("%H:%M:%S", time.gmtime(time.time() - t)))
                for row in cm:
                    logging.info(" ".join(str(col) for col in row))

            else:
                logging.info("[ModelTrainer] checkpoint directory not given")
