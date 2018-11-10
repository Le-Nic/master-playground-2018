from modelhandler.lstmodel_seq2seq import *
from modelhandler.inputgenerator import Generator

from sklearn.metrics import confusion_matrix
import tensorflow as tf
import numpy as np
import tables as tb
import os
import pathlib
import time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class ModelTrainer(object):
    def __init__(self, configs, dataset_meta, checkpoint_dir=None, saver_dir=None):
        self.seed_value = 147

        self.hyperparams = configs['hyperparameters']
        self.class_type = configs['class_type']
        self.seq_const = configs['seq_constant']
        self.batch_n_test = configs['batch_n_test']

        self.checkpoint_dir = checkpoint_dir + "/" if checkpoint_dir is not None else None
        self.saver_dir = saver_dir + "/" if saver_dir is not None else None
        self.model_name = "_s" + str(self.hyperparams['sequence_max_n']) + \
                          "b" + str(self.hyperparams['batch_n']) + \
                          "u" + str(self.hyperparams['units_n']) + \
                          "l" + str(self.hyperparams['layers_n']) + \
                          "d" + str(int(self.hyperparams['dropout_r'] * 100)) + \
                          "y" + str(self.class_type)

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

            print("[ModelTrainer] Features length:", self.features_len)
            print("[ModelTrainer] Labels:", ', '.join(['{0} - {1}'.format(k, v.decode("utf-8"))
                                                       for k, v in self.y_dict.items()]))

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
                        'gen': Generator(str(child), self.class_type, False if self.seq_const else True)
                    })
                    file_count += 1

        elif data_path.is_file():
            file_count += 1
            datasets.append({
                'name': data_path.stem,
                'path': data_dir,
                'gen': Generator(data_dir, self.class_type, False if self.seq_const else True)
            })

        print("[ModelTrainer]", file_count, "dataset(s) found in >", data_dir)
        return datasets

    def _dataset_prep(self, generator, batch_size):

        dataset = tf.data.Dataset.from_generator(
            generator,
            output_types=(tf.float32, tf.int32) if self.seq_const else (tf.float32, tf.int32, tf.int32),
            output_shapes=(
                tf.TensorShape([self.hyperparams['sequence_max_n'], self.features_len]),
                tf.TensorShape([self.hyperparams['sequence_max_n']])
            ) if self.seq_const else (
                tf.TensorShape([self.hyperparams['sequence_max_n'], self.features_len]),
                tf.TensorShape([self.hyperparams['sequence_max_n']]),
                tf.TensorShape([])
            )
        )
        dataset = dataset.batch(batch_size, drop_remainder=True)
        # dataset = dataset.padded_batch(self.hyperparams['batch_n'], padded_shapes=[])
        dataset = dataset.prefetch(buffer_size=batch_size)

        return dataset.make_one_shot_iterator().get_next()

    def validate(self, sess, devsets):
        predictions = []
        ground_truth = []
        acc_dev = 0
        step_dev = 0

        for devset in devsets:
            next_element = self._dataset_prep(devset['gen'], self.batch_n_test)

            try:
                while True:
                    features_batched, label_batched, seq_batched = sess.run(next_element)

                    truth, pred, acc = sess.run(
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

        return confusion_matrix(ground_truth, predictions,
                                labels=[label for label in range(len(self.y_dict))]), round(acc_dev / step_dev, 6)

    def train(self, train_dir, dev_dir):
        np.random.seed = self.seed_value
        tf.set_random_seed(self.seed_value)

        trainsets = self._get_files(train_dir)
        devsets = self._get_files(dev_dir)

        for trainset in trainsets:
            print("[ModelTrainer] Train Set instances:", trainset['gen'].get_instances())
        for devset in devsets:
            print("[ModelTrainer] Dev Set instances:", devset['gen'].get_instances())

        if self.seq_const:
            self.model_train = LSTModel({
                'features': tf.placeholder(tf.float32, name="features", shape=[self.hyperparams['batch_n'],
                                                                               self.hyperparams['sequence_max_n'],
                                                                               self.features_len]),
                'labels': tf.placeholder(tf.int32, name="labels", shape=[self.hyperparams['batch_n'],
                                                                         self.hyperparams['sequence_max_n']])
            }, self.hyperparams, self.features_len, len(self.y_dict), self.seed_value, True)

            self.model_dev = LSTModel({
                'features': tf.placeholder(tf.float32, name="features", shape=[self.batch_n_test,
                                                                               self.hyperparams['sequence_max_n'],
                                                                               self.features_len]),
                'labels': tf.placeholder(tf.int32, name="labels", shape=[self.batch_n_test,
                                                                         self.hyperparams['sequence_max_n']])
            }, self.hyperparams, self.features_len, len(self.y_dict), self.seed_value, False)

        else:
            self.model_train = LSTModel({
                'features': tf.placeholder(tf.float32, name="features", shape=[self.hyperparams['batch_n'],
                                                                               self.hyperparams['sequence_max_n'],
                                                                               self.features_len]),
                'labels': tf.placeholder(tf.int32, name="labels", shape=[self.hyperparams['batch_n'],
                                                                         self.hyperparams['sequence_max_n']]),
                'sequences': tf.placeholder(tf.int32, name="sequences", shape=self.hyperparams['batch_n'])
                # 'states': tf.placeholder(
                #     tf.float32,
                #     shape=[self.hyperparams['layers_n'], 2, self.hyperparams['batch_n'], self.hyperparams['units_n']]
                # ),  # passing state to next batch
            }, self.hyperparams, self.features_len, len(self.y_dict), self.seed_value, True)

            self.model_dev = LSTModel({
                'features': tf.placeholder(tf.float32, name="features", shape=[self.batch_n_test,
                                                                               self.hyperparams['sequence_max_n'],
                                                                               self.features_len]),
                'labels': tf.placeholder(tf.int32, name="labels", shape=[self.batch_n_test,
                                                                         self.hyperparams['sequence_max_n']]),
                'sequences': tf.placeholder(tf.int32, name="sequences", shape=self.batch_n_test)
                # 'states': tf.placeholder(
                #     tf.float32,
                #     shape=[self.hyperparams['layers_n'], 2, self.hyperparams['batch_n'], self.hyperparams['units_n']]
                # ),  # passing state to next batch
            }, self.hyperparams, self.features_len, len(self.y_dict), self.seed_value, False)

        saver = tf.train.Saver()

        with tf.Session() as sess:
            # summary_writer = tf.summary.FileWriter('./logs', graph_def=sess.graph_def)  # tensorboard

            epoch_current = 0

            if self.checkpoint_dir:
                saver.restore(sess, self.checkpoint_dir + trainsets[0]['name'] + self.model_name + ".ckpt")
                with open(self.checkpoint_dir + "epoch", "r") as f:
                    epoch_current = int(f.read())

                print("[ModelTrainer] Model restored")
                print(self.validate(sess, devsets))
            else:
                sess.run(tf.global_variables_initializer())

            for epoch_n in range(epoch_current, self.hyperparams['epochs_n']):
                t = time.time()
                loss_train = 0
                acc_train = 0
                step_train = 0

                # merged_summary = tf.summary.merge_all()  # tensorboard

                for trainset in trainsets:
                    # state_current = np.zeros((self.hyperparams['layers_n'], 2,
                    #                           self.hyperparams['batch_n'], self.hyperparams['units_n']))
                    next_element = self._dataset_prep(trainset['gen'], self.hyperparams['batch_n'])

                    try:
                        while True:
                            features_batched, label_batched, seq_batched = sess.run(next_element)

                            (_, loss), (_, _, acc) = sess.run(  # error prior backpropagation
                                [self.model_train.optimize, self.model_train.error],
                                feed_dict={
                                    self.model_train.features_placeholder: features_batched,
                                    self.model_train.label_placeholder: label_batched,
                                    self.model_train.seq_placeholder: seq_batched
                                    # self.model.state_placeholder: state_current
                                }
                            )

                            loss_train += loss
                            acc_train += acc
                            step_train += 1

                    except tf.errors.OutOfRangeError:
                        pass

                # epoch_n = sess.run(self.model_train.add_global_step)
                if self.saver_dir:
                    saver.save(sess, self.saver_dir + trainsets[0]['name'] + self.model_name)
                    with open(self.saver_dir + "epoch", 'w+') as f:
                        f.write(str(epoch_n + 1))

                print("[ModelTrainer] Epoch " + str(epoch_n + 1) +
                      ", loss: %.6f" % round(loss_train / step_train, 6) +
                      ", acc: %.6f" % round(acc_train / step_train, 6) +
                      ", time elapsed: " + time.strftime("%H:%M:%S", time.gmtime(time.time() - t)))

                if not ((epoch_n + 1) % 5):
                    cm, acc = self.validate(sess, devsets)

                    print("[ModelTrainer] acc: %.6f, time elapsed:" % acc,
                          time.strftime("%H:%M:%S", time.gmtime(time.time() - t)))
                    for row in cm:
                        print(" ".join(str(col) for col in row))

