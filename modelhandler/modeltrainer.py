from modelhandler.lstmodel import *
from modelhandler.inputgenerator import Generator
import tensorflow as tf
import numpy as np
import tables as tb
import os
import pathlib
import time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class ModelTrainer(object):
    def __init__(self, configs, resume_checkpoint=False, meta_path=None):
        self.seed_value = 147

        self.hyperparams = configs['hyperparameters']
        self.resume = resume_checkpoint
        self.class_type = configs['class_type']

        print("[MT Config.] Sequence:", self.hyperparams['sequence_max_n'])
        print("[MT Config.] Batch size:", self.hyperparams['batch_n'])
        print("[MT Config.] Epochs:", self.hyperparams['epochs_n'])
        print("[MT Config.] Hidden units:", self.hyperparams['units_n'])
        print("[MT Config.] Layer(s):", self.hyperparams['layers_n'])
        print("[MT Config.] Dropout rate:", self.hyperparams['dropout_r'])
        print("[MT Config.] Learning rate:", self.hyperparams['learning_r'])

        if meta_path:
            meta_h5 = tb.open_file(meta_path, mode='r')
            self.y_dict = dict(enumerate(meta_h5.get_node("/y" + str(self.class_type)).read()))
            self.features_len = len(meta_h5.get_node("/x").read())

            print("[ModelTrainer] Features length:", self.features_len)
            print("[ModelTrainer] Labels:",  ', '.join(['{0} - {1}'.format(k, v.decode("utf-8"))
                                                        for k, v in self.y_dict.items()]))

        else:
            self.y_dict = None  # TODO: unknown labels and features length (nid ways to input this)
            self.features_len = None

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
                file_count += 1

                datasets.append({
                    'name': child.stem,
                    'path': str(child),
                    'gen': Generator(str(child), self.class_type)
                }) if pathlib.Path(child).is_file() else None

        elif data_path.is_file():
            file_count += 1
            datasets.append({
                'name': data_path.stem,
                'path': data_dir,
                'gen': Generator(data_dir, self.class_type)
            })

        print("[ModelTrainer]", file_count, "dataset(s) found in >", data_dir)
        return datasets

    def _dataset_prep(self, generator, batch_size):

        dataset = tf.data.Dataset.from_generator(
            generator,
            output_types=(tf.float32, tf.int32, tf.int32),
            output_shapes=(tf.TensorShape([self.hyperparams['sequence_max_n'], self.features_len]),
                           tf.TensorShape([]), tf.TensorShape([]))
        )
        dataset = dataset.batch(batch_size, drop_remainder=True)
        # dataset = dataset.padded_batch(self.hyperparams['batch_n'], padded_shapes=[])
        dataset = dataset.prefetch(buffer_size=batch_size)

        return dataset.make_one_shot_iterator().get_next()

    def train(self, train_dir, dev_dir):

        self.model_train = LSTModel(
            tf.placeholder(
                tf.float32, name="features",
                shape=[self.hyperparams['batch_n'], self.hyperparams['sequence_max_n'], self.features_len]
            ),
            tf.placeholder(tf.int32, name="labels", shape=self.hyperparams['batch_n']),
            tf.placeholder(tf.int32, name="sequences", shape=self.hyperparams['batch_n']),
            # tf.placeholder(
            #     tf.float32,
            #     shape=[self.hyperparams['layers_n'], 2, self.hyperparams['batch_n'], self.hyperparams['units_n']]
            # ),  # passing state to next batch
            self.hyperparams, self.features_len, len(self.y_dict), self.seed_value, True
        )
        trainsets = self._get_files(train_dir)

        for trainset in trainsets:
            print("[ModelTrainer] Training Set instances:", trainset['gen'].get_instances())

        self.model_dev = LSTModel(
            tf.placeholder(
                tf.float32, name="features",
                shape=[1, self.hyperparams['sequence_max_n'], self.features_len]
            ),
            tf.placeholder(tf.int32, name="labels", shape=1),
            tf.placeholder(tf.int32, name="sequences", shape=1),
            # tf.placeholder(
            #     tf.float32,
            #     shape=[self.hyperparams['layers_n'], 2, self.hyperparams['batch_n'], self.hyperparams['units_n']]
            # ),  # passing state to next batch
            self.hyperparams, self.features_len, len(self.y_dict), self.seed_value, False
        )
        devsets = self._get_files(dev_dir)

        for devset in devsets:
            print("[ModelTrainer] Testing Set instances:", devset['gen'].get_instances())

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

            tf.set_random_seed(self.seed_value)
            sess.run(tf.global_variables_initializer())

            for _ in range(self.hyperparams['epochs_n']):
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

                            (_, loss), (_, acc) = sess.run(  # error prior backpropagation
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

                epoch_n = sess.run(self.model_train.add_global_step)

                if not (epoch_n % 10):
                    predictions = []
                    ground_truth = []
                    acc_dev = 0
                    step_dev = 0

                    for devset in devsets:
                        next_element = self._dataset_prep(devset['gen'], 1)

                        try:
                            while True:
                                features_batched, label_batched, seq_batched = sess.run(next_element)
                                pred, acc = sess.run(
                                    self.model_dev.error,
                                    feed_dict={
                                        self.model_dev.features_placeholder: features_batched,
                                        self.model_dev.label_placeholder: label_batched,
                                        self.model_dev.seq_placeholder: seq_batched
                                    }
                                )
                                predictions.extend(pred)
                                ground_truth.extend(label_batched)

                                # if pred_positive[0]:
                                #     try:
                                #         class_postives[label_batched[0]] += 1
                                #     except KeyError:
                                #         class_postives[label_batched[0]] = 1
                                # else:
                                #     try:
                                #         class_negatives[label_batched[0]] += 1
                                #     except KeyError:
                                #         class_negatives[label_batched[0]] = 1

                                acc_dev += acc
                                step_dev += 1

                        except tf.errors.OutOfRangeError:
                            pass

                    print("[ModelTrainer] acc: %.6f, truth:" % round(acc_dev/step_dev, 6), ground_truth,
                          "predictions:", predictions)

                print("[ModelTrainer] Epoch " + str(epoch_n) +
                      ", loss: %.6f" % round(loss_train/step_train, 6) +
                      ", acc: %.6f" % round(acc_train/step_train, 6) +
                      ", time elapsed: " + time.strftime("%H:%M:%S", time.gmtime(time.time() - t)))

