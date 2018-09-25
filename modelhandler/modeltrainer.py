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

        # placeholder init.
        features_placeholder = tf.placeholder(tf.float32, shape=[self.hyperparams['batch_n'],
                                                                 self.hyperparams['sequence_max_n'], self.features_len])
        label_placeholder = tf.placeholder(tf.int32, self.hyperparams['batch_n'])
        seq_placeholder = tf.placeholder(tf.int32, self.hyperparams['batch_n'])
        state_placeholder = tf.placeholder(tf.float32, [self.hyperparams['layers_n'], 2, self.hyperparams['batch_n'],
                                                        self.hyperparams['units_n']])

        self.model = LSTModel(
            features_placeholder, label_placeholder, seq_placeholder, state_placeholder,
            self.features_len, self.hyperparams['batch_n'], self.hyperparams['sequence_max_n'], len(self.y_dict),
            self.hyperparams['units_n'], self.hyperparams['layers_n'],
            self.hyperparams['learning_r'], self.hyperparams['decay_r'], self.seed_value)

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

    def _dataset_prep(self):
        """ assign Reader for each dataset found """
        dataset = tf.data.Dataset.from_tensor_slices((self.model.features_placeholder, self.model.label_placeholder,
                                                      self.model.seq_placeholder))
        dataset = dataset.batch(self.hyperparams['batch_n'])
        # dataset = dataset.padded_batch(self.hyperparams['batch_n'], padded_shapes=[])
        dataset = dataset.prefetch(buffer_size=self.hyperparams['batch_n'])

        return dataset

    def train(self, train_dir):
        trainsets = self._get_files(train_dir)

        for trainset in trainsets:
            print("[ModelTrainer] Training Set instances:", trainset['gen'].get_instances())

        with tf.Session() as sess:

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
                loss_total = 0
                error_total = 0
                step = 0

                for trainset in trainsets:

                    dataset = tf.data.Dataset.from_generator(
                        trainset['gen'],
                        output_types=(tf.float32, tf.int32, tf.int32),
                        output_shapes=(tf.TensorShape([self.hyperparams['sequence_max_n'], self.features_len]),
                                       tf.TensorShape([]), tf.TensorShape([]))
                    )
                    dataset = dataset.batch(self.hyperparams['batch_n'], drop_remainder=True)
                    dataset = dataset.prefetch(buffer_size=self.hyperparams['batch_n'])
                    next_element = dataset.make_one_shot_iterator().get_next()

                    state_current = np.zeros((self.hyperparams['layers_n'], 2,
                                              self.hyperparams['batch_n'], self.hyperparams['units_n']))

                    try:
                        while True:
                            features_batched, label_batched, seq_batched = sess.run(next_element)

                            (_, loss), error = sess.run([self.model.optimize, self.model.error], feed_dict={
                                self.model.features_placeholder: features_batched,
                                self.model.label_placeholder: label_batched,
                                self.model.seq_placeholder: seq_batched,
                                self.model.state_placeholder: state_current
                            })

                            loss_total += loss
                            error_total += error
                            step += 1

                    except tf.errors.OutOfRangeError:
                        epoch_n = sess.run(self.model.add_global_step)

                        print("[ModelTrainer] Epoch " + str(epoch_n) +
                              ", loss: %.6f" % round(loss_total/step, 6) +
                              ", error: %.6f" % round(error_total/step, 6) +
                              ", time elapsed: " + time.strftime("%H:%M:%S", time.gmtime(time.time() - t)))

