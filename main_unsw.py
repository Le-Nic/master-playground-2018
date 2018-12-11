from preprocesshandler.preprocess import PreProcessing
from segregationhandler.timesegt import TimeSegregation
from segregationhandler.winsegt import WindowSegregation
from inputhandler.dataset_shuffler import DatasetShuffler
from modelhandler.modeltrainer import ModelTrainer
import numpy as np
import re


''' Features '''
# timestamp of the end of a flow (te)
# duration of flow(td)
# source IP address (sa)
# destination IP (da)
# source port (sp)
# destination port (dp)
# protocol (pr)
# flags (flg)
# forwarding status (fwd)
# type of service (stos)
# packets exchanged in the flow (pkt)
# number of bytes (byt)

if __name__ == '__main__':

    ''' HD5 Conversion (step 1) '''

    with open("configs/unsw_splits.txt", 'r') as config_file:
        pp_config = eval(config_file.read())

    # pp = PreProcessing(pp_config['convert_hd5'])
    # pp.get_metadata()
    # pp.save_metadata("F:/data/UNSW_splits/meta", name="1_mappings")
    # pp.transform_trainset()

    ''' Win/Time IP Segregation (step 2) '''
    flowsgt_config = {
        'input_dir': "F:/data/UNSW_splits/1_converted/UNSW_NB15_testing-set.hd5",
        'output_dir': "F:/data/UNSW_splits/2_winsgt",
        'features_len': "F:/data/UNSW_splits/meta/1_mappings.hd5",
        'meta_output_name': "2_mappings"
    }

    # winsgt = WindowSegregation(flowsgt_config, sequence_max=4, ip_segt=False, stride=4,
    #                            single_output=True, const_sequence=True)
    # winsgt.window_segregate()
    # winsgt.close()

    # timesgt = TimeSegregation(flowsgt_config, time_window=10, time_out=60, sequence_max=4,
    #                           bidirectional=True, single_output=True)
    # timesgt.time_segregate()

    ''' Data Shuffling (step 3) '''
    shuffle_config = {
        'input_path': "F:/data/UNSW_splits/2_winsgt/UNSW_NB15_combined-set_winsgt4s4.hd5",
        'output_dir': "F:/data/UNSW_splits/3_shuffled/5-folds",
        'meta_path': "F:/data/UNSW_splits/meta/1_mappings.hd5",
        # 'meta_path': "F:/data/UNSW_splits/meta/2_mappings_timesgt4.hd5",
    }

    # hd5huffler = DatasetShuffler(shuffle_config)
    # hd5huffler.shuffle(n_splits=5, test_size=0.1)

    ''' Features Preprocessing (step 4) '''

    # pp = PreProcessing(pp_config['process_hd5'])  # HD5 for tensorflow
    # pp.get_metadata()
    # pp.save_metadata("F:/data/UNSW_splits/meta", name="4_mappings_winsgt4s4")
    # pp.transform_trainset()
    # pp.transform_testset()
    # pp.close()

    ''' Model Training (step 4) '''

    saver_dir = "F:/data/UNSW_splits/checkpoints"
    checkpoint_dir = "F:/data/UNSW_splits/checkpoints"

    metas_directory = "F:/data/UNSW_splits/meta"
    meta_name = "/4_mappings"

    datasets_directory = "F:/data/UNSW_splits/4_processed"
    trainset_name = "/UNSW_NB15_training-set"
    devset_name = "/UNSW_NB15_testing-set"

    dataset_types = {  # loop: every key and value in array
        "/winsgt4s1": ["_winsgt4s1"],  # Normal Strides
        # "/winsgt_ip": ["_winsgt4s1_ip"]  # IP Segt
    }
    batch_n_tests = [
        # 5787,  # 1140039 instances (/winsgt4s1)
        # 7349  # 1139095 instances (/winsgt4s1_ip)
        3438  # 175338 instances splits (/winsgt4s1)

        # 3985  # 43835 instances (/winsgt4s4)
        # 3221  # 6442 instances (/winsgt4s4) 1st fold
    ]

    general_configs = {  # loop: every key and value in array
        'class_type': [0]  # 0: 2-class, 1: 3-class, 2: 5-class, 3: 9-class
    }

    hyperparams_configs = [{  # loop: every hyperparameters pair
        'units_n': 32,
        'batch_n': 64
    }, {
        'units_n': 64,
        'batch_n': 64
    }, {
        'units_n': 64,
        'batch_n': 128
    }, {
        'units_n': 128,
        'batch_n': 256
    }, {
        'units_n': 64,
        'batch_n': 256
    }, {
       'units_n': 128,
       'batch_n': 512
    }, {
       'units_n': 256,
       'batch_n': 512
    }
    ]

    for batch_n, dataset_dict in enumerate(dataset_types.items()):  # ip / normal
        for segt_type in dataset_dict[1]:  # sequences

            model_configs = {  # default values
                'class_type': 1,
                'm1_labels': True,
                'batch_n_test': batch_n_tests[batch_n],
                'hyperparameters': {
                    'sequence_max_n': int(re.findall(r"\d+", segt_type)[0]),  # extract first int from segt_type
                    'batch_n': 32,
                    'epochs_n': 200,
                    'units_n': 128,
                    'layers_n': 1,
                    'dropout_r': 0.4,  # 0 when performing tests
                    'learning_r': 0.01,
                    'decay_r': 0.96
                }
            }

            for general_config, general_values in general_configs.items():  #
                for value in general_values:  # binary / multi-class or m:1 / m:n labeling strategy
                    model_configs[general_config] = value

                    for hyperparams in hyperparams_configs:  # hyperparameters
                        for hyperparam, hyperparam_value in hyperparams.items():
                            model_configs['hyperparameters'][hyperparam] = hyperparam_value
                        #
                        myModel = ModelTrainer(
                            model_configs,
                            metas_directory + meta_name + segt_type + ".hd5",
                            checkpoint_dir,
                            saver_dir
                        )

                        # myModel.train(
                        #     datasets_directory + dataset_dict[0] + trainset_name + segt_type + ".hd5",
                        #     datasets_directory + dataset_dict[0] + devset_name + segt_type + ".hd5"
                        # )

                        # myModel.validate(
                        #     datasets_directory + dataset_dict[0] + trainset_name + segt_type + ".hd5",
                        #     datasets_directory + dataset_dict[0] + devset_name + segt_type + ".hd5"
                        # )

                        myModel.gen_output(
                            datasets_directory + dataset_dict[0] + trainset_name + segt_type + ".hd5",
                            datasets_directory + dataset_dict[0] + devset_name + segt_type + ".hd5", True
                        )
