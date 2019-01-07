from preprocesshandler.preprocess import PreProcessing
from segregationhandler.timesegt import TimeSegregation
from segregationhandler.winsegt import WindowSegregation
# from inputhandler.dataset_shuffler import DatasetShuffler
from modelhandler.modeltrainer import ModelTrainer
# from modelhandler.modeltrainer_stateful import ModelTrainer
import numpy as np
import re


if __name__ == '__main__':

    # # # # # # # # # #
    # # START  TUNE # #
    # # # # # # # # # #

    # Data Set

    dataset_dir = "F:/data/KDD99_10"
    trainset_name = "/kdd_train_10"
    devset_name = "/kdd_test"

    # Data Structure
    winsgt_type = "winsgt4s1"
    is_winsgt_const = "_const"  # empty string if not
    is_reversed = ""  # swap train and test set if not empty (2-fold validation): "" or "_re"
    is_ip = ""  # is IP segregated: "" or "_ip"

    # Model
    saver_dir = "/checkpoints"
    checkpoint_dir = "/checkpoints"

    # Preprocessing
    with open("configs/kdd99.txt", 'r') as config_file:
        pp_config = eval(config_file.read())

    # # # # # # # # # #
    # #  END  TUNE  # #
    # # # # # # # # # #

    # # # # # # # # # #
    # # START SETUP # #
    # # # # # # # # # #

    winsgt_dir = dataset_dir + "/2_" + winsgt_type + is_winsgt_const
    saver_dir = winsgt_dir + saver_dir + (is_reversed if len(is_reversed) else "")
    checkpoint_dir = winsgt_dir + checkpoint_dir + (is_reversed if len(is_reversed) else "")

    # a/pre-pend directory paths
    for key in ['train_dir', 'test_dir', 'output_dir']:
        if pp_config['convert_hd5']['io_csv'][key] is not None:
            pp_config['convert_hd5']['io_csv'][key] = dataset_dir + pp_config['convert_hd5']['io_csv'][key]

    # assuming only 1 file for each train/test set
    pp_config['process_hd5']['io_hd5']['train_dir'] = winsgt_dir + "/2_winsgt" + \
        (devset_name if len(is_reversed) else trainset_name) + "_" + winsgt_type + is_ip + ".hd5"

    pp_config['process_hd5']['io_hd5']['test_dir'] = winsgt_dir + "/2_winsgt" + \
        (trainset_name if len(is_reversed) else devset_name) + "_" + winsgt_type + is_ip + ".hd5"

    pp_config['process_hd5']['io_hd5']['output_dir'] = winsgt_dir + \
        pp_config['process_hd5']['io_hd5'][key] + is_reversed

    pp_config['process_hd5']['io_hd5']['meta_path'] = dataset_dir + \
        pp_config['process_hd5']['io_hd5']['meta_path'] + \
        ("/2_mappings_" + winsgt_type + "_ip" + is_winsgt_const + ".hd5" if is_ip else "/1_mappings.hd5")

    # # # # # # # # # #
    # #  END SETUP  # #
    # # # # # # # # # #

    ''' HD5 Conversion (step 1) '''
    # pp = PreProcessing(pp_config['convert_hd5'])
    # pp.get_metadata()
    # pp.save_metadata(dataset_dir + "/meta", name="1_mappings")
    # pp.transform_trainset()

    ''' Win/Time IP Segregation (step 2) '''
    flowsgt_config = {
        'input_dir': dataset_dir + "/1_converted",  # TUNE THIS (specific / whole dir)
        'output_dir': winsgt_dir + "/2_winsgt",
        'features_len': dataset_dir + "/meta/1_mappings.hd5",
        'meta_output_name': "2_mappings"
    }
    winsgt_seq = [int(s) for s in winsgt_type.split("winsgt")[1].split("s")]

    # winsgt = WindowSegregation(flowsgt_config, sequence_max=winsgt_seq[0], ip_segt=bool(is_ip), stride=winsgt_seq[1],
    #                            single_output=False, const_sequence=bool(is_winsgt_const))
    # winsgt.window_segregate()
    # winsgt.close()

    # timesgt = TimeSegregation(flowsgt_config, time_window=10, time_out=60, sequence_max=4,
    #                           bidirectional=True, single_output=True)
    # timesgt.time_segregate()

    ''' Data Shuffling (step 3) '''
    shuffle_config = {
        'input_path': winsgt_dir + "/2_winsgt/UNSW_NB15_combined-set_winsgt4s4.hd5",  # TUNE THIS (specific / whole dir)
        'output_dir': winsgt_dir + "/3_shuffled",
        'meta_path': dataset_dir + "/meta/1_mappings.hd5",  # TUNE THIS (without IPSegt)
        # 'meta_path': dataset_dir + "/meta/2_mappings_timesgt4.hd5",  # TUNE THIS (with IPSegt)
    }

    # hd5huffler = DatasetShuffler(shuffle_config)
    # hd5huffler.shuffle(n_splits=5, test_size=0.1)

    ''' Features Preprocessing (step 4) '''
    # pp = PreProcessing(pp_config['process_hd5'])  # HD5 for tensorflow
    # pp.get_metadata()
    # pp.save_metadata(dataset_dir + "/meta", name="4_mappings_" + winsgt_type + is_ip + is_winsgt_const + is_reversed)
    # pp.transform_trainset()
    # pp.transform_testset()
    # pp.close()

    ''' Model Training (step 5) '''

    dataset_types = {  # loop: every key and value in array
        ("/4_processed" + is_reversed): ["_" + winsgt_type],  # Normal Strides
        # "/winsgt_ip": ["_winsgt4s1_ip"]  # IP Segt
    }
    batch_n_tests = [
        3793  # 311026 instances (/winsgt4s1)
        # 4203  # 311022 instances (/winsgt8s1)
        # 1474  # 311014 instances (/winsgt16s1)
        # 3049  # 310998 instances (/winsgt32s1)
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
                'stateful_ip': False,
                'm1_labels': True,
                'save_output': 'G' + winsgt_dir[1:] + '/5_output' + is_reversed,
                'batch_n_test': batch_n_tests[batch_n],
                'hyperparameters': {
                    'sequence_max_n': int(re.findall(r"\d+", segt_type)[0]),  # extract first int from segt_type
                    'batch_n': 32,
                    'epochs_n': 200,
                    'units_n': 128,
                    'layers_n': 1,
                    'dropout_r': 0.4,  # 0 when performing tests
                    'learning_r': 0.01,
                    'decay_r': 0.96,
                    'calc_dev': 3,
                    'e.stopping': 3  # use None or 0 if N/A
                }
            }

            for general_config, general_values in general_configs.items():  #
                for value in general_values:  # binary / multi-class or m:1 / m:n labeling strategy
                    model_configs[general_config] = value

                    for hyperparams in hyperparams_configs:  # hyperparameters
                        for hyperparam, hyperparam_value in hyperparams.items():
                            model_configs['hyperparameters'][hyperparam] = hyperparam_value

                        # myModel = ModelTrainer(
                        #     model_configs,
                        #     dataset_dir + "/meta/4_mappings" + segt_type +
                        #     is_ip + is_winsgt_const + is_reversed + ".hd5",
                        #     checkpoint_dir,
                        #     saver_dir
                        # )
                        #
                        # myModel.train(
                        #     winsgt_dir + dataset_dict[0] + (
                        #         devset_name if len(is_reversed) else trainset_name) + segt_type + is_ip + ".hd5",
                        #     winsgt_dir + dataset_dict[0] + (
                        #         trainset_name if len(is_reversed) else devset_name) + segt_type + is_ip + ".hd5"
                        # )

                        # myModel.validate(
                        #     winsgt_dir + dataset_dict[0] + (
                        #         devset_name if len(is_reversed) else trainset_name) + segt_type + is_ip + ".hd5",
                        #     winsgt_dir + dataset_dict[0] + (
                        #         trainset_name if len(is_reversed) else devset_name) + segt_type + is_ip + ".hd5"
                        # )
                        #
                        # myModel.gen_output(
                        #     winsgt_dir + dataset_dict[0] + (
                        #         devset_name if len(is_reversed) else trainset_name) + segt_type + is_ip + ".hd5",
                        #     winsgt_dir + dataset_dict[0] + (
                        #         trainset_name if len(is_reversed) else devset_name) + segt_type + is_ip + ".hd5", True
                        # )
