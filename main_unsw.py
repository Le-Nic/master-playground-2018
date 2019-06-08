import numpy as np
import re
from preprocesshandler.preprocess import PreProcessing
from segregationhandler.winsegt import WindowSegregation

np.random.seed(147)

if __name__ == '__main__':

    # # # # # # # # # #
    # # START  TUNE # #
    # # # # # # # # # #

    # Data Set
    dataset_dir = "F:/data/UNSW_splits"  # splits
    trainset_name = "/UNSW_NB15_training-set"  # splits
    testset_name = "/UNSW_NB15_testing-set"  # splits
    devset_name = "/UNSW_NB15_validation-set"  # splits

    # dataset_dir = "G:/data/UNSW"
    # trainset_name = "/UNSW-NB15_1"
    # testset_name = "/UNSW-NB15_3"
    # devset_name = "/UNSW-NB15_2"

    # Data Structure
    winsgt_type = "winsgt32s1"
    is_winsgt_const = ""  # is constance sequence: "" or "_const"
    is_ip = ""  # is IP segregated: "" or "_ip"

    # Model
    model_type = "rnn"  # hierc / tcn / rnn
    saver_dir = "/checkpoints_" + model_type + is_ip + is_winsgt_const
    checkpoint_dir = "/checkpoints_" + model_type + is_ip + is_winsgt_const

    # Preprocessing
    with open("configs/unsw.txt", 'r') as config_file:
        pp_config = eval(config_file.read())

    # # # # # # # # # #
    # #  END  TUNE  # #
    # # # # # # # # # #

    # # # # # # # # # #
    # # START SETUP # #
    # # # # # # # # # #

    if model_type == "hierc":
        from modelhandler.hierarchical.hierarchicalmodeltrainer import ModelTrainer
    elif model_type == "tcn":
        from modelhandler.tcn.tcnmodeltrainer import ModelTrainer
    else:
        from modelhandler.rnn.modeltrainer import ModelTrainer

    winsgt_dir = dataset_dir + "/2_" + winsgt_type + is_winsgt_const
    saver_dir = winsgt_dir + saver_dir
    checkpoint_dir = winsgt_dir + checkpoint_dir

    # a/pre-pend directory paths
    for key in ['train_dir', 'test_dir', 'output_dir']:
        if pp_config['convert_hd5']['io_csv'][key] is not None:
            pp_config['convert_hd5']['io_csv'][key] = dataset_dir + pp_config['convert_hd5']['io_csv'][key]

    # assuming only 1 file for each train/test set
    pp_config['process_hd5']['io_hd5']['train_dir'] = winsgt_dir + "/2_winsgt" + \
        trainset_name + "_" + winsgt_type + is_ip + ".hd5"

    if len(devset_name):
        pp_config['process_hd5']['io_hd5']['test_dir'] = [
            winsgt_dir + "/2_winsgt" + devset_name + "_" + winsgt_type + is_ip + ".hd5",
            winsgt_dir + "/2_winsgt" + testset_name + "_" + winsgt_type + is_ip + ".hd5"
        ]
    else:
        pp_config['process_hd5']['io_hd5']['test_dir'] = winsgt_dir + "/2_winsgt" + \
            testset_name + "_" + winsgt_type + is_ip + ".hd5"

    pp_config['process_hd5']['io_hd5']['output_dir'] = winsgt_dir + \
        pp_config['process_hd5']['io_hd5'][key]

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

    ''' Win/Time IP Segregation (step 2a) '''
    flowsgt_config = {
        'input_dir': dataset_dir + "/1_converted/train",  # TUNE THIS (specific / whole dir)
        'output_dir': winsgt_dir + "/2_winsgt",
        'features_len': dataset_dir + "/meta/1_mappings.hd5",
        'meta_output_name': "2_mappings"
    }
    winsgt_seq = [int(s) for s in winsgt_type.split("winsgt")[1].split("s")]

    # winsgt = WindowSegregation(flowsgt_config, sequence_max=winsgt_seq[0], ip_segt=bool(is_ip), stride=winsgt_seq[1],
    #                            single_output=True, const_sequence=bool(is_winsgt_const))
    # winsgt.window_segregate()
    # winsgt.close()

    ''' IP Batch Rearrange (step 2b) '''
    ipbatch_config = {
        'input_dir': winsgt_dir + "/2_winsgt" + trainset_name + "_" + winsgt_type + is_ip + ".hd5",
        'output_dir': winsgt_dir + "/3_winsgt",
        'features_len': dataset_dir + "/meta/" + (
            "2_mappings_" + winsgt_type + is_ip + is_winsgt_const + ".hd5" if is_ip else "/1_mappings.hd5")
    }
    # ipbatchsgt = IPBatchSegregation(ipbatch_config, 64)
    # ipbatchsgt.ip_batch_segregate()
    # ipbatchsgt.close()

    ''' Features Preprocessing (step 3a) '''
    # pp = PreProcessing(pp_config['process_hd5'])  # HD5 for tensorflow
    # pp.get_metadata()
    # pp.save_metadata(dataset_dir + "/meta", name="4_mappings_" + winsgt_type + is_ip + is_winsgt_const)
    # pp.transform_trainset()
    # pp.transform_testset()
    # pp.close()

    ''' Features Preprocessing (step 3b) '''
    hiercsgt_config = {
        'input_dir': "F" + winsgt_dir[1:] + "/4_processed" + trainset_name + "_" + winsgt_type + is_ip + ".hd5",
        'output_dir': "G" + winsgt_dir[1:] + "/5_hiercsgt",
        'features_len': dataset_dir + "/meta/" + "4_mappings_" + winsgt_type + is_ip + is_winsgt_const + ".hd5"
    }
    # hiercsgt = HierarchicalSegregation(hiercsgt_config, host_sequence=2)
    # hiercsgt.hierc_segregate()
    # hiercsgt.close()

    ''' Model Training (step 4) '''
    dataset_types = {
        '/5_hiercsgt' if model_type == "hierc" else '/4_processed': ["_" + winsgt_type]
    }

    general_configs = {  # loop: every key and value in array
        'class_type': [0]  # 0: 2-class, 1: 3-class, 2: 5-class, 3: 9-class
    }

    # Random Grid Search
    # hyperparams_n = 10
    # random_hyperparams_search = {
    #     'units_n': list(np.random.uniform(64, 512, hyperparams_n).astype('int')),
    #     'batch_n': list(np.random.choice([64, 128, 256, 512], hyperparams_n).astype('int')),
    #     'dropout_r': list(np.random.uniform(0., 0.5, hyperparams_n).round(2)),
    #     'learning_r': list(np.random.choice([0.0001, 0.001], hyperparams_n)),
    #     'channels_size': list(np.random.choice([3, 2, 4], hyperparams_n).astype('int'))
    # }
    #
    # if model_type == "tcn":
    #     random_hyperparams_search['channels_n'] = []
    #     for pairs in zip(random_hyperparams_search['units_n'], random_hyperparams_search['channels_size']):
    #         random_hyperparams_search['channels_n'].append([pairs[0] for _ in range(pairs[1])])
    #     random_hyperparams_search.pop('units_n', None)
    #
    # hyperparams_configs = [
    #     {next(iter(random_hyperparams_search.keys())): v} for v in next(iter(random_hyperparams_search.values()))
    # ]
    #
    # for key, values in random_hyperparams_search.items():
    #     for i, value in enumerate(values):
    #         hyperparams_configs[i][key] = value

    # Grid Search
    hyperparams_configs = [  # loop: every hyperparameters pair
        {'kernels_n': 4, 'channels_n': [64, 64, 64], 'batch_n': 128, 'dropout_r': .3},
        {'kernels_n': 8, 'channels_n': [64, 64, 64], 'batch_n': 128, 'dropout_r': .3},
        {'kernels_n': 16, 'channels_n': [64, 64, 64], 'batch_n': 128, 'dropout_r': .3},
        {'kernels_n': 4, 'channels_n': [128, 128, 128], 'batch_n': 128, 'dropout_r': .3},
        {'kernels_n': 8, 'channels_n': [128, 128, 128], 'batch_n': 128, 'dropout_r': .3},
        {'kernels_n': 16, 'channels_n': [128, 128, 128], 'batch_n': 128, 'dropout_r': .3},
        {'kernels_n': 4, 'channels_n': [64, 64, 64], 'batch_n': 256, 'dropout_r': .3},
        {'kernels_n': 8, 'channels_n': [64, 64, 64], 'batch_n': 256, 'dropout_r': .3},
        {'kernels_n': 16, 'channels_n': [64, 64, 64], 'batch_n': 256, 'dropout_r': .3},
        {'kernels_n': 4, 'channels_n': [128, 128, 128], 'batch_n': 256, 'dropout_r': .3},
        {'kernels_n': 8, 'channels_n': [128, 128, 128], 'batch_n': 256, 'dropout_r': .3},
        {'kernels_n': 16, 'channels_n': [128, 128, 128], 'batch_n': 256, 'dropout_r': .3},

        {'kernels_n': 4, 'channels_n': [64, 64, 64], 'batch_n': 128, 'dropout_r': .6},
        {'kernels_n': 8, 'channels_n': [64, 64, 64], 'batch_n': 128, 'dropout_r': .6},
        {'kernels_n': 16, 'channels_n': [64, 64, 64], 'batch_n': 128, 'dropout_r': .6},
        {'kernels_n': 4, 'channels_n': [128, 128, 128], 'batch_n': 128, 'dropout_r': .6},
        {'kernels_n': 8, 'channels_n': [128, 128, 128], 'batch_n': 128, 'dropout_r': .6},
        {'kernels_n': 16, 'channels_n': [128, 128, 128], 'batch_n': 128, 'dropout_r': .6},
        {'kernels_n': 4, 'channels_n': [64, 64, 64], 'batch_n': 256, 'dropout_r': .6},
        {'kernels_n': 8, 'channels_n': [64, 64, 64], 'batch_n': 256, 'dropout_r': .6},
        {'kernels_n': 16, 'channels_n': [64, 64, 64], 'batch_n': 256, 'dropout_r': .6},
        {'kernels_n': 4, 'channels_n': [128, 128, 128], 'batch_n': 256, 'dropout_r': .6},
        {'kernels_n': 8, 'channels_n': [128, 128, 128], 'batch_n': 256, 'dropout_r': .6},
        {'kernels_n': 16, 'channels_n': [128, 128, 128], 'batch_n': 256, 'dropout_r': .6},
        # {'layers_n': 1, 'units_n': 128, 'batch_n': 128}
        # {'layers_n': 1, 'units_n': 128, 'batch_n': 128}, {'layers_n': 1, 'units_n': 128, 'batch_n': 256},  # layer 1
        # {'layers_n': 1, 'units_n': 128, 'batch_n': 512}, {'layers_n': 1, 'units_n': 256, 'batch_n': 128},
        # {'layers_n': 1, 'units_n': 256, 'batch_n': 256}, {'layers_n': 1, 'units_n': 256, 'batch_n': 512},
        # {'layers_n': 1, 'units_n': 384, 'batch_n': 128}, {'layers_n': 1, 'units_n': 384, 'batch_n': 256},
        # {'layers_n': 1, 'units_n': 384, 'batch_n': 512},
        # {'layers_n': 2, 'units_n': 128, 'batch_n': 128}, {'layers_n': 2, 'units_n': 128, 'batch_n': 256},  # layer 2
        # {'layers_n': 2, 'units_n': 128, 'batch_n': 512}, {'layers_n': 2, 'units_n': 256, 'batch_n': 128},
        # {'layers_n': 2, 'units_n': 256, 'batch_n': 256}, {'layers_n': 2, 'units_n': 256, 'batch_n': 512},
        # {'layers_n': 2, 'units_n': 384, 'batch_n': 128}, {'layers_n': 2, 'units_n': 384, 'batch_n': 256},
        # {'layers_n': 2, 'units_n': 384, 'batch_n': 512},
        # {'layers_n': 3, 'units_n': 128, 'batch_n': 128}, {'layers_n': 3, 'units_n': 128, 'batch_n': 256},  # layer 3
        # {'layers_n': 3, 'units_n': 128, 'batch_n': 512}, {'layers_n': 3, 'units_n': 256, 'batch_n': 128},
        # {'layers_n': 3, 'units_n': 256, 'batch_n': 256}, {'layers_n': 3, 'units_n': 256, 'batch_n': 512},
        # {'layers_n': 3, 'units_n': 384, 'batch_n': 128}, {'layers_n': 3, 'units_n': 384, 'batch_n': 256},
        # {'layers_n': 3, 'units_n': 384, 'batch_n': 512}
    ]

    batch_n_tests = [
        # 1
        4574  # (82332, {4-32}, 194) instances (/winsgt{4-32}s1 splits)
        # 15  # (1140045, {8-32}, {2-32}, 238) instances (/winsgt{4-32}s1 hierarchical splits)
    ]

    batch_n_dev = [
        # 1
        1594  # (17534, {4-32}, 194) instances (/winsgt{4-32}s1 splits)
        # 3500  # (14000, {8-32}, {2-32}, 238) instances (/winsgt{4-32}s1 hierarchical splits)
    ]

    # BREAKING F****** NEWS, gen_output doesn't save all outputs for training set (mod batch amount of data are missed)
    for batch_n, dataset_dict in enumerate(dataset_types.items()):  # ip / normal
        for segt_type in dataset_dict[1]:  # sequences

            model_configs = {  # default values
                'class_type': 1,
                'save_output': 'G' + winsgt_dir[1:] + '/5_output' + is_ip,
                # 'save_output': None,
                'batch_n_test': batch_n_tests[batch_n],
                'batch_n_dev': batch_n_dev[batch_n],
                'stateful_ip': False,  # STATEFUL IP TUNING
                'stateful': False,
                'm1_labels': True,  # LABEL TUNING
                'hyperparameters': {
                    'netw_sequence': int(re.findall(r"\d+", segt_type)[0]),  # extract first int from segt_type
                    'host_sequence': 4,  # HIERARCHICAL TUNING
                    'netw_output_n': 100,  # HIERARCHICAL/TCN TUNING
                    'host_output_n': 100,  # HIERARCHICAL TUNING

                    'kernels_n': 4,  # TCN TUNING
                    'channels_n': [10, 10, 10],  # TCN TUNING

                    'units_n': 1,
                    'layers_n': 1,
                    'dropout_r': 0.,
                    'learning_r': 0.0001,

                    'e.stopping': 24,  # use None or 0 if N/A
                    'epochs_n': 400
                }
            }

            for general_config, general_values in general_configs.items():  #
                for value in general_values:  # binary / multi-class or m:1 / m:n labeling strategy
                    model_configs[general_config] = value

                    for hyperparams in hyperparams_configs:  # hyperparameters
                        for hyperparam, hyperparam_value in hyperparams.items():
                            model_configs['hyperparameters'][hyperparam] = hyperparam_value

                        myModel = ModelTrainer(
                            model_configs,
                            'F' + dataset_dir[1:] + "/meta/4_mappings" + segt_type + is_ip + is_winsgt_const + ".hd5",
                            checkpoint_dir,
                            saver_dir
                        )

                        myModel.train(
                            winsgt_dir + dataset_dict[0] + trainset_name + segt_type + is_ip + ".hd5",
                            winsgt_dir + dataset_dict[0] + testset_name + segt_type + is_ip + ".hd5",
                            winsgt_dir + dataset_dict[0] + devset_name + segt_type + is_ip + ".hd5",
                        )

                        # myModel.validate(
                        #     winsgt_dir + dataset_dict[0] + trainset_name + segt_type + is_ip + ".hd5",
                        #     winsgt_dir + dataset_dict[0] + testset_name + segt_type + is_ip + ".hd5"
                        # )
                        #
                        # myModel.gen_output(
                        #     winsgt_dir + dataset_dict[0] + trainset_name + segt_type + is_ip + ".hd5",
                        #     winsgt_dir + dataset_dict[0] + testset_name + segt_type + is_ip + ".hd5", True
                        # )
