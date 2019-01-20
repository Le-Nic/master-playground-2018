from preprocesshandler.preprocess import PreProcessing
from segregationhandler.timesegt import TimeSegregation
from segregationhandler.winsegt import WindowSegregation
from segregationhandler.ipbatchsegt import IPBatchSegregation
from segregationhandler.hierarchicalsegt import HierarchicalSegregation
# from modelhandler.modeltrainer import ModelTrainer
from modelhandler.hierarchicalmodeltrainer import ModelTrainer
import numpy as np
import re


if __name__ == '__main__':

    # # # # # # # # # #
    # # START  TUNE # #
    # # # # # # # # # #

    # Data Set
    # dataset_dir = "F:/data/UNSW_splits"  # splits
    # trainset_name = "/UNSW_NB15_training-set"  # splits
    # devset_name = "/UNSW_NB15_testing-set"  # splits

    dataset_dir = "G:/data/UNSW"
    trainset_name = "/UNSW-NB15_1"
    devset_name = "/UNSW-NB15_2"  # does not support is_reversed
    testset_name = "/UNSW-NB15_3"

    # Data Structure
    winsgt_type = "winsgt4s1"
    is_winsgt_const = "_const"  # empty string if not
    is_ip = ""  # is IP segregated: "" or "_ip"

    # Model
    saver_dir = "/checkpoints"
    checkpoint_dir = "/checkpoints"

    # Preprocessing
    with open("configs/unsw.txt", 'r') as config_file:
        pp_config = eval(config_file.read())

    # # # # # # # # # #
    # #  END  TUNE  # #
    # # # # # # # # # #

    # # # # # # # # # #
    # # START SETUP # #
    # # # # # # # # # #

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

    ''' Win/Time IP Segregation (step 2b) '''
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
        'input_dir': winsgt_dir + "/4_processed" + trainset_name + "_" + winsgt_type + is_ip + ".hd5",
        'output_dir': "G" + winsgt_dir[1:] + "/5_hiercsgt",
        'features_len': dataset_dir + "/meta/" + "4_mappings_" + winsgt_type + is_ip + is_winsgt_const + ".hd5"
    }
    # hiercsgt = HierarchicalSegregation(hiercsgt_config, host_sequence=2)
    # hiercsgt.hierc_segregate()
    # hiercsgt.close()

    ''' Model Training (step 4) '''

    dataset_types = {  # loop: every key and value in array
        # "/4_processed": ["_" + winsgt_type],  # Normal Strides
        "/5_hiercsgt": ["_" + winsgt_type],  # Normal Strides
        # "/winsgt_ip": ["_winsgt4s1_ip"]  # IP Segt
    }
    batch_n_tests = [
        # 109  # (1140031, 8, 8, 238) instances (/winsgt8s1)
        643  # (1140039, 4, 2, 238) instances (/winsgt4s1)

        # 5787  # 1140039 instances (/winsgt4s1)
        # 10459  # 1140031 instances (/winsgt8s1)
        # 76001  # 1140015 instances (/winsgt16s1)

        # 3438  # 175338 instances (/winsgt4s1 splits)
        # 3023  # 175334 instances (/winsgt8s1 splits)
        # 29221  # 175326 instances (/winsgt16s1 splits)
        # 3730  # 175310 instances (/winsgt32s1 splits)
    ]

    general_configs = {  # loop: every key and value in array
        'class_type': [0]  # 0: 2-class, 1: 3-class, 2: 5-class, 3: 9-class
    }

    hyperparams_configs = [{  # loop: every hyperparameters pair
        #     'units_n': 64,
        #     'batch_n': 32
        # }, {
        #     'units_n': 128,
        #     'batch_n': 32
        # }
    #     'units_n': 32,
    #     'batch_n': 64
    # }, {
    #     'units_n': 64,
    #     'batch_n': 64
    # }, {
    #     'units_n': 64,
    #     'batch_n': 128
    # }, {
    #     'units_n': 64,
    #     'batch_n': 256
    # }, {
        'units_n': 128,
        'batch_n': 256
    }, {
       'units_n': 128,
       'batch_n': 512
    }, {
       'units_n': 256,
       'batch_n': 128
    }, {
        'units_n': 256,
        'batch_n': 256
    }, {
       'units_n': 256,
       'batch_n': 512
    }
    ]

    for batch_n, dataset_dict in enumerate(dataset_types.items()):  # ip / normal
        for segt_type in dataset_dict[1]:  # sequences

            model_configs = {  # default values
                'class_type': 1,
                'save_output': 'G' + winsgt_dir[1:] + '/5_output' + is_ip,
                'batch_n_test': batch_n_tests[batch_n],
                'stateful_ip': False,  # STATEFUL IP TUNING
                'm1_labels': True,  # LABEL TUNING
                'hyperparameters': {
                    'netw_sequence': int(re.findall(r"\d+", segt_type)[0]),  # extract first int from segt_type
                    'host_sequence': 2,  # HIERARCHICAL TUNING
                    'netw_output_n': 100,  # HIERARCHICAL TUNING
                    'host_output_n': 100,  # HIERARCHICAL TUNING

                    'batch_n': 32,
                    'epochs_n': 400,
                    'units_n': 128,
                    'layers_n': 1,
                    'dropout_r': 0.4,
                    'learning_r': 0.001,
                    'calc_dev': 64,
                    'e.stopping': 24  # use None or 0 if N/A
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
                            dataset_dir + "/meta/4_mappings" + segt_type + is_ip + is_winsgt_const + ".hd5",
                            checkpoint_dir,
                            saver_dir
                        )

                        myModel.train(
                            winsgt_dir + dataset_dict[0] + trainset_name + segt_type + is_ip + ".hd5",
                            winsgt_dir + dataset_dict[0] + testset_name + segt_type + is_ip + ".hd5",
                            winsgt_dir + dataset_dict[0] + devset_name + segt_type + is_ip + ".hd5",
                        )
                        #
                        # myModel.validate(
                        #     winsgt_dir + dataset_dict[0] + trainset_name + segt_type + is_ip + ".hd5",
                        #     winsgt_dir + dataset_dict[0] + testset_name + segt_type + is_ip + ".hd5"
                        # )
                        #
                        # myModel.gen_output(
                        #     winsgt_dir + dataset_dict[0] + trainset_name + segt_type + is_ip + ".hd5",
                        #     winsgt_dir + dataset_dict[0] + testset_name + segt_type + is_ip + ".hd5", True
                        # )
