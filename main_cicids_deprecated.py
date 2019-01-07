from preprocesshandler.preprocess import PreProcessing
from segregationhandler.timesegt import TimeSegregation
from segregationhandler.winsegt import WindowSegregation
from inputhandler.dataset_shuffler import DatasetShuffler
from modelhandler.modeltrainer_seq2seq import ModelTrainer
import numpy as np


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

    with open("configs/cicids_deprecated.txt", 'r') as config_file:
        pp_config = eval(config_file.read())

    # pp = PreProcessing(pp_config['convert_hd5'])
    # pp.get_metadata()
    # pp.save_metadata("E:/data/CICIDS/meta", name="1_mappings")
    # pp.transform_trainset()

    ''' Win/Time IP Segregation (step 2) '''
    flowsgt_config = {
        'input_dir': "E:/data/CICIDS/1_converted",
        'output_dir': "E:/data/CICIDS/2_winsgt",
        'features_len': "E:/data/CICIDS/meta/1_mappings.hd5",
        'meta_output_name': "2_winsgt"
    }

    # winsgt = WindowSegregation(flowsgt_config, sequence_max=4,
    #                            ip_segt=False, stride=4, single_output=True)
    # winsgt.window_segregate()
    # winsgt.close()

    # timesgt = TimeSegregation(flowsgt_config, time_window=10, time_out=60, sequence_max=16,
    #                           bidirectional=True, single_output=True)
    # timesgt.time_segregate()

    ''' Data Shuffling (step 3) '''
    shuffle_config = {
        'input_path': "E:/data/CICIDS/2_winsgt/winsgt4.hd5",
        'output_dir': "F:/CICIDS/3_shuffled/winsgt",
        # 'meta_path': "E:/data/CICIDS/meta/2_mappings_winsgt16.hd5",
        'meta_path': "E:/data/CICIDS/meta/1_mappings.hd5",
    }

    # hd5huffler = DatasetShuffler(shuffle_config)
    # hd5huffler.shuffle(n_splits=3, test_size=0.1)

    ''' Features Preprocessing (step 4) '''

    # pp = PreProcessing(pp_config['process_hd5'])  # HD5 for tensorflow
    # pp.get_metadata()
    # pp.save_metadata("E:/data/CICIDS/meta", name="4_mappings_winsgt4")
    # pp.transform_trainset()
    # pp.transform_testset()

    ''' Model Training (step 4) '''

    dataset_meta = "E:/data/CICIDS/meta/4_mappings_winsgt4.hd5"
    train_dir = "E:/data/CICIDS/4_processed/winsgt/winsgt4_train_0.hd5"
    dev_dir = "E:/data/CICIDS/4_processed/winsgt/winsgt4_test_0.hd5"
    saver_dir = "E:/data/CICIDS/checkpoints"
    checkpoint_dir = None

    model_config = {
        'class_type': 1,  # 0: 2-class, 1: 3-class, 2: 5-class, 3: 9-class
        'seq_constant': False,
        'batch_n_test': 1,  # ipsgt3: 51969

        'hyperparameters': {
            'sequence_max_n': 4,
            'batch_n': 64,
            'epochs_n': 50,
            'units_n': 128,
            'layers_n': 1,
            'dropout_r': 0.4,  # 0 when performing tests
            'learning_r': 0.01,
            'decay_r': 0.96
        }
    }

    myModel = ModelTrainer(model_config, dataset_meta, checkpoint_dir, saver_dir)
    myModel.train(train_dir, dev_dir)
