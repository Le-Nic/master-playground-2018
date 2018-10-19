from preprocesshandler.preprocess import PreProcessing
from segregationhandler.ipsegt import IpSegregation
from segregationhandler.winsegt import WindowSegregation
from inputhandler.dataset_shuffler import DatasetShuffler
from modelhandler.modeltrainer import ModelTrainer
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

    ''' Features Preprocessing (step 1) '''

    with open("configs/cicids.txt", 'r') as config_file:
        pp_config = eval(config_file.read())

    # pp = PreProcessing(pp_config)
    # pp.get_metadata()
    # pp.save_metadata("E:/data/CICIDS/meta")
    # pp.transform_trainset()
    # pp.transform_testset()

    ''' Win/IP Segregation (step 2) '''
    '''
    IPSgt: time should be 'te' instead of 'ts', use preprocess-handler to convert the 'ts' to 'te',
    1st & 2nd column in /ip will be treated as source & destination address respectively. 
    '''
    # CIDDS
    """
    flowsgt_config = {
        # 'input_dir': "E:/data/CIDDS-001/OpenStack/processed/minmax1r/data",
        'input_dir': "E:/data/CIDDS-001/OpenStack/processed_test/normal/data",  # test
        # 'output_dir': "E:/data/CIDDS-001/OpenStack/processed/minmax1r/data",
        'output_dir': "E:/data/CIDDS-001/OpenStack/processed_test/normal",  # test
        # 'features_len': "E:/data/CIDDS-001/OpenStack/processed/minmax1r/mappings.hd5",
        'features_len': "E:/data/CIDDS-001/OpenStack/processed_test/mappings_normal.hd5",  # test
    }
    """

    # GURE
    flowsgt_config = {
        'input_dir': "E:/data/CICIDS/processed",
        'output_dir': "F:/CICIDS",
        'features_len': "E:/data/CICIDS/meta/mappings.hd5",
    }

    # winsgt = WindowSegregation(flowsgt_config, sequence_max=16, ip_segt=True, single_output=True)
    # winsgt.window_segregate()

    # ipsgt = IpSegregation(flowsgt_config, time_window=5, time_out=20, sequence_max=32, bidirectional=True)
    # ipsgt.ip_segregate()

    ''' Data Shuffling (step 3) '''
    shuffle_config = {
        'input_path': "E:/data/CICIDS/winsgt/1_1winsgt16.hd5",
        'output_dir': "F:/CICIDS",
        'meta_path': "E:/data/CICIDS/meta/mappings.hd5_winsgt16",
    }

    # hd5huffler = DatasetShuffler(shuffle_config)
    # hd5huffler.shuffle(n_splits=3, test_size=0.1)

    ''' Model Training (step 4) '''

    dataset_meta = "E:/data/CIDDS-001/OpenStack/processed/minmax1r/mappings_ipsgt32.hd5"
    # dataset_meta = "E:/data/CIDDS-001/OpenStack/processed_test/mappings_normal.hd5"  # test

    train_dir = "E:/data/CIDDS-001/OpenStack/processed/minmax1r/CIDDS-001-internal-week1_ipsgt32.hd5"
    # train_dir = "F:/CIDDS-001/CIDDS-001-internal-week1_winsgt8.hd5"
    # train_dir = "E:/data/CIDDS-001/OpenStack/processed_test/normal/CIDDS-001-train_winsgt8.hd5"  # test

    dev_dir = "E:/data/CIDDS-001/OpenStack/processed/minmax1r/CIDDS-001-internal-week2_ipsgt32.hd5"
    # dev_dir = "F:/CIDDS-001/CIDDS-001-internal-week2_winsgt8.hd5"
    # dev_dir = "E:/data/CIDDS-001/OpenStack/processed_test/normal/CIDDS-001-test_winsgt8.hd5"  # test

    saver_dir = "E:/data/CIDDS-001/OpenStack/processed/minmax1r/checkpoints"
    # saver_dir = "F:/CIDDS-001/checkpoints"
    # saver_dir = "E:/data/CIDDS-001/OpenStack/processed_test/checkpoints"  # test

    # checkpoint_dir = "F:/CIDDS-001/checkpoints"  # checkpoint directory to resume from, else None
    checkpoint_dir = None

    model_config = {
        'class_type': 2,  # 0: 2-class, 1: 3-class, 2: 5-class, 3: 9-class
        'data_type': 'ip',
        'batch_n_test': 53101,  # ipsgt3: 51969

        'hyperparameters': {
            'sequence_max_n': 32,
            'batch_n': 64,
            'epochs_n': 30,
            'units_n': 128,
            'layers_n': 1,
            'dropout_r': 0.4,  # 0 when performing tests
            'learning_r': 0.01,
            'decay_r': 0.96
        }
    }

    # myModel = ModelTrainer(model_config, dataset_meta, checkpoint_dir, saver_dir)
    # myModel.train(train_dir, dev_dir)
