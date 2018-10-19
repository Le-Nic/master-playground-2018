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

    ''' HD5 Conversion (step 1) '''

    with open("configs/cicids.txt", 'r') as config_file:
        pp_config = eval(config_file.read())

    # pp = PreProcessing(pp_config['convert_hd5'])
    # pp.get_metadata()
    # pp.save_metadata("E:/data/CICIDS/meta", name="1_mappings")
    # pp.transform_trainset()

    ''' Win/IP Segregation (step 2) '''
    flowsgt_config = {
        'input_dir': "E:/data/CICIDS/1_converted",
        'output_dir': "F:/CICIDS/2_winsgt",
        'features_len': "E:/data/CICIDS/meta/1_mappings.hd5",
        'meta_output_name': "2_mappings"
    }

    # winsgt = WindowSegregation(flowsgt_config, sequence_max=16, ip_segt=True, single_output=True)
    # winsgt.window_segregate()

    ''' Data Shuffling (step 3) '''
    shuffle_config = {
        'input_path': "E:/data/CICIDS/2_winsgt/winsgt16.hd5",
        'output_dir': "F:/CICIDS/3_shuffled",
        'meta_path': "E:/data/CICIDS/meta/2_mappings_winsgt16.hd5",
    }

    # hd5huffler = DatasetShuffler(shuffle_config)
    # hd5huffler.shuffle(n_splits=3, test_size=0.1)

    ''' Features Preprocessing (step 4) '''

    pp = PreProcessing(pp_config['process_hd5'])  # HD5 for tensorflow
    pp.get_metadata()
    pp.save_metadata("E:/data/CICIDS/meta", name="4_mappings.hd5")
    pp.transform_trainset()
    pp.transform_testset()

    # pp = PreProcessing(pp_config['process_csv'])  # CSV for weka
    # pp.get_metadata()
    # pp.transform_trainset()
    # pp.transform_testset()

    ''' Model Training (step 4) '''

    dataset_meta = "E:/data/CIDDS-001/OpenStack/processed/minmax1r/mappings_ipsgt32.hd5"
    train_dir = "E:/data/CIDDS-001/OpenStack/processed/minmax1r/CIDDS-001-internal-week1_ipsgt32.hd5"
    dev_dir = "E:/data/CIDDS-001/OpenStack/processed/minmax1r/CIDDS-001-internal-week2_ipsgt32.hd5"
    saver_dir = "E:/data/CIDDS-001/OpenStack/processed/minmax1r/checkpoints"
    checkpoint_dir = None

    model_config = {
        'class_type': 2,  # 0: 2-class, 1: 3-class, 2: 5-class, 3: 9-class
        'data_type': 'ip',
        'batch_n_test': 32768,  # ipsgt3: 51969

        'hyperparameters': {
            'sequence_max_n': 16,
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
