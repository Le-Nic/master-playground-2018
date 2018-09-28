from preprocesshandler.preprocess import PreProcessing
from segregationhandler.ipsegt import IpSegregation
# from modelhandler.modeltrainer import ModelTrainer
from modelhandler.modeltrainer_classic import ModelTrainer
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
    # EG: [] vs [1,2,3]
    # EG: None vs {0:0, 1:1, 3:3}
    # EG: None vs 1
    pp_config = {
        'io': {
            # 'train_dir': 'E:/data/CIDDS-001/OpenStack/CIDDS-001-internal-week1.csv',
            # 'test_dir': 'E:/data/CIDDS-001/OpenStack/CIDDS-001-internal-week2.csv',
            'train_dir': 'E:/data/CIDDS-001/OpenStack/CIDDS-001-train_normaltrimmed.csv',
            'test_dir': 'E:/data/CIDDS-001/OpenStack/CIDDS-001-test.csv',
            'output_dir': 'E:/data/CIDDS-001/OpenStack/processed_test',
            'read_chunk_size': 2000000,
            # parsing ith column(s) as dates
            'dates': [0],
            # input data types for each column (omit labels if it's in separate file)
            'dtypes_in': {
                1: np.float32, 2: np.object, 3: np.object, 4: np.float32,  # td, pr, sa, sp
                5: np.object, 6: np.float32, 7: np.int32, 8: np.float32,  # da, dp, pkt, byt
                9: np.int32, 10: np.object, 11: np.int32, 12: np.object,  # fl, flg, stos, lbl
                13: np.object, 14: np.object, 15: np.object  # a.type, a.id, a.desc
            },
            # output data types to be expected after preprocessing, current dtype: Float64Atom()
            'dtypes_out': {}
        },
        'normalization': 'minmax1r',  # zscore, minmax1r, minmax2r
        'label': {
            'i': [12, 13],  # col to use (max len: 2), treating ith column(s) until as label(s)
            'lbl_normal': ['normal', '---']
            # the value of label which is "benign/normal" (same alignment w/ index above)
        },
        'pp': {
            't': [0],  # zscore does not work for this
            'ips': [3, 5],
            'pts': [],
            '1hot': [2],
            'flg': [10],  # .A.... -> 010000 (6)
            'fwd_tos': [11],  # 4 -> 00000100 (8)
            'norm': [1, 7, 8],
            'rm': [9]  # col(s) to remove
        }
    }

    pp_ugr = PreProcessing(pp_config)
    pp_ugr.get_metadata()
    pp_ugr.transform_trainset()
    # pp_ugr.transform_testset()

    ''' IP Segregation (step 2) '''
    ipsgt_config = {
        # 'input_dir': 'E:/data/CIDDS-001/OpenStack/processed',
        'input_dir': 'E:/data/CIDDS-001/OpenStack/processed_test/CIDDS-001-test_normaltrimmed.hd5',  # test
        # 'output_dir': 'E:/data/CIDDS-001/OpenStack/processed',
        'output_dir': 'E:/data/CIDDS-001/OpenStack/processed_test',
        'ip_1': 5,
        'ip_2': 7
    }

    # meta_path = "E:/data/CIDDS-001/OpenStack/processed/minmax1r/mappings.hd5"
    meta_path = "E:/data/CIDDS-001/OpenStack/processed_test/mappings_normaltrimmed.hd5"  # test

    # ipsgt = IpSegregation(ipsgt_config, features_len=meta_path,
    #                       time_window=10, time_out=60, sequence_max=4,
    #                       bidirectional=True, flow_te=True)
    # ipsgt.ip_segregate()

    ''' Model Training (step 3) '''

    # meta_path = "E:/data/CIDDS-001/OpenStack/processed/minmax1r/mappings.hd5"
    # train_dir = "E:/data/CIDDS-001/OpenStack/processed/minmax1r/CIDDS-001-internal-week1_ipsgt32.hd5"
    # dev_dir = "E:/data/CIDDS-001/OpenStack/processed/minmax1r/CIDDS-001-internal-week2_ipsgt32.hd5"

    meta_path = "E:/data/CIDDS-001/OpenStack/processed_test/mappings_normaltrimmed.hd5"
    train_dir = "E:/data/CIDDS-001/OpenStack/processed_test/CIDDS-001-train_normaltrimmed_ipsgt4.hd5"
    dev_dir = "E:/data/CIDDS-001/OpenStack/processed_test/CIDDS-001-test_normaltrimmed_ipsgt4.hd5"
    validation_dir = None
    resume_checkpoint = False  # test / continue training from last epoch

    model_config = {
        'class_type': 2,  # 0: 2-class, 1: 3-class, 2: 5-class, 3: 9-class

        'hyperparameters': {
            'sequence_max_n': 4,
            'batch_n': 8,  # 1 when performing tests
            'epochs_n': 100,
            'units_n': 32,
            'layers_n': 1,
            'dropout_r': 0.5,  # 0 when performing tests
            'learning_r': 0.01,
            'decay_r': 0.96
        }
    }

    # myModel = ModelTrainer(model_config, resume_checkpoint, meta_path)
    # myModel.train(train_dir, dev_dir)
