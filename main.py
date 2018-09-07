from preprocesshandler.preprocess import PreProcessing
from segregationhandler.flowsegt import FlowSegregation
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

''' Preprocessing config. '''
# EG: [] vs [1,2,3]
# EG: None vs {0:0, 1:1, 3:3}
# EG: None vs 1

pp_config = {
    'io': {
        'train_dir': 'E:/data/CIDDS-001/OpenStack/CIDDS-001-internal-week1.csv',
        'test_dir': 'E:/data/CIDDS-001/OpenStack/CIDDS-001-internal-week2.csv',
        'output_dir': 'E:/data/CIDDS-001/OpenStack/processed',
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
    'normalization': 'zscore',  # zscore, minmax1r, minmax2r
    'label': {
        'i': [12, 13],  # col to use (max len: 2), treating ith column(s) until as label(s)
        'lbl_normal': ['normal', '---']  # the value of label which is "benign/normal" (same alignment w/ index above)
    },
    'pp': {
        't': [0],
        'ips': [3, 5],
        'pts': [],
        '1hot': [2],
        'flg': [10],  # .A.... -> 010000 (6)
        'fwd_tos': [11],  # 4 -> 00000100 (8)
        'norm': [1, 7, 8],
        'rm': [9]  # col(s) to remove
    }
}

ipsgt_config = {
    'input_dir': 'E:/data/CIDDS-001/OpenStack/processed',
    'output_dir': 'E:/data/CIDDS-001/OpenStack/processed',
    'ip_1': 5,
    'ip_2': 7
}

if __name__ == '__main__':
    # # Features Preprocessing (step 1)
    # pp_ugr = PreProcessing(pp_config)
    # pp_ugr.get_metadata()
    # pp_ugr.transform_trainset()
    # pp_ugr.transform_testset()

    # IP Segregation (step 2)
    ipsgt = FlowSegregation(ipsgt_config,
                            time_window=10000, time_out=60000,
                            flow_te=True, bidirectional=True)
    ipsgt.ip_segregate()
