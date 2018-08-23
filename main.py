from preprocesshandler.preprocess import PreProcessing
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

''' UGR '''
# pp_config = {
#     'io': {
#         'read_chunk_size': 1000000,
#         'train_sets': {
#             # 'dir': 'E:/data/CIDDS-001/OpenStack',
#             'dir': 'F:/uniq',
#             'labels': -1,
#         },
#         'test_sets': None,
#         'output_folder': 'processed'
#     },
#
#     'pp': {
#         'te': 0,
#         'ips': [2, 3],
#         '1hot': [8],
#         'flg': 7,  # .A.... -> 010000 (6)
#         'fwd_tos': [8, 9],  # 4 -> 00000100 (8)
#         'norm': [1, 10, 11],
#         'lbl': 12
#     }
# }

''' CIDDS '''
pp_config = {
    'io': {
        'train_dir': 'E:/data/CIDDS-001/OpenStack/CIDDS-001-internal-week1.csv',
        'test_dir': 'E:/data/CIDDS-001/OpenStack/CIDDS-001-internal-week2.csv',
        'output_folder': 'E:/data/CIDDS-001/OpenStack/processed',
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

''' TEST '''
# EG: [] vs [1,2,3]
# EG: None vs {0:0, 1:1, 3:3}
# EG: None vs 1
"""
pp_config = {
    'io': {
        'train_dir': 'E:/data/train.csv',
        'test_dir': 'E:/data/test.csv',
        'output_folder': 'E:\\data\\processed',
        'read_chunk_size': 2,

        # parsing ith column(s) as dates
        'dates': [0],
        # treating ith column until the last column as label(s)
        'labels': -1,
        # input data types for each column (omit labels if it's in separate file)
        'dtypes_in': {
            1: np.float32, 2: np.str, 3: np.str, 4: np.uint8, 5: np.int32
        },
        # output data types to be expected after preprocessing
        'dtypes_out': {
            1: np.float32, 2: np.float32, 3: np.float32, 4: np.uint8, 5: np.int32
        }
    },
    'normalization': 'minmax2r',  # zscore, minmax1r, minmax2r
    'pp': {
        't': [0],
        'ips': [3],
        'pts': [],
        '1hot': [4],
        'flg': [2],  # .A.... -> 010000 (6)
        'fwd_tos': [],  # 4 -> 00000100 (8)
        'norm': [],
        'rm': [1]  # col(s) to remove
    }
}
"""

mapping_one_hot = {

}

pp_ugr = PreProcessing(pp_config)
pp_ugr.get_metadata()
pp_ugr.transform_trainset()
pp_ugr.transform_testset()

# from inputhandler.input_reader import InputReader
# # testInput = InputReader("E:/test.csv", label_loc=-1, read_chunk_size=2)
# testInput = InputReader("E:/test_x.csv", labels_file="E:/test_y.csv", read_chunk_size=2)
# testInput = InputReader("E:/test_x.csv", read_chunk_size=2)
#
# z = True
# while z:
#     x, y, z = testInput.next()
#     print("data:", x)
#     print("labels:", y)
#
# z = True
# while z:
#     x, y, z = testInput.next()
#     print("data:", x)
#     print("labels:", y)
#
