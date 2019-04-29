import subprocess
# import logging

# parent_directory = "G:/data/KDD99/2_winsgt32s1"  # KDD
# parent_directory = "G:/data/KDD99_10/2_winsgt32s1"  # KDD 10%
# parent_directory = "G:/data/NSLKDD/2_winsgt8s1"  # NSL
# parent_directory = "G:/data/UNSW/2_winsgt32s1"  # UNSW
# parent_directory = "G:/data/UNSW_splits/2_winsgt32s1"  # UNSW split
parent_directory = "G:/data/CIDDS-002/2_winsgt32s1"  # CIDDS

hyperparams = [
    # "_y0_last"  # classical

    # "U228B64L1D25LR1e-03_y0_1_last",
    # "U347B128L1D8LR1e-04_y0_1_last",
    # "U77B64L1D38LR1e-03_y0_1_last",
    # "U373B512L1D14LR1e-03_y0_1_last",
    # "U484B64L1D9LR1e-04_y0_1_last",
    "U485B128L1D9LR1e-03_y0_1_last"
    # "U236B256L1D1LR1e-04_y0_1_last",
    # "U97B64L1D10LR1e-04_y0_1_last",
    # "U281B512L1D17LR1e-03_y0_1_last",
    # "U319B64L1D24LR1e-03_y0_1_last"

    # "U128B128L1D0LR1e-04_y0_1_last",  # layer 1
    # "U128B256L1D0LR1e-04_y0_1_last",
    # "U128B512L1D0LR1e-04_y0_1_last",
    # "U256B128L1D0LR1e-04_y0_1_last",
    # "U256B256L1D0LR1e-04_y0_1_last",
    # "U256B512L1D0LR1e-04_y0_1_last",
    # "U384B128L1D0LR1e-04_y0_1_last",
    # "U384B256L1D0LR1e-04_y0_1_last",
    # "U384B512L1D0LR1e-04_y0_1_last"

    # "NO100K4C[64, 64, 64]U1B128L1D30LR1e-04_y0_1_last",
    # "NO100K8C[64, 64, 64]U1B128L1D30LR1e-04_y0_1_last",
    # "NO100K16C[64, 64, 64]U1B128L1D30LR1e-04_y0_1_last",
    # "NO100K4C[128, 128, 128]U1B128L1D30LR1e-04_y0_1_last",
    # "NO100K8C[128, 128, 128]U1B128L1D30LR1e-04_y0_1_last",
    # "NO100K16C[128, 128, 128]U1B128L1D30LR1e-04_y0_1_last",
    # "NO100K4C[64, 64, 64]U1B256L1D30LR1e-04_y0_1_last",
    # "NO100K8C[64, 64, 64]U1B256L1D30LR1e-04_y0_1_last",
    # "NO100K16C[64, 64, 64]U1B256L1D30LR1e-04_y0_1_last",
    # "NO100K4C[128, 128, 128]U1B256L1D30LR1e-04_y0_1_last",
    # "NO100K8C[128, 128, 128]U1B256L1D30LR1e-04_y0_1_last",
    # "NO100K16C[128, 128, 128]U1B256L1D30LR1e-04_y0_1_last",

    # "NO100K4C[64, 64, 64]U1B128L1D60LR1e-04_y0_1_last",
    # "NO100K8C[64, 64, 64]U1B128L1D60LR1e-04_y0_1_last",
    # "NO100K16C[64, 64, 64]U1B128L1D60LR1e-04_y0_1_last",
    # "NO100K4C[128, 128, 128]U1B128L1D60LR1e-04_y0_1_last",
    # "NO100K8C[128, 128, 128]U1B128L1D60LR1e-04_y0_1_last",
    # "NO100K16C[128, 128, 128]U1B128L1D60LR1e-04_y0_1_last",
    # "NO100K4C[64, 64, 64]U1B256L1D60LR1e-04_y0_1_last",
    # "NO100K8C[64, 64, 64]U1B256L1D60LR1e-04_y0_1_last",
    # "NO100K16C[64, 64, 64]U1B256L1D60LR1e-04_y0_1_last",
    # "NO100K4C[128, 128, 128]U1B256L1D60LR1e-04_y0_1_last",
    # "NO100K8C[128, 128, 128]U1B256L1D60LR1e-04_y0_1_last",
    # "NO100K16C[128, 128, 128]U1B256L1D60LR1e-04_y0_1_last"

]

dataset_types = {
    # "/4_processed": [
    #     "_winsgt1s1"
    # ]  # classical
    "/5_output": [
       "_winsgt32s1_"  # sequence tuning
    ]  # stacked
}

# classical
# trainset_name = "/kdd_train"  # KDD
# testset_name = "/kdd_test"
# trainset_name = "/kdd_train_10"  # KDD 10%
# testset_name = "/kdd_test"
# trainset_name = "/KDDTrain+"  # NSL
# testset_name = "/KDDTest+"
# trainset_name = "/UNSW_NB15_training-set"  # UNSW split
# testset_name = "/UNSW_NB15_testing-set"

# trainset_name = "/train/kdd_train"  # KDD
# testset_name = "/test/kdd_test"
# trainset_name = "/train/kdd_train_10"  # KDD 10%
# testset_name = "/test/kdd_test"
# trainset_name = "/train/KDDTrain+"  # NSL
# testset_name = "/test/KDDTest+"
# trainset_name = "/train/UNSW-NB15_1"  # UNSW
# testset_name = "/test/UNSW-NB15_3"
# trainset_name = "/train/UNSW_NB15_training-set"  # UNSW split
# testset_name = "/test/UNSW_NB15_testing-set"
trainset_name = "/train/week1"  # CIDDS
testset_name = "/test/week2"


#
# label_types = [
#     # "_y0_last"  # classical
#     "_y0_1_last"
#     # "_y1_last"
# ]

models = [
    # "weka.classifiers.trees.J48 -C 0.25 -M 2",  # classical only
    #
    "weka.classifiers.trees.RandomForest -P 100 -I 100 -num-slots 1 -K 0 -M 1.0 -V 0.001 -S 1",
    #
    # "weka.classifiers.functions.SMO -C 1.0 -L 0.001 -P 1.0E-12 -N 0 -V -1 -W 1 -K " +
    # "\"weka.classifiers.functions.supportVector.PolyKernel -E 1.0 -C 250007\" " +
    # "-calibrator \"weka.classifiers.functions.Logistic -R 1.0E-8 -M 1 -num-decimal-places 4\""
]

for dataset_dir, type_lists in dataset_types.items():
    for dataset_type in type_lists:
        # for label_type in label_types:
        for hyperparam in hyperparams:
            for model_weka in models:
                train_path = parent_directory + dataset_dir + trainset_name + dataset_type + hyperparam
                test_path = parent_directory + dataset_dir + testset_name + dataset_type + hyperparam

                # log_file = logging.FileHandler(train_path + ".log")
                # log_console = logging.StreamHandler()
                # logging.basicConfig(level=logging.INFO, format="%(message)s", handlers=[log_file, log_console])

                full_command = "java -Xmx40000M -cp \"D:/Program Files/Weka-3-8/weka.jar\" " + model_weka + \
                               " -t \"" + train_path + ".arff\" -T \"" + test_path + ".arff\"" + \
                               " -d \"" + train_path + "_" + model_weka.split()[0].split(".")[-1] + "\""
                print("[WEKA] Processing > " + train_path + "_" + model_weka.split()[0].split(".")[-1])

                pipe = subprocess.Popen(full_command, shell=True, stdout=subprocess.PIPE).stdout
                command_out = pipe.read()

                # logging.info(command_out)
                with open(train_path + "_" + model_weka.split()[0].split(".")[-1] + ".log", "wb") as f:
                    f.write(command_out)

