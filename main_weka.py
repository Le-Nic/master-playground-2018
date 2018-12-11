import subprocess
# import logging

parent_directory = "F:/data/UNSW_splits"

hyperparams = [
    ""
    # "_s4u32b64l1d40",
    # "_s4u64b64l1d40",
    # "_s4u64b128l1d40",
    # "_s4u128b256l1d40",
    # "_s4u64b256l1d40",
    "_s4u128b512l1d40",
    "_s4u256b512l1d40"
]

dataset_types = {
    # "/4_processed": [
    #     "_winsgt4s4"
    # ]  # Normal Strides
    "/5_output": [
        "_winsgt4s1"
        # "_winsgt16s1_ip"
    ]  # IP Segt
}

trainset_name = "/train/UNSW_NB15_training-set"
testset_name = "/dev/UNSW_NB15_testing-set"

label_types = [
    "_y0_last"
    # "_y1_last"
]

models = [
    # "weka.classifiers.trees.J48 -C 0.25 -M 2",
    "weka.classifiers.trees.RandomForest -P 100 -I 100 -num-slots 1 -K 0 -M 1.0 -V 0.001 -S 1",
    "weka.classifiers.functions.SMO -C 1.0 -L 0.001 -P 1.0E-12 -N 0 -V -1 -W 1 -K " +
    "\"weka.classifiers.functions.supportVector.PolyKernel -E 1.0 -C 250007\" " +
    "-calibrator \"weka.classifiers.functions.Logistic -R 1.0E-8 -M 1 -num-decimal-places 4\""
]

for dataset_dir, type_lists in dataset_types.items():
    for dataset_type in type_lists:
        for label_type in label_types:
            for hyperparam in hyperparams:
                for model_weka in models:
                    train_path = parent_directory + dataset_dir + trainset_name + dataset_type + \
                                 hyperparam + label_type
                    test_path = parent_directory + dataset_dir + testset_name + dataset_type + \
                                hyperparam + label_type

                    # log_file = logging.FileHandler(train_path + ".log")
                    # log_console = logging.StreamHandler()
                    # logging.basicConfig(level=logging.INFO, format="%(message)s", handlers=[log_file, log_console])

                    full_command = "java -Xmx40000M -cp \"D:/Program Files/Weka-3-8/weka.jar\" " + model_weka + \
                                   " -t \"" + train_path + ".arff\" -T \"" + test_path + ".arff\""
                    print("[WEKA] Processing > " + train_path + "_" + model_weka.split()[0].split(".")[-1])

                    pipe = subprocess.Popen(full_command, shell=True, stdout=subprocess.PIPE).stdout
                    command_out = pipe.read()

                    # logging.info(command_out)
                    with open(train_path + "_" + model_weka.split()[0].split(".")[-1] + ".log", "wb") as f:
                        f.write(command_out)

