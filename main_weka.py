import subprocess

parent_directory = "F:/UNSW/4_processed"

dataset_types = {
    "/winsgt_ip": ["_winsgt4s1_ip", "_winsgt16s1_ip"],  # IP Segt
    "/winsgt": [ "_winsgt4s1", "_winsgt16s1"]  # Normal Strides
}

trainset_name = "/UNSW-NB15_1"
testset_name = "/UNSW-NB15_3"

label_types = ["_y0", "_y1"]


for dataset_dir, type_lists in dataset_types.items():
    for dataset_type in type_lists:
        for label_type in label_types:
            train_path = parent_directory + dataset_dir + trainset_name + dataset_type + label_type + ".arff"
            test_path = parent_directory + dataset_dir + testset_name + dataset_type + label_type + ".arff"

            # model_weka = "weka.classifiers.trees.J48 -C 0.25 -M 2"
            model_weka = "weka.classifiers.trees.RandomForest -P 100 -I 100 -num-slots 1 -K 0 -M 1.0 -V 0.001 -S 1"
            # model_weka = "weka.classifiers.functions.SMO -C 1.0 -L 0.001 -P 1.0E-12 -N 0 -V -1 -W 1 -K " + \
            #     "\"weka.classifiers.functions.supportVector.PolyKernel -E 1.0 -C 250007\" " + \
            #     "-calibrator \"weka.classifiers.functions.Logistic -R 1.0E-8 -M 1 -num-decimal-places 4"

            full_command = "java -Xmx20000M -cp \"D:/Program Files/Weka-3-8/weka.jar\" " + model_weka + \
                " -t \"" + train_path + "\" -T \"" + test_path + "\""

            print("[WEKA] Processing >", train_path)
            pipe = subprocess.Popen(full_command, shell=True, stdout=subprocess.PIPE).stdout
            command_out = pipe.read()

            with open(parent_directory + "/logs" + trainset_name + dataset_type + label_type +
                      ".txt", "wb+") as text_file:

                text_file.write(command_out)
