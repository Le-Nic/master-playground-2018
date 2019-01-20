import numpy as np
import arff
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib

parent_directory = "G:/data/UNSW_splits/2_winsgt8s1_const/5_output_seperateLayers"  # change this [SEQUENCE]
trainset_name = "/train/UNSW_NB15_training-set"
devset_name = "/dev/UNSW_NB15_testing-set"

winsgt_type = "_winsgt8s1_s8"  # change this [SEQUENCE]
label_type = "_y0_"
output_type = "_last"

hyperparams = [
    "u32b64l3d40",
    "u64b64l3d40",
    "u64b128l3d40",
    "u64b256l3d40",
    "u128b256l3d40",
    "u128b512l3d40",
    "u256b512l3d40"
]

labels_n = 2 if label_type == "_y0_" else 5  # specify # of classes for multi-class scenario
trainsets_path = []  # [hyperparams len]
devsets_path = []  # [hyperparams len]

for hyperparam in hyperparams:
    trainsets_path.append(parent_directory + trainset_name + winsgt_type + hyperparam + label_type)
    devsets_path.append(parent_directory + devset_name + winsgt_type + hyperparam + label_type)

layer_n = 1
dev_instances = 0
y_dev = None

try:
    is_devinfo_obtained = False
    while True:
        with open(trainsets_path[0] + str(layer_n) + output_type + ".arff", 'r'):  # check train set
            with open(devsets_path[0] + str(layer_n) + output_type + ".arff", 'r') as devset:  # check dev set
                if not is_devinfo_obtained:
                    dataset = np.array(arff.load(devset)['data'])
                    dev_instances = dataset.shape[0]
                    y_dev = dataset[:, -1].astype(int)
                    is_devinfo_obtained = True
                layer_n += 1
except FileNotFoundError:
    pass

models = [[] for _ in trainsets_path]

for dataset_n, trainset_path in enumerate(trainsets_path):
    for i in range(layer_n-1):
        dataset_path = trainset_path + str(i+1) + output_type
        print("[Layered RF] Processing trainset >", dataset_path + ".arff")

        dataset = np.array(arff.load(open(dataset_path + ".arff", 'r'))['data'])
        x_train, y_train = dataset[:, :-1], dataset[:, -1]

        models[dataset_n].append(RandomForestClassifier())
        models[dataset_n][i].fit(x_train, y_train)
        _ = joblib.dump(models[i], dataset_path + ".pkl", compress=9)

for dataset_n, devset_path in enumerate(devsets_path):
    proba = [np.zeros((dev_instances, labels_n)) for _ in range(layer_n-1)]
    pred = []

    # average score
    for i in range(layer_n-1):
        dataset_path = devset_path + str(i+1) + output_type + ".arff"
        print("[Layered RF] Processing devset >", dataset_path)

        dataset = np.array(arff.load(open(dataset_path, 'r'))['data'])
        x_dev = dataset[:, :-1]

        proba[i] = models[dataset_n][i].predict_proba(x_dev)
        pred.append(np.argmax(proba[i], axis=1))

    with open(trainsets_path[dataset_n] + ".txt", "w") as f:
        print("Average Score:")
        f.write("\n--- Average Score ---\n")
        for i in range(layer_n-1):
            print("Layer", i+1)
            f.write("\nLayer" + str(i+1) + ":\n")
            cm = confusion_matrix(y_dev, pred[i], labels=[label for label in range(labels_n)])

            for row in cm:
                print(" ".join(str(col) for col in row))
                f.write(" ".join(str(col) for col in row) + "\n")

        print("Averaged Vote")
        f.write("\nAveraged Vote:\n")
        cm = confusion_matrix(
            y_dev, np.argmax(np.sum(proba, axis=0), axis=1), labels=[label for label in range(labels_n)])

        for row in cm:
            print(" ".join(str(col) for col in row))
            f.write(" ".join(str(col) for col in row) + "\n")

        print("Best Vote")
        f.write("\nBest Vote:\n")
        cm = confusion_matrix(
            y_dev,
            np.argmax(
                np.concatenate(proba, axis=1), axis=1
            ) % labels_n, labels=[label for label in range(labels_n)]
        )
        for row in cm:
            print(" ".join(str(col) for col in row))
            f.write(" ".join(str(col) for col in row) + "\n")

# seq = 5
# w = np.zeros(seq)
# for i in range(seq):
#     w[i] = (0.5**i) / ((2**seq) - 1)  # exponential penalized
