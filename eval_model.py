import os

directory = os.fsencode("evaluation_rnn_classifiers")

def evaluate_model(trajectory):
    label = np.array([0, 0])
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if filename.endswith(".h5"):
            model.load_weights(filename)
            label += model.predict(trajectory)
    return np.mean(label)[0]
