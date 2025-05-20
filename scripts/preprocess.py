

import os
import pandas as pd
import _pickle as cPickle
import numpy as np

def eeg_data_preprocess():

    root = "./data/data_preprocessed_python/" # Directory containing the EEG data files
    des = "./data/DEAP/bio_normalized/" # Directory to save the processed data
    labels = pd.read_csv(
        "./data/DEAP/labels/participant_ratings.csv") # CSV file with participant ratings

    for file in os.listdir(root):
        subject = file.split(".")[0]           
        subject_id = int(subject[1:])        

        try:
            os.makedirs(os.path.join(des, f"s{subject_id}"))
        except Exception as e:
            print(e)
            print(des + f"s{subject_id}", "already created")

        f = open(root + file, "rb") 
        d = cPickle.load(f, encoding="latin")

        data = d["data"]  # Shape: (40 trials, 40 channels, 8064 samples)

        for experiment in range(40):

            trial = labels[
                (labels["Participant_id"] == subject_id) &
                (labels["Experiment_id"] == experiment + 1)
            ]["Trial"].iloc[0]

            # Extracting 3-second baseline (384 samples at 128 Hz)
            l = []
            for i in range(3):  # 3 baseline windows, each 1 second
                # (40 channels, 128 samples)
                l.append(data[experiment][:, i * 128: (i + 1) * 128])
            baseline = np.concatenate(l, axis=1)  # (40, 384)
            baseline_mean = baseline.mean(axis=1, keepdims=True)

            # Extract and baseline-correct each 1-second segment from the 60-second trial
            for i in range(int(60 / 1)):
                start = 384 + i * 128  # Skip first 3 seconds (384 samples)
                end = start + (128 * 1)  # Extract 1 seconds of data (128 samples per second)
                data_seg = data[experiment][:, start:end]  # Shape: (40, 128)
                data_seg_removed = data_seg - baseline_mean  # Remove baseline bias


                mean = data_seg_removed.mean(axis=1, keepdims=True)
                std = data_seg_removed.std(axis=1, keepdims=True) + 1e-6
                data_seg_removed = (data_seg_removed - mean) / std

                np.save(
                    f'{os.path.join(des, f"s{subject_id}")}/{subject_id}_{trial}_{i + 1}.npy', data_seg_removed)
if __name__ == "__main__":
    eeg_data_preprocess()