from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import pandas as pd
import os
import numpy as np
import torch
from tqdm.notebook import tqdm
from utils.data_augmentation import augment_eeg

class DEAP(Dataset):
    def __init__(self, data, labels, val_arousal_scores, augment=False):
        self.data = data
        self.labels = labels
        self.val_arousal = val_arousal_scores
        self.augment = augment

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx].clone().numpy()

        # Only augment during training
        if self.augment:
            x = augment_eeg(x)

        x = torch.tensor(x, dtype=torch.float32)
        y = self.labels[idx]
        val_aro = self.val_arousal[idx]
        return x, y, val_aro


def load_deap_dataset(label_type, label_path, bio_path, subjects=list(range(1, 33))):
    df = pd.read_csv(os.path.join(label_path, 'participant_ratings.csv'))
    label_lookup = {
        (row.Participant_id, row.Trial): (row.Valence, row.Arousal)
        for _, row in df.iterrows()
    }

    all_data = []
    all_labels = []
    all_val_aro = []

    for subject in subjects:
        for trial in tqdm(range(1, 41), desc=f"Loading subject {subject}"):
            val, aro = label_lookup[(subject, trial)]

            if label_type == 'valence':
                label = 0 if val < 5 else 1
            elif label_type == 'arousal':
                label = 0 if aro < 5 else 1
            else:
                if val >= 5 and aro >= 5:
                    label = 0
                elif val >= 5 and aro < 5:
                    label = 1
                elif val < 5 and aro >= 5:
                    label = 2
                else:
                    label = 3

            for segment in range(1, 61):
                path = os.path.join(bio_path, f's{subject}', f'{subject}_{trial}_{segment}.npy')
                if not os.path.exists(path):
                    continue

                eeg = np.load(path)[:32]
                # Normalize per channel
                eeg_tensor = torch.tensor(eeg, dtype=torch.float32)

                all_data.append(eeg_tensor)
                all_labels.append(label)
                all_val_aro.append(torch.tensor([val, aro], dtype=torch.float32))

    return all_data, all_labels, all_val_aro


def create_train_val_datasets(label_type, label_path, bio_path, subjects, test_size=0.3, save_datasets=False, dataset_path = 'datasets'):
    all_data, all_labels, all_val_aro = load_deap_dataset(label_type, label_path, bio_path, subjects)
    
    X_train, X_val, y_train, y_val, valaro_train, valaro_val = train_test_split(
        all_data, all_labels, all_val_aro,
        test_size=test_size,
        random_state=42,
        stratify=all_labels
    )

    train_dataset = DEAP(X_train, torch.tensor(y_train, dtype=torch.float32 if label_type != "both" else torch.long), valaro_train, augment=True)
    val_dataset = DEAP(X_val, torch.tensor(y_val, dtype=torch.float32 if label_type != "both" else torch.long), valaro_val, augment=False)
    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)
    if save_datasets:
        torch.save(train_dataset, os.path.join(dataset_path,'train_dataset.pt'))
        torch.save(val_dataset, os.path.join(dataset_path, 'val_dataset.pt'))
    return train_dataset, val_dataset

def create_test_dataset(label_type, label_path, bio_path, subjects):
    all_data, all_labels, all_val_aro = load_deap_dataset(label_type, label_path, bio_path, subjects)
    

    test_dataset = DEAP(all_data, torch.tensor(all_labels, dtype=torch.float32 if label_type != "both" else torch.long), all_val_aro, augment=False)

    return test_dataset