import numpy as np
def z_score_normalize(eeg):
    """
    Normalize each channel to zero mean and unit variance.
    eeg: shape [channels, samples]
    """
    mean = eeg.mean(axis=1, keepdims=True)
    std = eeg.std(axis=1, keepdims=True) + 1e-6
    return (eeg - mean) / std

def add_gaussian_noise(eeg, mean=0.0, std=0.05):
    noise = np.random.normal(mean, std, eeg.shape)
    return eeg + noise

def time_shift(eeg, max_shift=10):
    shift = np.random.randint(-max_shift, max_shift)
    return np.roll(eeg, shift, axis=1)

def channel_dropout(eeg, drop_prob=0.1):
    eeg = eeg.copy()
    num_channels = eeg.shape[0]
    for ch in range(num_channels):
        if np.random.rand() < drop_prob:
            eeg[ch, :] = 0
    return eeg

def augment_eeg(eeg):
    eeg = add_gaussian_noise(eeg, std=0.1)
    eeg = time_shift(eeg, max_shift=10)
    eeg = channel_dropout(eeg, drop_prob=0.1)
    return eeg