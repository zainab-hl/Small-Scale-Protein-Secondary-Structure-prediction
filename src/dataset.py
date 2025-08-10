import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder

# Define labels: amino acids and secondary structure
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"  # 20 common AAs
AA_to_idx = {aa: i for i, aa in enumerate(AMINO_ACIDS)}

SS_LABELS = ['C', 'E', 'H']  # coil, Beta-sheet, alpha-helix
ss_encoder = LabelEncoder()
ss_encoder.fit(SS_LABELS)

MAX_LEN = 512  # max length 
#one hot encoding for the amino acides
def one_hot_encode(seq):
    encoding = torch.zeros((MAX_LEN, len(AMINO_ACIDS)))
    for i, aa in enumerate(seq[:MAX_LEN]):
        if aa in AA_to_idx:
            encoding[i, AA_to_idx[aa]] = 1.0
    return encoding

def encode_ss(ss_seq):
    ss_encoded = ss_encoder.transform(list(ss_seq[:MAX_LEN]))
    if len(ss_encoded) < MAX_LEN:
        ss_encoded = list(ss_encoded) + [-1] * (MAX_LEN - len(ss_encoded))
    return torch.tensor(ss_encoded)

class ProteinSSDataset(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        seq = self.data.iloc[idx]['seq']
        ss = self.data.iloc[idx]['sst3']
        x = one_hot_encode(seq)
        y = encode_ss(ss)
        return x, y
