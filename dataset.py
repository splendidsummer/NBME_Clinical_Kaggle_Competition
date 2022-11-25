from utils import *
from torch.utils.data import  Dataset


# # Baseline
class TrainDataset(Dataset):
    def __init__(self, cfg, df):
        self.cfg = cfg
        self.feature_texts = df['feature_text'].values
        self.pn_historys = df['pn_history'].values
        self.annotation_lengths = df['annotation_length'].values
        self.locations = df['location'].values
        self.answers = df['annotation'].values

    def __len__(self):
        return len(self.feature_texts)

    def __getitem__(self, item):
        inputs = prepare_input(self.cfg,
                               self.pn_historys[item],
                               self.feature_texts[item])

        label = create_label(self.cfg,
                             self.feature_texts[item],
                             self.pn_historys[item],
                             self.annotation_lengths[item],
                             self.locations[item],
                             self.answers[item]
                             )
        return inputs, label



