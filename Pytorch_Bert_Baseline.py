import torch
import re, time
from torch import nn
from torch.utils.data import DataLoader, Dataset
from ast import literal_eval
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import train_test_split, GroupKFold, KFold, StratifiedKFold
from torch import optim
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, AutoConfig
from config import *
from utils import *
from preprocess_data import correct_data
from dataset import *


data_dir = './data/'
note_dir = 'patient_notes/'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
LOGGER = get_logger()


class BaseModel(nn.Module):
    def __init__(self, cfg, config_path=None, pretrained=False):
        super().__init__()
        self.cfg = cfg
        if config_path is None:
            self.config = AutoConfig.from_pretrained(cfg.model, output_hidden_states=True)
        else:
            self.config = torch.load(config_path)
        if pretrained:
            # model should be built from here
            self.model = AutoModel.from_pretrained(cfg.model, config=self.config)
        else:
            self.model = AutoModel(self.config)

        self.hidden_dim = self.config.hidden_size

        self.hidden1 = nn.Linear(self.hidden_dim, 512)
        self.hidden2 = nn.Linear(512, 512)
        # self.hidden3 = nn.Linear(128, 128)
        self.dropout = nn.Dropout(p=0.2)  # p is the probability to be dropped
        self.prediction = nn.Linear(512, 1)
        self._init_weights(self.hidden1)
        self._init_weights(self.hidden2)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()

        # Embedding
        # LayerNorm

    def forward(self, inputs):
        outputs = self.model(**inputs)
        last_hidden_states = outputs[0]
        out = self.dense1(last_hidden_states)
        out = self.dropout(self.dense2(out))
        # here out is logits, no activation in dense2
        out = self.prediction(out)
        return out


def process_feature_text(text):
    text = re.sub('I-year', '1-year', text)
    text = re.sub('-OR-', ' or ', text)
    text = re.sub('-', ' ', text)
    return text


def prepare_datasets():
    feature_df = pd.read_csv(data_dir + 'features.csv')
    patient_note_df = pd.read_csv(data_dir + 'patient_notes.csv')
    train_df = pd.read_csv(data_dir + note_dir + 'train.csv')

    train_df['annotation_list'] = train_df.annotation.apply(lambda x: literal_eval(x))
    train_df['location_list'] = train_df.location.apply(lambda x: literal_eval(x))

    merge_df = train_df.merge(patient_note_df, on=['pn_num', 'case_num'], how='left')
    merge_df = merge_df.merge(merge_df, on=['feature_num', 'case_num'], how='left')
    merge_df = merge_df['pn_history'].apply(lambda x: x.replace('dad with recent heart attcak',
                                                                'dad with recent heart attack'))

    merge_df['feature_text'] = [process_feature_text(x) for x in merge_df['feature_text']]
    merge_df['feature_text'] = merge_df['feature_text'].apply(lambda x: x.lower())
    merge_df['pn_history'] = merge_df['pn_history'].apply(lambda x: x.lower())

    return merge_df


def split_fold(df):
    Fold = GroupKFold(n_splits=CFG.n_fold)
    groups = df['pn_num'].values()
    splits =  Fold.split(df, df['location'], groups)
    for n, (_, val_index) in enumerate(splits):
        df.loc[val_index, 'fold'] = int(n)

    df['fold'] = df['fold'].astype(int)

    return df


def train_model(model, dataloader, optimizer, loss_fn):
    model.train()
    train_loss = []
    for inputs, labels in tqdm(dataloader):
        inputs.to(device)
        labels.to(device)
        logits = model(inputs)
        loss = torch.masked_select(loss_fn(logits, labels), labels > -1.0)
        train_loss.append(loss.item() * inputs.shape[0])

        loss.backward()
        nn.utils.clip_grad_norm(model.parameters(), 1.0)
        optimizer.step()

        return sum(train_loss)/len(train_loss)


# df == merge_df
def train(df):

    for nfold in CFG.trn_fold:
        train_fold = df[df['fold'] != nfold].reset_index(drop=True)
        val_fold = df[df['fold'] == nfold].reset_index(drop=True)

        train_set = TrainDataset(CFG, train_fold)
        train_loader = DataLoader(
            train_set,
            batch_size=CFG.batch_size,
            shuffle=True,
            num_workers=CFG.num_workers,
            pin_memory=True,
            drop_last=True,
        )

        valset = TestDataset(CFG, val_fold)
        val_loader = DataLoader(
            train_set,
            batch_size=CFG.batch_size,
            shuffle=False,
            num_workers=CFG.num_workers,
            pin_memory=True,
            drop_last=True,
        )

        warmup_steps = int(CFG.num_warmup_prob * CFG.epochs * (len(train_set)/ CFG.batch_size))

        model = BaseModel(CFG, pretrained=True).to(device)
        torch.save(model.config, OUTPUT_DIR + 'config.pth')
        optimizer = optim.AdamW(model.parameters(), lr=CFG.encoder_lr)
        loss_fn = nn.BCEWithLogitsLoss(reduction='none')

        for epoch in range(CFG.epochs):
            start_time = time.time()
            epoch_loss = train_model(model, train_loader, optimizer, loss_fn)
            epoch_loss.backward()
            optimizer.step()

            if (epoch+1) % val_loader == 0:
                pass

        





if __name__ == '__main__':
    train_df = prepare_datasets()
    train_df = correct_data(train_df)

    test_text = 'I-year where are you going -OR- 10000??'
    processed_text = process_feature_text(test_text)
    print(1111)


