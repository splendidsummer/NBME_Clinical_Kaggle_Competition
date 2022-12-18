import torch, ast, tqdm
from torch import nn
import pandas as pd
import numpy as np
from torch.utils.data import Dataloader
from models import *
from config import *
from dataset import *

model_path = 'nmbe_bert_v1.pth'

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
test_df = pd.read_csv('./data/preprocessed_test.csv')
test_set = TestDataset(CFG, test_df)
test_loader = Dataloader(test_set,
                         batch_size=5,
                         shuffle=False,
                         # num_workers=CFG.num_workers, pin_memory=True, drop_last=True
                         )

model = CustomModel(CFG)
model.load_state_dict(torch.load(model_path), map_location=DEVICE)
model.eval()

preds = []
offsets = []
seq_ids = []

for batch in tqdm(test_loader):
    input_ids, attention_mask, token_type_ids, offset_mapping, sequence_ids \
        = batch[0].to(DEVICE), batch[1].to(DEVICE), batch[2].to(DEVICE), \
          batch[3], batch[4]

    logits = model(input_ids, attention_mask, token_type_ids)
    preds.append(logits.detach().cpu().numpy())
    offsets.append(offset_mapping.numpy())
    seq_ids.append()



