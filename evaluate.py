import torch, ast
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
prie

