import torch, ast
from torch import nn
import pandas as pd
import numpy as np
from models import *
from config import *

model_path = 'nmbe_bert_v1.pth'

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
features = pd.read_csv('./data/features.csv')
patient_notes = pd.read_csv('./data/patient_notes/patient_notes.csv')
test = pd.read_csv('./data/test.csv')
merged =
mer
train['annotation'] = train['annotation'].apply(ast.literal_eval)
model = CustomModel(CFG)

model.load_state_dict(torch.load(model_path), )