from transformers import AutoConfig, AutoTokenizer
import numpy as np
from  config import *
trans_config = AutoConfig.from_pretrained(CFG.model, output_hidden_states=True)
print(dir(trans_config))
print(1111)

