from pathlib import Path
import shutil, os
from tqdm.auto import tqdm
from transformers.models.deberta_v2 import DebertaV2TokenizerFast
import pandas as pd


# ====================================================
# CFG
# ====================================================
class CFG:
    debug = False
    apex = False
    print_freq = 100
    num_workers = 4
    model = "microsoft/deberta-v3-large"
    scheduler = 'cosine'  # ['linear', 'cosine']
    batch_scheduler = True
    num_cycles = 0.5
    num_warmup_steps = 0.1
    epochs = 3
    encoder_lr = 6e-6
    decoder_lr = 6e-6
    min_lr = 1e-6
    eps = 1e-6
    betas = (0.9, 0.999)
    batch_size = 1
    fc_dropout = 0.2
    max_len = 512
    weight_decay = 0.01
    gradient_accumulation_steps = 1
    max_grad_norm = 500
    seed = 2001
    n_fold = 5
    trn_fold = [0]
    train = True

if CFG.debug:
    CFG.epochs = 5
    CFG.trn_fold = [0, 1, 2, 3, 4]


tokenizer = DebertaV2TokenizerFast.from_pretrained(CFG.model)
# tokenizer = AutoTokenizer.from_pretrained(CFG.model)
CFG.tokenizer = tokenizer

# ====================================================
# Define max_len
# ====================================================

# patient_notes = pd.read_csv('./data/patient_notes/patient_notes.csv')
# features = pd.read_csv('./data/preprocessed_features.csv')
#
# for text_col in ['pn_history']:
#     pn_history_lengths = []
#     tk0 = tqdm(patient_notes[text_col].fillna("").values, total=len(patient_notes))
#     for text in tk0:
#         length = len(tokenizer(text, add_special_tokens=False)['input_ids'])
#         pn_history_lengths.append(length)
#
# for text_col in ['feature_text']:
#     features_lengths = []
#     tk0 = tqdm(features[text_col].fillna("").values, total=len(features))
#     for text in tk0:
#         length = len(tokenizer(text, add_special_tokens=False)['input_ids'])
#         features_lengths.append(length)
#
# CFG.max_len = max(pn_history_lengths) + max(features_lengths) + 3  # cls & sep & sep

OUTPUT_DIR = './outputs/'


# ====================================================
# Fix fast tokenizer
# ====================================================
def fast_tokenizer():
    transformers_path = Path("C:/Users/ZHANG/.conda/envs/torch/Lib/site-packages/transformers")
    input_dir = Path("./input/deberta-v2-3-fast-tokenizer")

    convert_file = input_dir / "convert_slow_tokenizer.py"
    conversion_path = transformers_path / convert_file.name

    if conversion_path.exists():
        conversion_path.unlink()
    shutil.copy(convert_file, conversion_path)

    deberta_v2_path = transformers_path / "models" / "deberta_v2"
    if not deberta_v2_path.exists():  deberta_v2_path.mkdir()

    for filename in ['tokenization_deberta_v2.py', 'tokenization_deberta_v2_fast.py', "deberta__init__.py"]:
        if str(filename).startswith("deberta"):
            filepath = deberta_v2_path / str(filename).replace("deberta", "")
        else:
            filepath = deberta_v2_path / filename
        if filepath.exists():
            filepath.unlink()
        shutil.copy(input_dir / filename, filepath)


if __name__ == '__main__':
    fast_tokenizer()

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    print(CFG.tokenzier)
    # print(CFG.max_len)



