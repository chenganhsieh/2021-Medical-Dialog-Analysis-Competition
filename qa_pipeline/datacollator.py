from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.file_utils import PaddingStrategy
from typing import Optional, Union
from dataclasses import dataclass, field
import torch
import pdb

tokenizer = []
padding = True
predict = False

def set_parm(new_tokenizer,new_padding,new_predict=False):
    global tokenizer
    global padding
    global predict
    tokenizer = new_tokenizer
    padding = new_padding
    predict = new_predict
def collateFunctionForMultipleChoice(features):
    global tokenizer
    global padding
    # predict 要額外紀錄id 而沒predict則是要label
    if not predict:
        label_name = "label" if "label" in features[0].keys() else "labels"
        labels = [feature.pop(label_name) for feature in features]
    else:
        ids = [feature.pop("ids") for feature in features]

    batch_size = len(features)
    data_amount = len(features[0]["input_ids"])
                    
    flattened_features = [
        [{k: v[i] for k, v in feature.items() if k =="input_ids" or k=="attention_mask" or k=="token_type_ids"} for i in range(data_amount)] for feature in features
    ]
    flattened_features = sum(flattened_features, []) # 1878 -> 629 *3
            
    batch =  tokenizer.pad(
        flattened_features,
        padding= padding,
        return_tensors="pt",
    )
    
    # Un-flatten
    batch = {k: v.view(batch_size, data_amount, -1) for k, v in batch.items()}
    # Add back labels
    if not predict:
        # batch["labels"] = torch.tensor(labels, dtype=torch.float)
        batch["labels"] = torch.tensor(labels, dtype=torch.int64)
    else:
        batch["ids"] = torch.tensor(ids, dtype=torch.int64)
    return batch