import os
import argparse
import random
import torch
import pdb
import pytorch_lightning as pl
from torch.utils.data.distributed import DistributedSampler # multi processing
from torch.utils.data import DataLoader 
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning import seed_everything
from dataset import Riskdataset
from pltrainer import PLTrainer
from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import LearningRateMonitor
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForMultipleChoice,
    BertModel,
    BertTokenizer,
    BertForSequenceClassification,
    AutoModel
)
from torch.utils.data import DataLoader 
# from pltrainer import PLTrainer
# os.environ["CUDA_VISIBLE_DEVICES"] = "2" 
seed_everything(42)

def main(args):
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        cache_dir=args.cache_dir,
    )
    model =  AutoModel.from_pretrained(
        args.model_name_or_path,
        cache_dir=args.cache_dir,
        # num_labels=args.num_labels
    )

    if not args.predict:
        train_dataset = Riskdataset(args,tokenizer)
        train_dataset.get_dataset()
        val_dataset = Riskdataset(args,tokenizer,mode="validation")
        val_dataset.get_dataset()
    else:
        eval_dataset = Riskdataset(args,tokenizer,mode="eval")
        eval_dataset.get_dataset()

    datasets = {
        'train': train_dataset if not args.predict else None,
        'val': val_dataset  if not args.predict else None,
        'eval': eval_dataset  if args.predict else None,
    }

    # log tensor blog !!!!!  可以印loss 整個分析圖
    tb_logger = pl_loggers.TensorBoardLogger(args.log_dir)

    checkpoint_callback = ModelCheckpoint(
        filename= '{step:05d}-{v_loss:.2f}-{v_acc:.2f}-{t_acc:.2f}', # step: 5個整數 
        save_top_k=3, # 前三名
        verbose=False,
        monitor='v_loss', # 看自己的loss dict name
        mode='min'
    )
    lr_monitor = LearningRateMonitor(logging_interval='step')


    if not args.predict:
        pltrainer = PLTrainer(args, model,datasets,tokenizer)
    else:
        if args.eval_mode == "prep":
            pltrainer = PLTrainer.load_from_checkpoint(args.predict_model_prep,args=args, model=model,datasets=datasets,tokenizer=tokenizer) 
        else:
            pltrainer = PLTrainer.load_from_checkpoint(args.predict_model_symptom,args=args, model=model,datasets=datasets,tokenizer=tokenizer) 
    trainer = pl.Trainer(
        fast_dev_run=False,
        logger=tb_logger, # 自己的logger
        gpus=1, 
        max_steps=args.max_steps, 
        auto_scale_batch_size='binsearch', # 自己設定batch size
        progress_bar_refresh_rate=1, # 進度條
        # accelerator='ddp',
        accumulate_grad_batches=2, # distrubuted training 
        # amp_backend='apex',
        # amp_level='O3',
        # precision=32, # fp_16 or fp_32
        # gradient_clip_val=0,
        val_check_interval= 1.0,
        log_every_n_steps=1,
        flush_logs_every_n_steps=1,
        # plugins=DDPPlugin(find_unused_parameters=True),
        callbacks=[checkpoint_callback, lr_monitor],
    )
    if args.predict:
        trainer.test(pltrainer)
    else:
        trainer.fit(pltrainer)
        # trainer.tune(pltrainer)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_prep", default="./data/train_risk_prep_clean.csv", type=str)
    parser.add_argument("--data_symptom", default="./data/train_risk_symptom_clean.csv", type=str)
    parser.add_argument("--val_data_prep", default="./data/valid_risk_prep_clean.csv", type=str)
    parser.add_argument("--val_data_symptom", default="./data/valid_risk_symptom_clean.csv", type=str)
    parser.add_argument("--eval_data_prep", default="./data/eval_risk_prep_clean.csv", type=str)
    parser.add_argument("--eval_data_symptom", default="./data/eval_risk_symptom_clean.csv", type=str)
    parser.add_argument("--train_mode", default = "prep",type =str)
    parser.add_argument("--eval_mode", default = "prep",type =str)
    parser.add_argument("--cache_dir", default="./cache", type=str)
    parser.add_argument('--log_dir', default='logs/',type=str,required=False)
    parser.add_argument("--max_seq_length", default=512, type=int)
    parser.add_argument("--num_labels", default=1, type=int)
    parser.add_argument("--num_workers", default=32, type=int)
    parser.add_argument("--drop_out", default=0.1, type=int)
    parser.add_argument("--lr", default=3e-5, type=float) # 3e-6
    parser.add_argument("--weight_decay", default=0.0, type=float) # 3e-6 # bert-base-chinese(batch6 ac:2) #hfl/chinese-xlnet-mid #hfl/chinese-macbert-base  # hfl/chinese-electra-180g-base-discriminator
    parser.add_argument('--model_name_or_path', type=str, default = "bert-base-chinese", help='specific model name of the given model type') # bert-base-chinese
    parser.add_argument('--predict_model_prep', type=str, default = "./bestmodel/prep.ckpt", help='specific model name of the given model type')
    parser.add_argument('--predict_model_symptom', type=str, default = "./bestmodel/symptom.ckpt", help='specific model name of the given model type')
    parser.add_argument('--pad_to_max_length', 
        type=bool, 
        default = True, 
        help="Whether to pad all samples to `max_seq_length`. "
        "If False, will pad the samples dynamically when batching to the maximum length in the batch (which can "
        "be faster on GPU but will be slower on TPU)."
    )
    parser.add_argument("--max_steps", default=100000, type=int)
    parser.add_argument("--batch_size", default=6, type=int)
    parser.add_argument("--predict", action="store_true")
    parser.add_argument("--output_file", default="decision.csv",type=str)
    parser.add_argument("--output_score", action="store_true")

    args = parser.parse_args()
    
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir, exist_ok=True)
    if not os.path.exists(os.path.join(args.log_dir,args.train_mode)):
        os.makedirs(os.path.join(args.log_dir,args.train_mode), exist_ok=True)
    args.log_dir = os.path.join(args.log_dir,args.train_mode)
    args.output_file = args.output_file.split(".")[0]+"_"+args.eval_mode+"."+args.output_file.split(".")[1]

    main(args)