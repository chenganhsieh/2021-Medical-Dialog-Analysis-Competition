import os
import argparse
import random
import torch
import pdb
import pytorch_lightning as pl
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader 
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning import seed_everything
from dataset import QADataset
from pltrainer import PLTrainer
from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import LearningRateMonitor
from datacollator import collateFunctionForMultipleChoice,set_parm
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForMultipleChoice,
    BertModel,
    BertForSequenceClassification
)
from torch.utils.data import DataLoader 
os.environ["CUDA_VISIBLE_DEVICES"] = "7" 
seed_everything(42)

def main(args):
    config = AutoConfig.from_pretrained(
        args.model_name_or_path,
        cache_dir=args.cache_dir,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path, #args.model_name_or_path
        cache_dir=args.cache_dir,
        use_fast=True,
        revision="main",
    )
    model = AutoModelForMultipleChoice.from_pretrained(
        args.model_name_or_path,
    )



    if not args.predict:
        train_dataset = QADataset(args,tokenizer)
        # train_dataset.get_dataset()
        train_dataset = train_dataset.get_dataset()["train"]
        val_dataset = QADataset(args,tokenizer,mode="validation")
        # val_dataset.get_dataset()
        val_dataset = val_dataset.get_dataset()["validation"]
    else:
        eval_dataset = QADataset(args,tokenizer,mode="eval")
        # eval_dataset.get_dataset()
        # pdb.set_trace()
        eval_dataset = eval_dataset.get_dataset()["eval"]

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
        monitor='v_acc', # 看自己的loss dict name
        mode='max'
    )
    lr_monitor = LearningRateMonitor(logging_interval='step')


    if not args.predict:
        pltrainer = PLTrainer(args, model,datasets,tokenizer)
    else:
        if args.data_mode == "pos":
            pltrainer = PLTrainer.load_from_checkpoint(args.predict_model_pos,args=args, model=model,datasets=datasets,tokenizer=tokenizer) 
        elif args.data_mode == "neg":
            pltrainer = PLTrainer.load_from_checkpoint(args.predict_model_neg,args=args, model=model,datasets=datasets,tokenizer=tokenizer) 
        else:
            pltrainer = PLTrainer.load_from_checkpoint(args.predict_model_neural,args=args, model=model,datasets=datasets,tokenizer=tokenizer) 


    # 看doc 知道參數
    trainer = pl.Trainer(
        fast_dev_run=False,
        logger=tb_logger, # 自己的logger
        gpus=1, 
        max_steps=args.max_steps, 
        auto_scale_batch_size='binsearch', # 自己設定batch size
        progress_bar_refresh_rate=1, # 進度條
        # accelerator='ddp',
        accumulate_grad_batches=args.accumulate_grad_batches, # distrubuted training  # 48 1
        # amp_backend='apex',
        # amp_level='O3',
        # precision=32, # fp_16 or fp_32
        # gradient_clip_val=0,
        val_check_interval=1.0,
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
    parser.add_argument("--data_pos", default="./data/train_qa_pos_clean.json", type=str)
    parser.add_argument("--data_neg", default="./data/train_qa_neg_clean.json", type=str)
    parser.add_argument("--data_neural", default="./data/train_qa_neural_clean.json", type=str)
    parser.add_argument("--val_pos", default="./data/valid_qa_pos_clean.json", type=str)
    parser.add_argument("--val_neg", default="./data/valid_qa_neg_clean.json", type=str)
    parser.add_argument("--val_neural", default="./data/valid_qa_neural_clean.json", type=str)
    parser.add_argument("--eval_pos", default="./data/eval_qa_pos_clean.json", type=str)
    parser.add_argument("--eval_neg", default="./data/eval_qa_neg_clean.json", type=str)
    parser.add_argument("--eval_neural", default="./data/eval_qa_neural_clean.json", type=str)
    parser.add_argument("--mode", default = "train",type =str)
    parser.add_argument("--data_mode", default = "neg",type =str)
    parser.add_argument("--cache_dir", default="./cache", type=str)
    parser.add_argument('--log_dir', default='logs/',type=str,required=False)
    parser.add_argument("--max_seq_length", default=512, type=int)
    parser.add_argument("--num_layers", default=6, type=int)
    parser.add_argument("--num_labels", default=2, type=int)
    parser.add_argument("--hidden_size", default=768, type=int)
    parser.add_argument("--num_workers", default=10, type=int)
    parser.add_argument("--head", default=4, type=int)
    parser.add_argument("--dropout", default=0.2, type=float)
    parser.add_argument("--stop_threshold", default=0.5, type=float)
    parser.add_argument("--lr", default=3e-5, type=float) # 3e-6
    parser.add_argument("--b1", default=0.9, type=float)
    parser.add_argument("--b2", default=0.999, type=float) # bert-base-chinese batch:3 accumulate_grad_batches:16  #hfl/chinese-xlnet-mid #hfl/chinese-macbert-base  # hfl/chinese-electra-180g-base-discriminator
    parser.add_argument('--model_name_or_path', type=str, default = "hfl/chinese-macbert-base", help='specific model name of the given model type')
    parser.add_argument('--predict_model_pos', type=str, default = "./bestmodel/pos.ckpt", help='specific model name of the given model type')
    parser.add_argument('--predict_model_neg', type=str, default = "./bestmodel/neg.ckpt", help='specific model name of the given model type')
    parser.add_argument('--predict_model_neural', type=str, default = "./bestmodel/neural.ckpt", help='specific model name of the given model type')
    parser.add_argument('--doc_stride', type=int, default = 32, help="When splitting up a long document into chunks, how much stride to take between chunks.")
    parser.add_argument('--pad_to_max_length', 
        type=bool, 
        default = True, 
        help="Whether to pad all samples to `max_seq_length`. "
        "If False, will pad the samples dynamically when batching to the maximum length in the batch (which can "
        "be faster on GPU but will be slower on TPU)."
    )
    parser.add_argument("--max_steps", default=100000, type=int)
    parser.add_argument("--batch_size", default=3, type=int)
    parser.add_argument("--accumulate_grad_batches", default=16, type=int)
    parser.add_argument("--predict", action="store_true")
    parser.add_argument("--output_file", default="qa.csv",type=str)
    

    args = parser.parse_args()

    if args.data_mode == "pos":
        args.model_name_or_path = "bert-base-chinese"
        if not args.predict:
            args.batch_size = 3
            args.accumulate_grad_batches = 16
    else:
        args.model_name_or_path = "hfl/chinese-macbert-base"
        if not args.predict:
            args.batch_size = 1
            args.accumulate_grad_batches = 48

    
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir, exist_ok=True)
    if not os.path.exists(os.path.join(args.log_dir,args.data_mode)):
        os.makedirs(os.path.join(args.log_dir,args.data_mode), exist_ok=True)
    args.log_dir = os.path.join(args.log_dir,args.data_mode)
    args.output_file = args.output_file.split(".")[0]+"_"+args.data_mode+"."+args.output_file.split(".")[1]

    main(args)