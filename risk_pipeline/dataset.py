import os
import random
import pickle
import pandas as pd
import torch
import pdb
import argparse
import logging
import json
from multiprocessing import Pool
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import (
	AutoTokenizer,
)

class Riskdataset(Dataset):
	def __init__(self, args, tokenizer,mode='train'):
		self.args = args
		self.mode = mode
		self.tokenizer = tokenizer
		self.max_seq_length = self.args.max_seq_length
		self.dataset= []
		self.cached_features_file = os.path.join(
		args.cache_dir,
		"cached_{}_{}_{}".format(
			mode,
			list(filter(None, args.model_name_or_path.split("/"))).pop(),
			str(args.max_seq_length),
		),
		)

	def __len__(self):
		return len(self.dataset)
	def __getitem__(self,idx):
		return self.dataset[idx]




	def get_dataset(self):
		# if os.path.exists(self.cached_features_file):
		#     print(f"Loading features from cached file {self.cached_features_file}")
		#     self.dataset = torch.load(self.cached_features_file)
		# else:
		self.dataset = self.load_file()
		self.dataset = self.process_dataset()


		amount_of_data = len(self.dataset)
		print(f"{self.mode} dataset size: {amount_of_data}")

	def load_file(self):
		if self.mode == "train":
			if self.args.train_mode == "prep":
				df = pd.read_csv(self.args.data_prep)
			else:
				df = pd.read_csv(self.args.data_symptom)
		elif self.mode == "validation":
			if self.args.train_mode == "prep":
				df = pd.read_csv(self.args.val_data_prep)
			else:
				df = pd.read_csv(self.args.val_data_symptom)
		else:
			if self.args.eval_mode == "prep":
				df = pd.read_csv(self.args.eval_data_prep)
			else:
				df = pd.read_csv(self.args.eval_data_symptom)
		article_id = df["article_id"].values.tolist()
		org_context = df["text"].values.tolist()
		text_len = df["textlength"].values.tolist()
		if self.mode == "train" or self.mode == "validation":
			class_result = df["label"].values.tolist()
			dataset = {"text":org_context,"article_id":article_id,"label":class_result,"textlen":text_len}
		else:
			dataset = {"text":org_context,"article_id":article_id,"textlen":text_len}
		return dataset

	def process_dataset(self):
		if self.args.max_seq_length > self.tokenizer.model_max_length:
			logger.warning(
				f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the"
				f"model ({self.tokenizer.model_max_length}). Using max_seq_length={self.tokenizer.model_max_length}."
			)
			self.max_seq_length = min(self.args.max_seq_length, self.tokenizer.model_max_length)

		sentences = self.dataset["text"]
		tokenized_examples = self.tokenizer(
			sentences,
			truncation= True, # 設成True會從長的seq切，照理說都是從context
			max_length=self.max_seq_length,
			padding="max_length" if self.args.pad_to_max_length else False,
			return_overflowing_tokens = False,
			return_tensors = "pt"
		)

		final_dataset = []
		# for idx,curr_context in enumerate(tokenized_examples["overflow_to_sample_mapping"]):
		# 	if len(tokenized_examples["input_ids"][idx]) > 10:
		# 		temp_data = {}
		# 		if self.mode == "train" or self.mode == "validation":
		# 			label = float(self.dataset["label"][curr_context])
		# 			label = torch.tensor(label,dtype=torch.float)
		# 			temp_data["labels"] = label
		# 		temp_data["input_ids"] = tokenized_examples["input_ids"][idx]
		# 		temp_data["token_type_ids"] =  tokenized_examples["token_type_ids"][idx]
		# 		temp_data["attention_mask"] =  tokenized_examples["attention_mask"][idx]
		# 		final_dataset.append(temp_data)

		for idx,curr_context in enumerate(tokenized_examples["input_ids"]):
			temp_data = {}
			if self.mode == "train" or self.mode == "validation":
				label = float(self.dataset["label"][idx])
				label = torch.tensor(label,dtype=torch.float)
				temp_data["labels"] = label
			temp_data["ids"] = self.dataset["article_id"][idx]
			temp_data["input_ids"] = tokenized_examples["input_ids"][idx]
			temp_data["token_type_ids"] =  tokenized_examples["token_type_ids"][idx]
			temp_data["attention_mask"] =  tokenized_examples["attention_mask"][idx]
			temp_data["text_length"] =  self.dataset["textlen"][idx]
			final_dataset.append(temp_data)
		
		

				
		
		torch.save(final_dataset, self.cached_features_file)
		return final_dataset


