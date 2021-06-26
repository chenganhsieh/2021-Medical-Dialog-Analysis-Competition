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

choice2index = {
	'A':0,
	'B':1,
	'C':2,
}
index2choice = {
	0:'A',
	1:'B',
	2:'C',
}




logger = logging.getLogger(__name__)
class QADataset(Dataset):
	def __init__(self, args, tokenizer,mode='train'):
		self.args = args
		self.mode = mode
		self.tokenizer = tokenizer
		self.max_seq_length = self.args.max_seq_length
		self.cached_features_file = os.path.join(
		args.cache_dir,
		"cached_{}_{}_{}_{}".format(
			mode,
			list(filter(None, args.model_name_or_path.split("/"))).pop(),
			str(args.max_seq_length),
			args.mode,
		),
		)



	def get_dataset(self):

		self.dataset = self.load_file()
		self.dataset = self.process_dataset()
		amount_of_data = len(self.dataset)
		print(f"{self.mode} dataset size: {amount_of_data}")

		return self.dataset


	def load_file(self):
		files = {}
		if self.mode == "train":
			if self.args.data_mode == "pos":
				files[self.mode] = self.args.data_pos
			elif self.args.data_mode == "neg":
				files[self.mode] = self.args.data_neg
			else:
				files[self.mode] = self.args.data_neural
		elif self.mode == "validation":
			if self.args.data_mode == "pos":
				files[self.mode] = self.args.val_pos
			elif self.args.data_mode == "neg":
				files[self.mode] = self.args.val_neg
			else:
				files[self.mode] = self.args.val_neural
		else:
			if self.args.data_mode == "pos":
				files[self.mode] = self.args.eval_pos
			elif self.args.data_mode == "neg":
				files[self.mode] = self.args.eval_neg
			else:
				files[self.mode] = self.args.eval_neural
		dataset = load_dataset('json', data_files=files,field="data")
		return dataset

	def process_dataset(self):

		if self.args.max_seq_length > self.tokenizer.model_max_length:
			logger.warning(
				f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the"
				f"model ({self.tokenizer.model_max_length}). Using max_seq_length={self.tokenizer.model_max_length}."
			)
			self.max_seq_length = min(self.args.max_seq_length, self.tokenizer.model_max_length)


		# dataset 包含的key: answer/ input_ids / token_type_ids attention_mask / article_ids
		dataset = self.dataset.map(
			self.preprocess_function,
			batched=True,
			num_proc=self.args.num_workers,  # 
		)
		torch.save(dataset, self.cached_features_file)
		return dataset
	def preprocess_function(self,examples):
		question_column_name = "question"
		questiontext_column_name ="stem"
		questionchoice_column_name = "choices"
		context_column_name = "text"
		answer_column_name = "answer"

		# 目前都是固定3個選項
		first_sentences = [[context] * 3 for context in examples[context_column_name]]
		question_headers = examples[question_column_name]
		second_sentences = []
		for i, header in enumerate(question_headers):
			temp_list = []
			for idx in range(3):
				question = header[questiontext_column_name]
				answer  = header[questionchoice_column_name][idx]["text"]
				temp_list.append(f"{question} {answer}")
			second_sentences.append(temp_list)
		# Flatten out
		first_sentences = sum(first_sentences, [])
		second_sentences = sum(second_sentences, [])

		# 每3個選項都會pad到一樣的長度，但每一組長度未相同，需要到dataloader變成一樣長度
		tokenized_examples = self.tokenizer(
			first_sentences,
			second_sentences,
			truncation= True, # 設成True會從長的seq切，照理說都是從context
			max_length=self.max_seq_length,
			padding="max_length" if self.args.pad_to_max_length else False,
		)
		# Un-flatten
		dataset_dict = {k: [v[i : i + 3] for i in range(0, len(v), 3)] for k, v in tokenized_examples.items()}
		# label A-> 0 , B->1 , C->2
		if not self.args.predict:
			dataset_dict['labels'] = [self.choice_index(i.strip()) for i in examples[answer_column_name]]
		else:
			dataset_dict['ids'] = [i for i in examples["id"]]
		return dataset_dict
	def choice_index(self,choice):
		return choice2index[choice]
	def index_choice(self,index):
		return index2choice[index]

