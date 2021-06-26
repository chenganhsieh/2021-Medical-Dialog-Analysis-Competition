import torch
import pytorch_lightning as pl
import pdb
import csv
import pickle
from torch import nn
from torch.nn import CrossEntropyLoss,BCELoss
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
from torch.optim import Adam
from torch.utils.data import DataLoader 
from datacollator import collateFunctionForMultipleChoice,set_parm

class PLTrainer(pl.LightningModule):
	def __init__(self, args, model,datasets,tokenizer):
		super().__init__()
		self.hparams.update(vars(args))
		self.save_hyperparameters(args)
		self.model = model
		self.loss = CrossEntropyLoss()
		self.train_dataset = datasets['train']
		self.val_dataset = datasets['val']
		self.test_dataset = datasets['eval']
		self.fn =  nn.Sequential(
			nn.Linear(768, 1),
			nn.Sigmoid()
		)
		self.tokenizer  = tokenizer
		set_parm(tokenizer,True,args.predict)
		# self.sample_dataset = datasets['sample']
		
	def forward(self, x):
		# in lightning, forward defines the prediction/inference actions
		embedding = self.model(x)
		return embedding
	
	def configure_optimizers(self):
		optimizer = AdamW(self.parameters(),
						 lr=self.hparams.lr)
		# warmup_steps = self.steps_per_epoch // 3
		# total_steps = self.steps_per_epoch * self.n_epochs - warmup_steps
		# scheduler = get_linear_schedule_with_warmup(optimizer,warmup_steps,total_steps)
		scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 0.95 ** epoch)
		return {
			'optimizer': optimizer,
			'lr_scheduler': scheduler
		}
	# 自己的loss
	def criterion(self, outputs, targets):
		loss = self.loss(outputs, targets)
		return loss
	
	# 一個step
	def training_step(self, batch, batch_idx):
		inputs = {
			"input_ids": batch["input_ids"],
			"attention_mask": batch["attention_mask"],
			"token_type_ids": batch["token_type_ids"],
		}
		# pdb.set_trace()
		output = self.model(**inputs)
		loss = self.criterion(output["logits"],batch["labels"])
		predicted =torch.argmax(output["logits"].cpu().data,dim=1)
		train_correct = (predicted==batch["labels"].cpu().data).sum()
		
		logs = {
			't_loss': loss,
			't_correct': train_correct.item(),
			't_batch_size' : len(batch["input_ids"])
		}

		for k, v in logs.items():
			self.log(k, v, on_step=True, on_epoch=False, prog_bar=False, logger=True)

		output = {
			"loss":loss,
			'train_correct': train_correct.item(),
			't_batch_size': len(batch["input_ids"])
		}
		return output
	def training_epoch_end(self, outputs):
		train_correct =  sum([output['train_correct'] for output in outputs]) 
		train_data_size = sum([output['t_batch_size'] for output in outputs])
		logs = {
			't_acc': train_correct / train_data_size, 
		}
		for k, v in logs.items():
			self.log(k, v, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

	def validation_step(self, batch, batch_idx):
		inputs = {
			"input_ids": batch["input_ids"],
			"attention_mask": batch["attention_mask"],
			"token_type_ids": batch["token_type_ids"],
		}
		# pdb.set_trace()


		output = self.model(**inputs) # 1 
		loss = self.criterion(output["logits"],batch["labels"])

		predicted =torch.argmax(output["logits"].cpu().data,dim=1)
		val_correct = (predicted==batch["labels"].cpu().data).sum()



		output = {
			'v_loss_one': loss, 
			'v_correct': val_correct,
			'v_batch_size' : len(batch["input_ids"]),
			'label': predicted.tolist()
		}
		
		return output
	def validation_epoch_end(self, outputs):
		v_loss = torch.stack([output['v_loss_one'] for output in outputs]).mean()
		v_correct = torch.stack([output['v_correct'] for output in outputs]).sum()
		# v_correct = sum([output['v_correct'] for output in outputs])
		v_data_size = sum([output['v_batch_size'] for output in outputs])


		logs = {
			'v_loss': v_loss, 
			'v_acc': v_correct.item()/ v_data_size, 
		}

		# label_list = []
		# for output in outputs:
		# 	for label in output["label"]:
		# 		if label == 0:
		# 			label_list+= 'A'
		# 		elif label == 1:
		# 			label_list+= 'B'
		# 		else:
		# 			label_list+= 'C'
		# with open("log.txt",'w') as txtfile:
		# 	txtfile.write(f'{label_list}\n')


		for k, v in logs.items():
			self.log(k, v, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
	def test_step(self, batch, batch_idx): #定義 Test 階段
		inputs = {
			"input_ids": batch["input_ids"],
			"attention_mask": batch["attention_mask"],
			"token_type_ids": batch["token_type_ids"],
		}
	

		output = self.model(**inputs) # 1 
		predicted =torch.argmax(output["logits"].cpu().data,dim=1)

		return {'output_score': predicted.tolist(),'ids':batch["ids"].tolist()}
	def test_epoch_end(self, outputs):
		# 這裡每3個是一題的答案，取最高分數者作為回答
		choice = ['A','B','C']
		with open(self.hparams.output_file,'w') as f:
			writer = csv.writer(f)
			writer.writerow(['id','answer']) 
			for idx,data in enumerate(outputs):
				for ii,answer in enumerate(data["output_score"]):
					content = [data["ids"][ii],choice[answer]]
					writer.writerow(content) 
	
	def train_dataloader(self):
		return DataLoader(self.train_dataset, batch_size=self.hparams.batch_size, shuffle=True,num_workers=self.hparams.num_workers, pin_memory=True,collate_fn = collateFunctionForMultipleChoice)
	
	
	def val_dataloader(self):
		val_dataloader = DataLoader(self.val_dataset, batch_size=self.hparams.batch_size, shuffle=False, num_workers=self.hparams.num_workers, pin_memory=True,collate_fn = collateFunctionForMultipleChoice)
		return val_dataloader

	def test_dataloader(self):
		return DataLoader(self.test_dataset, batch_size=self.hparams.batch_size, shuffle=False,num_workers=self.hparams.num_workers, pin_memory=True,collate_fn = collateFunctionForMultipleChoice)