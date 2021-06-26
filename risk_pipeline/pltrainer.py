import torch
import pytorch_lightning as pl
import pdb
import csv
import pickle
from torch import nn
from torch.nn import CrossEntropyLoss,BCEWithLogitsLoss
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
from torch.optim import Adam
from torch.utils.data import DataLoader 

# class BertPooler(nn.Module):
#     def __init__(self, config):
#         super().__init__()
#         self.dense = nn.Linear(config.hidden_size, config.hidden_size)
#         self.activation = nn.Tanh()

#     def forward(self, hidden_states):
#         # We "pool" the model by simply taking the hidden state corresponding
#         # to the first token.
#         first_token_tensor = hidden_states[:, 0]
#         pooled_output = self.dense(first_token_tensor)
#         pooled_output = self.activation(pooled_output)
#         return pooled_output

class PLTrainer(pl.LightningModule):
	def __init__(self, args, model,datasets,tokenizer):
		super().__init__()
		self.hparams.update(vars(args))
		self.model = model
		self.loss = nn.BCEWithLogitsLoss()
		self.train_dataset = datasets['train']
		self.val_dataset = datasets['val']
		self.test_dataset = datasets['eval']
		self.dropout = nn.Dropout(args.drop_out)
		self.classifier =  nn.Sequential(
			nn.Linear(769, 64), # 768: hidden size 1:paragraph length
			nn.ReLU(),
			nn.Linear(64, args.num_labels), # 3 choices
		)
		self.sigmoid = nn.Sigmoid()
		self.tokenizer  = tokenizer
		# self.pooler = BertPooler(model.config)
		
	def forward(self, x):
		# in lightning, forward defines the prediction/inference actions
		embedding = self.model(x)
		return embedding
	
	def configure_optimizers(self):
		optimizer = AdamW(self.parameters(),
						 lr=self.hparams.lr,weight_decay=self.hparams.weight_decay)
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
		#xlnet
		# output = output["last_hidden_state"]
		# pooled_output = self.pooler(output)

		pooled_output = output[1]
		pooled_output = self.dropout(pooled_output)

		text_len = batch["text_length"].unsqueeze(1)
		pooled_output = torch.cat((text_len,pooled_output),dim=1)
		logits = self.classifier(pooled_output)
		output_score = logits

		loss = self.criterion(output_score.view(-1),batch["labels"])
		output_score = self.sigmoid(output_score)

		train_correct = 0
		for idx,score in enumerate(output_score):
			score = score.item()
			if score > 0.5:
				if int(batch["labels"].cpu().data[idx].item()) == 1:
					train_correct+=1
			else:
				if int(batch["labels"].cpu().data[idx].item()) == 0:
					train_correct+=1
		# pdb.set_trace()

		# predicted =torch.argmax(output_score.cpu().data,dim=1)
		# train_correct = (predicted==batch["labels"].cpu().data).sum()

		logs = {
			't_loss': loss,
			'train_correct': train_correct,
			't_batch_size': len(batch["input_ids"])
		}

		for k, v in logs.items():
			if k == "t_loss":
				self.log(k, v, on_step=True, on_epoch=False, prog_bar=True, logger=True)
		output = {
			"loss":loss,
			'train_correct': train_correct,
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
		output = self.model(**inputs)
	
		# xlnet
		# output = output["last_hidden_state"]
		# pooled_output = self.pooler(output)

		pooled_output = output[1]
		pooled_output = self.dropout(pooled_output)

		# cat with text len
		text_len = batch["text_length"].unsqueeze(1)
		pooled_output = torch.cat((text_len,pooled_output),dim=1)
		logits = self.classifier(pooled_output)
		output_score = logits
		
		loss = self.criterion(output_score.view(-1),batch["labels"])
		output_score = self.sigmoid(output_score)

		val_correct = 0
		labels = []
		for idx,score in enumerate(output_score):
			score = score.item()
			if score > 0.5:
				labels.append(1)
				if int(batch["labels"].cpu().data[idx].item()) == 1:
					val_correct+=1
			else:
				labels.append(0)
				if int(batch["labels"].cpu().data[idx].item()) == 0:
					val_correct+=1

		output = {
			'v_loss_one': loss, 
			'v_correct': val_correct,
			'v_batch_size' :len(batch["input_ids"]),
			"label": labels,
		}
		
		return output
	def validation_epoch_end(self, outputs):
		v_loss = torch.stack([output['v_loss_one'] for output in outputs]).mean()
		v_correct =  sum([output['v_correct'] for output in outputs]) 
		v_data_size = sum([output['v_batch_size'] for output in outputs])

		label_list = []
		for output in outputs:
			label_list+=output['label']


		with open("log.txt",'w') as txtfile:
			txtfile.write(f'{label_list}\n')

		logs = {
			'v_loss': v_loss.item(), 
			'v_acc': v_correct / v_data_size, 
		}


		for k, v in logs.items():
			self.log(k, v, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
	def test_step(self, batch, batch_idx): #定義 Test 階段
		inputs = {
			"input_ids": batch["input_ids"],
			"attention_mask": batch["attention_mask"],
			"token_type_ids": batch["token_type_ids"],
		}
		output = self.model(**inputs)
		pooled_output = output[1]
		pooled_output = self.dropout(pooled_output)
		# cat with text len
		text_len = batch["text_length"].unsqueeze(1)
		pooled_output = torch.cat((text_len,pooled_output),dim=1)
		logits = self.classifier(pooled_output)
		output_score = logits

		# output_score = output["logits"]
		predicted = self.sigmoid(output_score.cpu().data)
		return {'output_score': predicted.tolist(),'ids':batch["ids"].tolist()}

	def test_epoch_end(self, outputs):
		with open(self.hparams.output_file,'w') as f:
			writer = csv.writer(f)
			writer.writerow(['article_id','probability']) 
			for output in outputs:
				for idx,ans in enumerate(output["output_score"]):
					if self.hparams.output_score:
						content = [output['ids'][idx],ans[0]]
					else:
						if ans[0]>0.5:
							content = [output['ids'][idx],1]
						else:
							content = [output['ids'][idx],0]
					writer.writerow(content) 
	
	
	def train_dataloader(self):
		return DataLoader(self.train_dataset, batch_size=self.hparams.batch_size, shuffle=True,num_workers=self.hparams.num_workers, pin_memory=True)
	
	
	def val_dataloader(self):
		val_dataloader = DataLoader(self.val_dataset, batch_size=self.hparams.batch_size, shuffle=False, num_workers=self.hparams.num_workers, pin_memory=True)
		return val_dataloader

	def test_dataloader(self):
		return DataLoader(self.test_dataset, batch_size=self.hparams.batch_size, shuffle=False,num_workers=self.hparams.num_workers, pin_memory=True)