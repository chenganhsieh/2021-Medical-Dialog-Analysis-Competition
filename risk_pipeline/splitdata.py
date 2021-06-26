import json
import re
import spacy
import pdb
import os
import random
import pickle
import argparse
import tqdm
import pandas as pd
from math import log
from multiprocessing import Pool
random.seed(42)
nlp = spacy.load('zh_core_web_md', disable=['ner', 'parser', 'tagger'])

def get_new_context(tags,context):
	context = strQ2B(context)
	new_context = ""
	split_char = {"民眾": "","個管師":"","醫師":"","護理師":"","家屬":""}
	qa_pairs = []
	position = "民眾"
	gothrough_position = 0 
	save_stride = 1 # 2個2個對接
	dont_recored = 0
	temp = ""
	for idx,char in enumerate(context):
		if char == ":":
			continue
		if dont_recored == 0:
			if context[idx:idx+3] in split_char and (context[idx+3] == ":" or context[idx+3] == "A" or context[idx+3] == "B"):
				position = context[idx:idx+3]
				dont_recored = 2
				gothrough_position += 1
				if gothrough_position == save_stride+1:
					gothrough_position -= save_stride
					qa_pairs.append(temp)
					temp = ""
				continue
			elif context[idx:idx+2] in split_char and (context[idx+2] == ":" or context[idx+2] == "A" or context[idx+2] == "B"):
				position = context[idx:idx+2]
				
				dont_recored = 1
				gothrough_position += 1 
				if gothrough_position == save_stride+1:
					gothrough_position -= save_stride
					qa_pairs.append(temp)
					temp = ""
				continue
			temp += char
		else:
			if dont_recored == 1:
				temp+= position+":"
			dont_recored -= 1
	
	has_add_idx = []
	limit = 0
	avoid_empty = ""
	for idx,pair in enumerate(qa_pairs):
		for tag in tags:
			if tag in pair and len(avoid_empty) == 0:
				avoid_empty = pair
				if idx+1 < len(qa_pairs):
					avoid_empty += qa_pairs[idx+1]
			if tag in pair and len(pair) < 150: # 限制對話150字以內，超過100字大部分是醫生的宣導，沒必要
				# if idx>-1:
				# 	if idx-1 not in has_add_idx:
				# 		new_context+=qa_pairs[idx-1]
				if idx not in has_add_idx:
					# limit +=1
					new_context+=pair
					has_add_idx+=[idx] 
					if idx+1 < len(qa_pairs) and idx+1 not in has_add_idx  and len(qa_pairs[idx+1])<150:
						new_context+=qa_pairs[idx+1]
						has_add_idx+=[idx+1] 
				# if idx+1 < len(qa_pairs):
				# 	if idx+1 not in has_add_idx and len(qa_pairs[idx+1])<100:
				# 		new_context+=qa_pairs[idx+1]  
				# has_add_idx+=[idx-1,idx,idx+1]
				has_add_idx+=[idx] 
				# break
		# if limit >= 4:
		# 	break
	if len(new_context) == 0:
		new_context = avoid_empty

	return new_context


# 全形轉半形
def strQ2B(s):
	rstring = ""
	for uchar in s:
		u_code = ord(uchar)
		if u_code == 12288:  # 全形空格直接轉換
			u_code = 32
		elif 65281 <= u_code <= 65374:  # 全形字元（除空格）根據關係轉化
			u_code -= 65248
		rstring += chr(u_code)
	return rstring


def main(args):
	symptom = ["流感","血壓","血脂","血糖","痔瘡","細菌","免疫力","肝功能","貧血",
	"腎功能","減肥","飲食控制","流鼻水","三酸甘油酯","痠痛","CD4","感冒","痰","冷汗","梅毒",
	"咳嗽","頭暈","耳鳴","弱視","脹氣","耳鳴","結石","恍神","發燒","腫","病毒","大便潛血",
	"凍甲","水腫","菜花","腹瀉","水泡","化膿","血尿","感染","疣","急性","X光","結核菌","幻想","幻覺","凝血劑"]

	# 癲癇, 腦缺血,腎炎症候群

	prep = ["套","性行為","愛滋病"] # 約: 0.800 loss: 0.4

	risk_data = pd.read_csv(args.data)
	org_context = risk_data["text"].values.tolist()
	article_id = risk_data["article_id"].values.tolist()
	if args.mode == "train" or args.mode =="valid": 
		class_result = risk_data["label"].values.tolist()
		class_result = [int(i) for i in class_result]

	new_context_prep = []
	new_article_id_prep = []
	new_class_result_prep = []
	new_text_length_prep = []

	new_context_symptom = []
	new_article_id_symptom = []
	new_class_result_symptom = []
	new_text_length_symptom = []

	max_len_prep = 0
	max_len_symptom = 0
	prep_pos_record = []
	prep_neg_record = []
	symptom_pos_record = []
	symptom_neg_record = []
	print("extract words...")
	# 觀察了一下data分佈，發現篇幅越長，代表是有風險族群(講的話越多)
	# 以data來說，跟戴套有關的問題長度： 沒風險前三名: 4708 4433 4035 有風險:7246 7081 6951 5537 5045
	# 以data來說，跟問診有關的問題長度： 沒風險前三名: 4275 4184 3610 有風險:5856 5646 5268 5120 4395
	# 因此多一個條件，把文章長度放進去training
	for idx,text in enumerate(org_context):
		text = strQ2B(text)
		text = re.sub(r'(.)\1+', r'\1', text) # 疊字全部刪除 -> (.) 任意字 \1第一個字 + 有重複 變成 \1 第一個字
		text = re.sub(r'啊', '', text)
		text = re.sub(r'阿', '', text)
		text = re.sub(r'蛤', '', text)
		text = re.sub(r'吧', '', text)
		text = re.sub(r'喔', '', text) 
		text = re.sub(r'…', '', text)
		text = re.sub(r'痾', '', text)
		text = re.sub(r'ok', '好的', text)
		text = re.sub(r'HPV', '乳突病毒', text)
		text = re.sub(r'HIV', '愛滋病', text)
		text = re.sub(r'PrEP', '預防藥', text)
		text = re.sub(r'PrEp', '預防藥', text)
		text = re.sub(r'prep', '預防藥', text)
		if ("套" in text or "性行為" in text) and "手套" not in text:
			new_text = get_new_context(prep,text)
			if len(new_text) > max_len_prep:
				max_len_prep  =len(new_text)
			if len(new_text) == 0:
				new_text = "沒風險"
			new_context_prep.append(new_text)
			new_article_id_prep.append(article_id[idx])
			new_text_length_prep.append(len(text))
			if args.mode == "train" or args.mode =="valid": 
				new_class_result_prep.append(class_result[idx])
				if class_result[idx] == 0:
					prep_pos_record.append(len(text))
				else:
					prep_neg_record.append(len(text))
		else:
			new_text = get_new_context(symptom,text)
			if len(new_text) > max_len_symptom:
				max_len_symptom  =len(new_text)
			if len(new_text) == 0:
				new_text = "沒風險"
			new_context_symptom.append(new_text)
			new_article_id_symptom.append(article_id[idx])
			new_text_length_symptom.append(len(text))
			if args.mode == "train" or args.mode =="valid": 
				new_class_result_symptom.append(class_result[idx])
				if class_result[idx] == 0:
					symptom_pos_record.append(len(text))
				else:
					symptom_neg_record.append(len(text))
	# prep_pos_record.sort(reverse = True)
	# prep_neg_record.sort(reverse = True)
	# symptom_pos_record.sort(reverse = True)
	# symptom_neg_record.sort(reverse = True)
	# print(prep_pos_record)
	# print(prep_neg_record)
	# print(symptom_pos_record)
	# print(symptom_neg_record)
	print(f"max prep len: {max_len_prep}")
	print(f"max symptom len: {max_len_symptom}")

	if args.mode == "train":
		outputfile_prep = "train_risk_prep_clean.csv"
		outputfile_symptom = "train_risk_symptom_clean.csv"
	elif args.mode == "valid":
		outputfile_prep = "valid_risk_prep_clean.csv"
		outputfile_symptom = "valid_risk_symptom_clean.csv"
	else:
		outputfile_prep = "eval_risk_prep_clean.csv"
		outputfile_symptom = "eval_risk_symptom_clean.csv"

	outputfile_prep_path = os.path.join(args.output_dir,outputfile_prep)
	outputfile_symptom_path = os.path.join(args.output_dir,outputfile_symptom)
	if args.mode == "train":
		temp_prep = list(zip(new_context_prep, new_article_id_prep,new_class_result_prep,new_text_length_prep))
		random.shuffle(temp_prep)
		new_context_prep, new_article_id_prep,new_class_result_prep,new_text_length_prep = zip(*temp_prep)

		temp_symptom = list(zip(new_context_symptom, new_article_id_symptom,new_class_result_symptom,new_text_length_symptom))
		random.shuffle(temp_symptom)
		new_context_symptom, new_article_id_symptom,new_class_result_symptom,new_text_length_symptom = zip(*temp_symptom)


	if args.mode == "train":
		print(f"Prep train data amount:{len(new_article_id_prep)}")
		print(f"Symptom val data amount:{len(new_article_id_symptom)}")
		prep_train_df = pd.DataFrame({'article_id': new_article_id_prep,
				   'text': new_context_prep,
				   'label': new_class_result_prep,
				   'textlength':new_text_length_prep})
		prep_train_df.to_csv(outputfile_prep_path,index=False)

		symptom_train_df = pd.DataFrame({'article_id': new_article_id_symptom,
				   'text': new_context_symptom,
				   'label': new_class_result_symptom,
				   'textlength':new_text_length_symptom})
		symptom_train_df.to_csv(outputfile_symptom_path,index=False)

	elif args.mode == "valid":
		prep_valid_df = pd.DataFrame({'article_id': new_article_id_prep,
				   'text': new_context_prep,
				   'label': new_class_result_prep,
				   'textlength':new_text_length_prep})
		prep_valid_df.to_csv(outputfile_prep_path,index=False)

		symptom_valid_df = pd.DataFrame({'article_id': new_article_id_symptom,
				   'text': new_context_symptom,
				   'label': new_class_result_symptom,
				   'textlength':new_text_length_symptom})
		symptom_valid_df.to_csv(outputfile_symptom_path,index=False)

	else:
		prep_eval_df = pd.DataFrame({'article_id': new_article_id_prep,
				   'text': new_context_prep,
				   'textlength':new_text_length_prep})
		prep_eval_df.to_csv(outputfile_prep_path,index=False)

		symptom_eval_df = pd.DataFrame({'article_id': new_article_id_symptom,
				   'text': new_context_symptom,
				   'textlength':new_text_length_symptom})
		symptom_eval_df.to_csv(outputfile_symptom_path,index=False)

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--data", default="../data/Train_risk_classification_ans.csv", type=str)
	parser.add_argument("--output_dir", default="./data", type=str)
	parser.add_argument("--mode", default="train", type=str)
	args = parser.parse_args()
	main(args)


	