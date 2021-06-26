import json
import re
import spacy
import pdb
import os
import random
import pickle
import argparse
import pandas as pd
from tqdm import tqdm
from math import log
from bm25 import BM25_Model
from multiprocessing import Pool
random.seed(42)
nlp = spacy.load('zh_core_web_md', disable=['ner', 'parser', 'tagger'])

def get_new_context(question,choices,context,answer=None):
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
	
	# split context to tag
	new_qa_pairs = []
	for idx,pair in enumerate(qa_pairs):
		pair = list(nlp(pair))
		fix_pair = []
		pass_tag = False
		for idx,i in enumerate(pair):
			if pass_tag:
				pass_tag = False
				continue
			if str(i) == "固" and idx+1<len(pair) and(str(pair[idx+1]) == "砲" or str(pair[idx+1])=="炮"):
				fix_pair.append("固砲")
				pass_tag = True
			elif str(i) == "約" and idx+1<len(pair) and (str(pair[idx+1]) == "炮" or str(pair[idx+1])=="砲"):
				fix_pair.append("約砲")
				pass_tag = True
			else:
				fix_pair.append(str(i))
		new_qa_pairs.append(fix_pair)
	bm25 = BM25_Model(new_qa_pairs)

	# question top-2
	fix_question = re.sub(r'下列', '', question)
	fix_question = re.sub(r'何者', '', fix_question)
	fix_question = re.sub(r'醫生', '', fix_question)
	fix_question = re.sub(r'民眾', '', fix_question)
	fix_question = re.sub(r'護士', '', fix_question)
	splitquestion = list(nlp(fix_question))
	question_tags = [str(i) for i in splitquestion]
	question_score_list = bm25.get_documents_score(question_tags)
	question_index_list = sorted(range(len(question_score_list)), key=lambda k: question_score_list[k],reverse=True)
	if "誤" in question or "不是" in question or "沒有" in question:
		question_index_list = question_index_list[:2]
	elif "確" in question or '何者是' in question or "下列何者原因" in question:
		question_index_list = question_index_list[:1]
	else:
		question_index_list = question_index_list[:2]
	
	
	#補上前後句
	new_question_index_list = []
	# if ("關於此次看診" not in question and "身體狀況" not in question and "民眾的敘述" not in question and "下列關於民眾" not in question and "下列敘述何者" not in question) or ("下列敘述何者" in question and "關於" in question):
	# 	for idx in question_index_list:
	# 		if idx not in new_question_index_list:
	# 			if idx-1>=0 and idx-1 not in new_question_index_list:
	# 				new_question_index_list+= [idx-1]
	# 			new_question_index_list+= [idx]
	# 			if idx+1 < len(new_qa_pairs) and idx+1 not in new_question_index_list:
	# 				new_question_index_list+= [idx+1]
	# 			if "確" not in question and '何者是' not in question and "下列何者原因" not in question:
	# 				if idx+2 < len(new_qa_pairs) and idx+2 not in new_question_index_list:
	# 					new_question_index_list+= [idx+2]
	# 				if idx+3 < len(new_qa_pairs) and idx+3 not in new_question_index_list:
	# 					new_question_index_list+= [idx+3]

	# choices top-2
	new_choice_index_list = []
	find = False
	for label,choice in enumerate(choices):
		choice = re.sub(r'目前有', '', choice)
		choice = list(nlp(choice))
		fix_choice = []
		pass_tag = False
		# 為了避免固砲都找不到...
		for idx,i in enumerate(choice):
			if pass_tag:
				pass_tag = False
				continue
			if str(i) == "固" and idx+1<len(pair) and (str(pair[idx+1]) == "砲" or str(pair[idx+1])=="炮"):
				fix_choice.append("固砲")
				pass_tag = True
			elif str(i) == "約" and  idx+1<len(pair) and(str(pair[idx+1]) == "炮" or str(pair[idx+1])=="砲"):
				fix_pair.append("約砲")
				pass_tag = True
			else:
				fix_choice.append(str(i))
		choice_score_list = bm25.get_documents_score(fix_choice)
		choice_index_list = sorted(range(len(choice_score_list)), key=lambda k: choice_score_list[k],reverse=True)
		choice_index_list = choice_index_list[:3]
		fix_choice_index_list = choice_index_list
		curr = 0
		# for idx,choice_index in enumerate(choice_index_list):
		# 	if idx == 0:
		# 		curr = choice_score_list[choice_index]
		# 		fix_choice_index_list.append(choice_index)
		# 	elif idx>=1:
		# 		if curr - choice_score_list[choice_index]>2:
		# 			break
		# 		else:
		# 			fix_choice_index_list.append(choice_index)
		# 			curr = choice_score_list[choice_index]
		for choice in fix_choice_index_list:
			if choice in new_question_index_list:
				find = True
				for choice in fix_choice_index_list:
					if choice not in new_question_index_list:
						new_question_index_list+= [choice]
					if choice+1 < len(new_qa_pairs) and choice+1 not in new_question_index_list:
						new_question_index_list+= [choice+1]
				break
			else:
				if choice not in new_choice_index_list:
					if choice-2>=0 and choice-2 not in new_choice_index_list:
						new_choice_index_list+= [choice-2]
					if choice-1>=0 and choice-1 not in new_choice_index_list:
						new_choice_index_list+= [choice-1]
					new_choice_index_list+= [choice]
					if choice+1 < len(new_qa_pairs) and choice+1 not in new_choice_index_list:
						new_choice_index_list+= [choice+1]
		if find:
			for idx in new_question_index_list:
				new_context+= qa_pairs[idx]
			break
	if not find:
		for idx in new_choice_index_list:
			new_context+= qa_pairs[idx]
	# print(question)
	# print(choices)
	# print(new_context)
	# print(answer)
	# pdb.set_trace()
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

def clean_data(content):
	if args.mode == "train":
		answer = content['answer']
		if content['id'] == 6: # 6號他媽根本沒答案 幹
			return content
	else:
		answer = None
	text = content["text"]
	text = strQ2B(text)
	text = re.sub(r'(.)\1+', r'\1', text) # 疊字全部刪除 -> (.) 任意字 \1第一個字 + 有重複 變成 \1 第一個字
	text = re.sub(r'啊', '', text)
	text = re.sub(r'阿', '', text)
	text = re.sub(r'蛤', '', text)
	text = re.sub(r'吧', '', text)
	text = re.sub(r'喔', '', text) 
	text = re.sub(r'…', '', text)
	text = re.sub(r'⋯','',text)
	text = re.sub(r'痾', '', text)
	text = re.sub(r'ok', '好的', text)
	text = re.sub(r'OK', '好的', text)
	text = re.sub(r'HPV', '乳突病毒', text)
	text = re.sub(r'HIV', '愛滋病', text)
	text = re.sub(r'PrEP', '預防藥', text)
	text = re.sub(r'PrEp', '預防藥', text)
	text = re.sub(r'prep', '預防藥', text)
	text = re.sub(r'PREP', '預防藥', text)
	text = re.sub(r'Prep', '預防藥', text)
	text = re.sub(r'預防H', '預防愛滋病', text)
	text = re.sub(r'1', '一', text)
	text = re.sub(r'2', '二', text)
	text = re.sub(r'3', '三', text)
	text = re.sub(r'4', '四', text)
	text = re.sub(r'5', '五', text)
	text = re.sub(r'6', '六', text)
	text = re.sub(r'7', '七', text)
	text = re.sub(r'8', '八', text)
	text = re.sub(r'9', '九', text)
	text = re.sub(r'0', '零', text)
	

	question = content["question"]["stem"]
	question = strQ2B(question)
	question = re.sub(r'prep', '預防藥', question)
	question = re.sub(r'PrEP', '預防藥', question)
	question = re.sub(r'PrEp', '預防藥', question)
	question = re.sub(r'Prep', '預防藥', question)
	question = re.sub(r'prep', '預防藥', question)
	question = re.sub(r'PREP', '預防藥', question)
	question = re.sub(r'HPV', '乳突病毒', question)
	question = re.sub(r'HIV', '愛滋病', question)
	question = re.sub(r'ok', '好的', question)
	question = re.sub(r'OK', '好的', question)
	question = re.sub(r'1', '一', question)
	question = re.sub(r'2', '二', question)
	question = re.sub(r'3', '三', question)
	question = re.sub(r'4', '四', question)
	question = re.sub(r'5', '五', question)
	question = re.sub(r'6', '六', question)
	question = re.sub(r'7', '七', question)
	question = re.sub(r'8', '八', question)
	question = re.sub(r'9', '九', question)
	question = re.sub(r'0', '零', question)


	choices = [choice for choice in content["question"]["choices"]]
	new_choice_format = []
	new_choices = []
	for choice in choices:
		temp_format = {"text":"","label":""}
		choice_text = choice["text"]
		choice_text = strQ2B(choice_text)
		choice_text = re.sub(r'prep', '預防藥', choice_text)
		choice_text = re.sub(r'PrEP', '預防藥', choice_text)
		choice_text = re.sub(r'PrEp', '預防藥', choice_text)
		choice_text = re.sub(r'Prep', '預防藥', choice_text)
		choice_text = re.sub(r'prep', '預防藥', choice_text)
		choice_text = re.sub(r'PREP', '預防藥', choice_text)
		choice_text = re.sub(r'Prep', '預防藥', choice_text)
		choice_text = re.sub(r'HPV', '乳突病毒', choice_text)
		choice_text = re.sub(r'HIV', '愛滋病', choice_text)
		choice_text = re.sub(r'ok', '好的', choice_text)
		choice_text = re.sub(r'OK', '好的', choice_text)
		choice_text = re.sub(r'1', '一', choice_text)
		choice_text = re.sub(r'2', '二', choice_text)
		choice_text = re.sub(r'3', '三', choice_text)
		choice_text = re.sub(r'4', '四', choice_text)
		choice_text = re.sub(r'5', '五', choice_text)
		choice_text = re.sub(r'6', '六', choice_text)
		choice_text = re.sub(r'7', '七', choice_text)
		choice_text = re.sub(r'8', '八', choice_text)
		choice_text = re.sub(r'9', '九', choice_text)
		choice_text = re.sub(r'0', '零', choice_text)
		choice_text = re.sub(r'民眾的B', '民眾的病', choice_text)
		new_choices.append(choice_text)
		temp_format["text"] = choice_text
		temp_format["label"] = strQ2B(choice["label"])
		new_choice_format.append(temp_format)

	new_context = get_new_context(question,new_choices,text,answer)
	content["text"] = new_context
	content["question"]["stem"] = question
	content["question"]["choices"] = new_choice_format
	if args.mode == "train":
		content["answer"] = strQ2B(content["answer"])
	return content


def main(args):
	with open(args.data) as f:
		data = json.load(f)
	print("extract words...")

	pool = Pool(processes = 32)
	alldata = list(tqdm(pool.imap(clean_data, data), total=len(data)))
	pool.close()
	pool.join()	

	correct = []
	error = []
	neural = []
	for contentdata in alldata:
		if args.mode == "train":
			if contentdata["id"] == 6 or contentdata["id"] == 216:
				continue
		question = contentdata["question"]["stem"]
		if "誤" in question or "不是" in question or "沒有" in question:
			error.append(contentdata)
		elif "確" in question or '何者是' in question or "下列何者原因" in question:
			correct.append(contentdata)
		else:
			neural.append(contentdata)

	print(f"Correct data ratio:{len(correct)/len(alldata)}")
	print(f"Error data ratio:{len(error)/len(alldata)}")
	print(f"Neural data ratio:{len(neural)/len(alldata)}")


	if args.mode == "train":
		outputfile_pos = "train_qa_pos_clean.json"
		outputfile_neg = "train_qa_neg_clean.json"
		outputfile_neural = "train_qa_neural_clean.json"
	elif args.mode == "valid":
		outputfile_pos = "valid_qa_pos_clean.json"
		outputfile_neg = "valid_qa_neg_clean.json"
		outputfile_neural = "valid_qa_neural_clean.json"
	else:
		outputfile_pos = "eval_qa_pos_clean.json"
		outputfile_neg = "eval_qa_neg_clean.json"
		outputfile_neural = "eval_qa_neural_clean.json"

	outputfile_pos_path = os.path.join(args.output_dir,outputfile_pos)
	outputfile_neg_path = os.path.join(args.output_dir,outputfile_neg)
	outputfile_neural_path = os.path.join(args.output_dir,outputfile_neural)

	if args.mode == "train":
		random.shuffle(correct)
		random.shuffle(error)
		random.shuffle(neural)

	correct_new_format = {"data":correct}
	with open(outputfile_pos_path, 'w') as json_file:
		json.dump(correct_new_format, json_file,ensure_ascii=False)
	error_new_format = {"data":error}
	with open(outputfile_neg_path, 'w') as json_file:
		json.dump(error_new_format, json_file,ensure_ascii=False)
	neural_new_format = {"data":neural}
	with open(outputfile_neural_path, 'w') as json_file:
		json.dump(neural_new_format, json_file,ensure_ascii=False)


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--data", default="../data/Train_qa_ans.json", type=str)
	parser.add_argument("--output_dir", default="./data", type=str)
	parser.add_argument("--mode", default="train", type=str)
	args = parser.parse_args()
	main(args)


# 原始for迴圈

# for idx,content in enumerate(data):
	# print(f"Data id:{idx} | total {len(data)}",end="\r")
	# if args.mode == "train":
	# 	answer = content['answer']
	# 	if content['id'] == 6: # 6號他媽根本沒答案 幹
	# 		continue
	# text = content["text"]
	# text = strQ2B(text)
	# text = re.sub(r'(.)\1+', r'\1', text) # 疊字全部刪除 -> (.) 任意字 \1第一個字 + 有重複 變成 \1 第一個字
	# text = re.sub(r'啊', '', text)
	# text = re.sub(r'阿', '', text)
	# text = re.sub(r'蛤', '', text)
	# text = re.sub(r'吧', '', text)
	# text = re.sub(r'喔', '', text) 
	# text = re.sub(r'…', '', text)
	# text = re.sub(r'⋯','',text)
	# text = re.sub(r'痾', '', text)
	# text = re.sub(r'ok', '好的', text)
	# text = re.sub(r'HPV', '乳突病毒', text)
	# text = re.sub(r'HIV', '愛滋病', text)
	# text = re.sub(r'PrEP', '預防藥', text)
	# text = re.sub(r'PrEp', '預防藥', text)
	# text = re.sub(r'prep', '預防藥', text)
	# text = re.sub(r'預防H', '預防愛滋病', text)
	

	# question = content["question"]["stem"]
	# question = strQ2B(question)
	# question = re.sub(r'prep', '預防藥', question)
	# question = re.sub(r'PrEP', '預防藥', question)
	# question = re.sub(r'PrEp', '預防藥', question)
	# question = re.sub(r'prep', '預防藥', question)
	# question = re.sub(r'HPV', '乳突病毒', question)
	# question = re.sub(r'HIV', '愛滋病', question)


	# choices = [choice for choice in content["question"]["choices"]]
	# new_choice_format = []
	# new_choices = []
	# for choice in choices:
	# 	temp_format = {"text":"","label":""}
	# 	choice_text = choice["text"]
	# 	choice_text = strQ2B(choice_text)
	# 	choice_text = re.sub(r'prep', '預防藥', choice_text)
	# 	choice_text = re.sub(r'PrEP', '預防藥', choice_text)
	# 	choice_text = re.sub(r'PrEp', '預防藥', choice_text)
	# 	choice_text = re.sub(r'Prep', '預防藥', choice_text)
	# 	choice_text = re.sub(r'prep', '預防藥', choice_text)
	# 	choice_text = re.sub(r'HPV', '乳突病毒', choice_text)
	# 	choice_text = re.sub(r'HIV', '愛滋病', choice_text)
	# 	new_choices.append(choice_text)
	# 	temp_format["text"] = choice_text
	# 	temp_format["label"] = strQ2B(choice["label"])
	# 	new_choice_format.append(temp_format)

	# new_context = get_new_context(question,new_choices,text,answer)
	# content["text"] = new_context
	# content["question"]["stem"] = question
	# content["question"]["choices"] = new_choice_format
	# content["answer"] = strQ2B(content["answer"])
	
	# if "誤" in question or "不是" in question or "沒有" in question:
	# 	error.append(content)
	# elif "確" in question or '何者是' in question or "下列何者原因" in question:
	# 	correct.append(content)
	# else:
	# 	neural.append(content)


	