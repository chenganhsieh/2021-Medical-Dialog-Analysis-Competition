import numpy as np
from collections import Counter
class BM25_Model(object):
	def __init__(self, documents_list, k1=2, k2=1, b=0.5):
			self.documents_list = documents_list # 需要输入的context列表,内部每个文本需要事先分好词
			self.documents_number = len(documents_list)
			# 所有context的平均長度
			self.avg_documents_len = sum([len(document) for document in documents_list]) / self.documents_number
			self.f = [] # 每個詞在每個文件的次數
			self.idf = {} # 每個詞的權重
			self.k1 = k1 # 2
			self.k2 = k2 # 1
			self.b = b # 0.75
			
			self.init()

	def init(self):
			df = {}
			for document in self.documents_list:
					temp = {}
					for word in document:
						if str(word) not in temp:
							temp[str(word)] = 1
						else:
							temp[str(word)] +=1
					self.f.append(temp)
					for key in temp.keys():
						if str(key) not in df:
							df[str(key)] = 1
						else:
							df[str(key)] += 1
					
			for key, value in df.items():
					self.idf[key] = np.log((self.documents_number - value + 0.5) / (value + 0.5))

	def get_score(self, index, query):
			score = 0.0
			document_len = len(self.f[index])
			qf = Counter(query)
			
			for q in query:
					if str(q) not in self.f[index]:
						continue
					else:
						score += self.idf[str(q)] * (self.f[index][str(q)] * (self.k1 + 1) / (
											self.f[index][str(q)] + self.k1 * (1 - self.b + self.b * document_len / self.avg_documents_len))) * (
																qf[str(q)] * (self.k2 + 1) / (qf[str(q)] + self.k2))
			return score

	def get_documents_score(self, query):
			score_list = []
			for i in range(self.documents_number):
					score_list.append(self.get_score(i, query))
			return score_list