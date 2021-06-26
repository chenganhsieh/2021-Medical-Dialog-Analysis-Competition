import pandas as pd
import json

with open("../data/Develop_clean_QA.json") as f:
	data = json.load(f)
predict_df = pd.read_csv("qa.csv")

predict_data = predict_df["answer"].values.tolist()

assert len(data) == len(predict_data), '兩個檔案長度不同'

correct = 0

for idx,i in enumerate(data):
    answer = i["answer"]
    if answer == predict_data[idx]:
        correct+=1
print(f"Acc:{correct/len(data)}")
