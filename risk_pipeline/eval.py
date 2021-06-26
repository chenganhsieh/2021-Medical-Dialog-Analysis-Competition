import pandas as pd
import json
from sklearn.metrics import roc_auc_score

org_df = pd.read_csv("../data/Develop_risk.csv")
predict_df = pd.read_csv("decision.csv")

org_data = org_df["label"].values.tolist()
predict_data = predict_df["probability"].values.tolist()

assert len(org_data) == len(predict_data), '兩個檔案長度不同'

correct = 0
# with open('./fault.csv', 'w') as f:
#     writer = csv.writer(f)
#     writer.writerow(['id','choices','answer','context']) 
for idx,i in enumerate(org_data):
    temp_result = 0
    if predict_data[idx]>0.5:
        temp_result = 1
    if i == temp_result:
        correct += 1
    # else:
    #     fault_data = [org_data[idx]["id"],result[idx],org_data[idx]["question"]["choices"],org_data[idx]["text"]]
    #     writer.writerow(fault_data)

print(roc_auc_score(org_data,predict_data))
print(f"Acc:{correct/len(org_data)}")
