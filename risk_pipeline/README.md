# 醫病風險判斷 NTUNLP_隊名想了十秒鐘

### Download best model
Download our pre-train model 下載我們已訓練好的模型
```shell
bash download.sh
```

### Reproduce
You have to run download.sh first. 須先下載我們訓練好的模型
Test_risk_calssification.csv: input file 比賽提供的資料
decision.csv: output file 輸出的檔名和位置
```shell
bash reproduce.sh /path/to/Test_risk_calssification.csv /path/to/decision.csv 
```

### Train
We have fixed some error in the training data. If you want to have the same result based on our model, you should execute download.sh first to get our Train_risk_classification_ans.csv and Develop_risk.csv
我們有修改training data裡的錯誤，如果要得到跟我們相同的訓練結果，請先執行download.sh，並下載我們的資料。

We have two tasks on our training: Prep and Symptom。
訓練過程分成2個task: Prep: 詢問性生活的風險(是否戴保險套) Symptom: 一般醫療問診

```shell
# 如果已執行download.sh並且想要reproduce訓練過程，可以忽略preprocess
bash preprocess.sh /path/to/Train_risk_classification_ans.csv /path/to/Develop_risk.csv
```
```python
# pos: bert-base-uncased batchsize:3 accumulate_grad_batches:16
# neg / neural: hfl/chinese-xlnet-mid #hfl/chinese-macbert-base batchsize:1 accumulate_grad_batches:48
python3.8 main.py --train_mode "prep" 
python3.8 main.py --train_mode "neg"
```

### Eval
We save our checkpoint in ./logs dir. If you want to reproduce the result, you should train and choose this three checkpoint: 
Prep: step=00181-v_loss=0.32-v_acc=0.87-t_acc=0.98.ckpt
Symptom: step=00152-v_loss=0.62-v_acc=0.75.ckpt
如果要reproduce我們實驗，請使用./logs底下對應的checkpoint
```shell
bash reproduce.sh /path/to/Test_risk_calssification.json /path/to/decision.csv /path/to/prep.ckpt /path/to/neg.ckpt /path/to/symptom.ckpt
```



## Score
![AI CUP 2021 Score](https://i.imgur.com/1SyeIbO.png)


