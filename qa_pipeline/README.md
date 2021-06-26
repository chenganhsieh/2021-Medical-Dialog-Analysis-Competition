# 醫病問答 NTUNLP_隊名想了十秒鐘

### Download best model
Download our pre-train model  
下載我們已訓練好的模型
```shell
bash download.sh
```

### Reproduce
You have to run download.sh first.  
須先下載我們訓練好的模型  
Test_QA.json: input file 比賽提供的資料  
qa.csv: output file 輸出的檔名和位置  
```shell
bash reproduce.sh /path/to/Test_QA.json /path/to/qa.csv 
```

### Train
We have fixed some error in the training data. If you want to have the same result based on our model, you should execute download.sh first to get our Train_QA.json and Develop_QA.json  
我們有修改training data裡的錯誤(ex:label="菜花")，如果要得到跟我們相同的訓練結果，請先執行download.sh，並下載我們的資料。  
  
We have three tasks on our training: Pos, Neg, Neural。    
訓練過程分成3個task: Pos: 詢問何者正確 Neg:詢問何者錯誤 Neural:其他    

```shell
# 如果已執行download.sh並且想要reproduce訓練過程，可以忽略preprocess
bash preprocess.sh /path/to/Train_QA.json /path/to/Develop_QA.json_
```
  
```python
# pos: bert-base-uncased batchsize:3 accumulate_grad_batches:16
# neg / neural: hfl/chinese-xlnet-mid #hfl/chinese-macbert-base batchsize:1 accumulate_grad_batches:48
python3.8 main.py --mode "train" --data_mode "pos"  
python3.8 main.py --mode "train" --data_mode "neg"
python3.8 main.py --mode "train" --data_mode "neural"
```

### Eval
We save our checkpoint in ./logs dir. If you want to reproduce the result, you should train and choose this three checkpoint:   
Pos: step=00045-v_loss=1.27-v_acc=0.61.ckpt  
Neg: step=00089-v_loss=1.17-v_acc=0.60-t_acc=0.88.ckpt  
Neural: step=00044-v_loss=1.26-v_acc=0.64-t_acc=0.85.ckpt  
如果要reproduce我們實驗，請使用./logs底下對應的checkpoint  
  
```shell
bash reproduce.sh /path/to/Test_QA.json /path/to/qa.csv /path/to/pos.ckpt /path/to/neg.ckpt /path/to/neural.ckpt
```



## Score
![AI CUP 2021 Score](https://i.imgur.com/1SyeIbO.png)


