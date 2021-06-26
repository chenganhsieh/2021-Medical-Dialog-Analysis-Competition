# preprocess
if [ -z "$1" ]
then
    python3.8 splitdata.py --data "./data/Train_risk_classification_ans.csv" --mode "train"
    python3.8 splitdata.py --data "./data/Develop_risk.csv" --mode "valid"
else
    python3.8 splitdata.py --data "$1" --mode "train"
    python3.8 splitdata.py --data "$2" --mode "valid"  
fi