# bm25 to process data
python3.8 splitdata.py --data "$1" --mode "eval"

# 3 tasks
if [ -z "$3"]
then
    python3.8 main.py --predict --output_score --eval_mode "prep"
	python3.8 main.py --predict --output_score --eval_mode "symptom"
else
    python3.8 main.py --predict --output_score --eval_mode "prep" --predict_model_prep "$3"
	python3.8 main.py --predict --output_score --eval_mode "symptom" --predict_model_symptom "$4"
fi

# combine answer
python3.8 combine.py --output_dir "$2"