# bm25 to process data
python3.8 splitdata.py --data "$1" --mode "eval"

# 3 tasks
if [ -z "$3"]
then
	python3.8 main.py --predict --mode "eval" --data_mode "pos"
	python3.8 main.py --predict --mode "eval" --data_mode "neg"
	python3.8 main.py --predict --mode "eval" --data_mode "neural"
else
	python3.8 main.py --predict --mode "eval" --data_mode "pos" --predict_model_pos "$3"
	python3.8 main.py --predict --mode "eval" --data_mode "neg" --predict_model_neg "$4"
	python3.8 main.py --predict --mode "eval" --data_mode "neural" --predict_model_neural "$5"
fi

# combine answer
python3.8 combine.py --output_dir "$2"