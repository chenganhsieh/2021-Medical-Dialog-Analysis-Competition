import pandas as pd
import argparse

def main(args):
    pos_df = pd.read_csv("qa_pos.csv")
    neg_df = pd.read_csv("qa_neg.csv")
    neural_df = pd.read_csv("qa_neural.csv")

    # Merge two Dataframes on index of both the dataframes
    mergeddf = pd.concat([pos_df,neg_df])
    mergeddf = pd.concat([mergeddf,neural_df])
    mergeddf = mergeddf.sort_values('id')
    mergeddf.to_csv(args.output_dir,index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="./data/qa.csv", type=str,required = True)
    args = parser.parse_args()

    main(args)


