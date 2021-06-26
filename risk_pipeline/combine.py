import pandas as pd
import argparse
def main(args):
    prep_df = pd.read_csv("decision_prep.csv")
    symptom_df = pd.read_csv("decision_symptom.csv")

    # Merge two Dataframes on index of both the dataframes
    mergeddf = pd.concat([prep_df,symptom_df])
    mergeddf = mergeddf.sort_values('article_id')
    mergeddf.to_csv(args.output_dir,index=False)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="./data/qa.csv", type=str,required = True)
    args = parser.parse_args()

    main(args)