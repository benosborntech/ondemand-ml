import argparse
import os
import xgboost as xgb
import pandas as pd


def main(args):
    max_depth = int(args.max_depth)
    eta = float(args.eta)
    objective = args.objective
    eval_metric = args.eval_metric
    num_class = int(args.num_class)
    num_round = int(args.num_round)
    train_data = args.train_data
    model_dir = args.model_dir

    df = pd.read_csv(train_data)

    X_train = df.drop("target", axis=1)
    y_train = df["target"]

    dtrain = xgb.DMatrix(X_train, label=y_train)

    params = {
        "objective": objective,
        "max_depth": max_depth,
        "eta": eta,
        "eval_metric": eval_metric,
        "num_class": num_class
    }

    bst = xgb.train(params, dtrain, num_round)

    model_dir = args.model_dir
    bst.save_model(os.path.join(model_dir, "model.json"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--max_depth", type=int, required=True, help="Max depth")
    parser.add_argument("--eta", type=float, required=True, help="Model learning rate")

    parser.add_argument("--objective", type=str, required=True, help="Objective of model")
    parser.add_argument("--eval_metric", type=str, required=True, help="Eval metric of the model")
    parser.add_argument("--num_class", type=int, required=True, help="Number of classes")
    parser.add_argument("--num_round", type=int, required=True, help="Number of rounds")

    parser.add_argument("--train_data", type=str, required=True, help="S3 location of the training data")
    parser.add_argument("--model_dir", type=str, required=True, help="Directory to save the model")

    args = parser.parse_args()
    main(args)
