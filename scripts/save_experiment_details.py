import argparse
import os
import time

import pandas as pd
import yaml

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str)
    parser.add_argument("--log_dir", type=str)
    parser.add_argument("--job_id", type=str)

    save_paths = vars(parser.parse_args())

    save_paths["log_path"] = save_paths["log_dir"] + "predictions.csv"

    with open(save_paths["config_path"], "r") as stream:
        args = yaml.safe_load(stream)

    # All IS2RE CGCNN Evidential Lamb0p4 Val ID Prediction, 2022-07-02-17-20-00, job-9609357, 0
    logs = {
        "task": args["logging"]["task"],
        "desc": args["logging"]["desc"],
        "data_split": args["logging"]["data_split"],
        "model": args["model"]["name"],
        "lambda_": args["model"]["lambda_"],
        "start_time": time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()),
        "job_id": save_paths["job_id"],
    }

    if os.path.exists(save_paths["log_path"]):
        df_results = pd.read_csv(save_paths["log_path"])

    else:
        os.makedirs(save_paths["log_dir"], exist_ok=True)
        df_results = pd.DataFrame(
            columns=[
                "data_split",
                "task",
                "model",
                "lambda_",
                "desc",
                "start_time",
                "job_id",
            ]
        )

    df_results = df_results.append(logs, ignore_index=True)
    df_results.to_csv(save_paths["log_path"], index=False)
