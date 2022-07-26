import argparse
import os
import time

import pandas as pd
import yaml

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", type=str)
    parser.add_argument("--log_dir", type=str)
    parser.add_argument("--job_id", type=str)
    parser.add_argument("--mode", type=str)
    parser.add_argument("--epochs", type=str, default="20")
    parser.add_argument("--prev_timestamp", type=str, default="-")

    save_paths = vars(parser.parse_args())
    save_paths["log_path"] = save_paths["log_dir"] + "production.csv"

    with open(save_paths["output_path"], "r") as f:
        output = f.readlines()

    # All IS2RE CGCNN Evidential Lamb0p4 Val ID Prediction, 2022-07-02-17-20-00, job-9609357, 0
    logs = {
        "job_id": save_paths["job_id"],
        "mode": save_paths["mode"],
        "epochs": save_paths["epochs"],
        "prev_timestamp": save_paths["prev_timestamp"],
    }

    parameters_to_read = [
        "timestamp_id",
        "lambda_",
        "seed",
        "exp_task_name",
        "exp_desc",
        "exp_data_size",
        "exp_model",
        "job_id",
    ]

    for line in output:
        line = line.strip()
        for param in parameters_to_read:
            if (line.startswith(param)):
                logs[param] = line.split(": ")[1]

    if os.path.exists(save_paths["log_path"]):
        df_results = pd.read_csv(save_paths["log_path"])

    else:
        os.makedirs(save_paths["log_dir"], exist_ok=True)
        df_results = pd.DataFrame(
            columns=[
                "timestamp_id",
                "lambda_",
                "seed",
                "exp_task_name",
                "exp_desc",
                "exp_data_size",
                "exp_model",
                "job_id",
                "mode",
                "epochs",
                "prev_timestamp",
            ]
        )

    df_results = pd.concat([df_results, pd.DataFrame([logs])], ignore_index=True, axis=0)
    df_results.to_csv(save_paths["log_path"], index=False)

    # with open(save_paths["config_path"], "r") as stream:
    #     args = yaml.safe_load(stream)
    