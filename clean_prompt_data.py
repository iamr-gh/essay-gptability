import pandas as pd
import socket
from sys import argv


def clean_string(data: str) -> str:
    # remove Assistant:
    data = data.replace("Assistant:", "")
    # remove User:
    data = data.replace("User:", "")
    # remove System:
    data = data.replace("System:", "")
    return data


#
def main():
    # filename from command line argument
    filename = argv[1]
    print(f"Cleaning {filename}")
    if filename.split(".")[-1] != "csv":
        print("File must be a csv")
        return

    df = pd.read_csv(filename)
    df["response"] = df["response"].apply(clean_string)
    df.to_csv(f"{filename.split('.')[0]}_cleaned.csv", index=False)


if __name__ == "__main__":
    main()
