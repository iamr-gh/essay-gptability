from sys import argv
import pandas as pd


def main():
    # need to read in the
    # output to a csv with label 0/1 for AI, and generation as the text
    filename = argv[1]

    df = pd.read_csv(filename)
    labels = []
    generations = []
    for i in range(len(df)):
        human_gen = df["full_text"][i]
        ai_gen = df["response"][i]

        generations.append(human_gen)
        labels.append(0)

        generations.append(ai_gen)
        labels.append(1)

    out_df = pd.DataFrame(
        {
            "generation": generations,
            "label": labels,
        }
    )
    out_df.to_csv(f"{filename.split('.')[0]}_with_labels.csv", index=False)


if __name__ == "__main__":
    main()
