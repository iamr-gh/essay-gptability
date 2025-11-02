import pandas as pd


def main():
    # need to read in the
    # output to a csv with label 0/1 for AI, and generation as the text
    df = pd.read_csv("generated.csv")
    labels = []
    generations = []
    for i in range(len(df)):
        ai_gen = df["ai_output"][i]
        human_gen = df["human_output"][i]

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
    out_df.to_csv("generated_with_labels.csv", index=False)


if __name__ == "__main__":
    main()
