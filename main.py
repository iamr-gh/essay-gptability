from numpy import greater
import pandas as pd
import pdb
import jaclang
import time
from byllm.lib import Model, by

# openrouter/nvidia/nemotron-nano-12b-v2-vl:free

# also fine and fast
# llm = Model(model_name="openrouter/nvidia/nemotron-nano-12b-v2-vl:free")

# slower, smarter, has some annoying tokens at end
# llm = Model(model_name="openrouter/deepseek/deepseek-chat-v3.1:free")

# sufficiently fast
llm = Model(model_name="ollama/qwen2.5:7b-instruct")


@by(llm)
def answer_question(question: str) -> str: ...


def model_speed_tester():
    while True:
        start = time.perf_counter()
        q = input(">>")
        print(answer_question(q))
        elapsed = time.perf_counter() - start
        print(f"Elapsed: {elapsed:.6f} s")


# may consider wrapping in a struct to see how it is
@by(llm)
def write_essay(
    prompt_name: str,
    assignment: str,
    source_text_1: str,
    source_text_2: str,
    source_text_3: str,
    source_text_4: str,
) -> str: ...


"""Given an assignment and source texts, answers the question with an essay. No additional formatting is included, it is only paragraphs."""


def main():
    df = pd.read_csv("ASAP2_train_sourcetexts.csv")

    to_consider = 100
    # there is multiple human answers per prompt

    ai_answers = []
    # there's multiple source texts
    for i in range(to_consider):
        # context window might not be long enough
        ai_answer = write_essay(
            df["prompt_name"][i],
            df["assignment"][i],
            df["source_text_1"][i],
            df["source_text_2"][i],
            df["source_text_3"][i],
            df["source_text_4"][i],
        )

        ai_answers.append(ai_answer)

        yellow_begin = "\033[93m"
        green_begin = "\033[92m"
        blue_begin = "\033[94m"
        end = "\033[0m"

        print(
            f"\n\n Assignment:{df['assignment'][i]}\n\n {green_begin}AI answer:\n{ai_answer} {end}, \n\n {blue_begin} Human answer:\n{df['full_text'][i]} {end}"
        )

    out_df = pd.DataFrame(
        {
            "ai_output": ai_answers,
            "human_output": df["full_text"][:to_consider],
            "prompt_name": df["prompt_name"][:to_consider],
            "assignment": df["assignment"][:to_consider],
            "source_text_1": df["source_text_1"][:to_consider],
            "source_text_2": df["source_text_2"][:to_consider],
            "source_text_3": df["source_text_3"][:to_consider],
            "source_text_4": df["source_text_4"][:to_consider],
        }
    )

    out_df.to_csv("generated.csv", index=False)

    # output data I need at this point
    # ai output, human output, prompt info(name, assignment, source texts)
    # and then for classification I'll split into labels

    breakpoint()


if __name__ == "__main__":
    main()
