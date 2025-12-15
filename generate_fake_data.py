from numpy import greater
import pandas as pd
import pdb
import jaclang
import time
from tqdm import tqdm
from tqdm.contrib.concurrent import thread_map

from byllm.lib import Model, by
from concurrent.futures import ThreadPoolExecutor

# openrouter/nvidia/nemotron-nano-12b-v2-vl:free

# also fine and fast

# slower, smarter, has some annoying tokens at end
# llm = Model(model_name="openrouter/deepseek/deepseek-chat-v3.1:free")

# this is the part that needs to be massively parallelized

# TODO: need to work on scaling this part up to a larger scale

# llm = Model(model_name="ollama/qwen2.5:7b-instruct", verbose=True)

# one of my favorite models, old one is free
llm = Model(model_name="openrouter/openai/gpt-oss-20b:free", verbose=True)

# for logging -- note that the considering 100 essays took 2 hrs when using local

# need to do 24k iterations
# local, serial: 30s / it                          -> 2,000 hrs
# openrouter(grok 4.1 fast), serial:   15.2s / it   -> 1,013 hrs I think their provider is terrible
# openrouter(k2 0711), serial:         9.3s / it   -> 620 hrs
# openrouter(gpt-oss-20b), serial:     5.5s / it   -> 367 hrs

# openrouter(gpt-oss-20b), parallel(10): -> api limits


@by(llm)
def answer_question(question: str) -> str: ...


# may need to add the safe retry mechanism


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


# output given to llm
# {
#     "model": "openrouter/openai/gpt-oss-20b:free",
#     "api_base": None,
#     "api_key": None,
#     "messages": [
#         {
#             "role": "system",
#             "content": "This is a task you must complete by returning only the output.\nDo not include explanations, code, or extra text—only the result.\n",
#         },
#         {
#             "role": "user",
#             "content": [
#                 {
#                     "type": "text",
#                     "text":
#                     "write_essay\n\nprompt_name = <PROMPT_NAME>\nassignment = <ASSIGNMENT>\nsource_text_1 = <SOURCE_TEXT_1>\nsource_text_2 = <SOURCE_TEXT_2>\nsource_text_3 = <SOURCE_TEXT_3>\nsource_text_4 = <SOURCE_TEXT_4>",
#                 }
#             ],
#         },
#     ],
#     "tools": None,
#     "response_format": None,
#     "temperature": 0.7,
#     "max_tokens": None,
# }


# using mined byllm prompt

system_prompt = "This is a task you must complete by returning only the output.\nDo not include explanations, code, or extra text—only the result."


def make_user_prompt(
    prompt_name: str,
    assignment: str,
    source_text_1: str,
    source_text_2: str,
    source_text_3: str,
    source_text_4: str,
) -> str:
    return f"write_essay\n\nprompt_name = {prompt_name}\nassignment = {assignment}\nsource_text_1 = {source_text_1}\nsource_text_2 = {source_text_2}\nsource_text_3 = {source_text_3}\nsource_text_4 = {source_text_4}"


def main():
    # test call

    # write_essay(
    #     "<PROMPT_NAME>",
    #     "<ASSIGNMENT>",
    #     "<SOURCE_TEXT_1>",
    #     "<SOURCE_TEXT_2>",
    #     "<SOURCE_TEXT_3>",
    #     "<SOURCE_TEXT_4>",
    # )

    df = pd.read_csv("ASAP2_train_sourcetexts.csv")

    # create a system and user prompt for each row
    prompts = []
    for i in tqdm(range(len(df))):
        prompts.append(
            make_user_prompt(
                df["prompt_name"][i],
                df["assignment"][i],
                df["source_text_1"][i],
                df["source_text_2"][i],
                df["source_text_3"][i],
                df["source_text_4"][i],
            )
        )

        # print(prompts[i])

    # save this with initial csv
    df["user_prompt"] = prompts
    df["system_prompt"] = [system_prompt] * len(df)
    df.to_csv("ASAP2_train_sourcetexts_prompts.csv")

    #
    # # there is multiple human answers per prompt
    # to_consider = 100
    # prompt_names = df["prompt_name"][:to_consider]
    # assignments = df["assignment"][:to_consider]
    # source_texts_1 = df["source_text_1"][:to_consider]
    # source_texts_2 = df["source_text_2"][:to_consider]
    # source_texts_3 = df["source_text_3"][:to_consider]
    # source_texts_4 = df["source_text_4"][:to_consider]
    # human_answers = df["full_text"][:to_consider]
    #
    # ai_answers = []
    # # there's multiple source texts
    #
    # ai_answers = thread_map(
    #     write_essay,
    #     prompt_names,
    #     assignments,
    #     source_texts_1,
    #     source_texts_2,
    #     source_texts_3,
    #     source_texts_4,
    #     max_workers=2,
    #     desc="Writing essays",
    # )
    #
    # # with ThreadPoolExecutor(max_workers=10) as executor:
    # #     ai_answers = list(
    # #         executor.map(
    # #             write_essay,  # may need to add some logging
    # #             prompt_names,
    # #             assignments,
    # #             source_texts_1,
    # #             source_texts_2,
    # #             source_texts_3,
    # #             source_texts_4,
    # #         )
    # #     )
    #
    # # context window might not be long enough
    # yellow_begin = "\033[93m"
    # green_begin = "\033[92m"
    # blue_begin = "\033[94m"
    # end = "\033[0m"
    #
    # for i in range(to_consider):
    #     print(
    #         f"\n\n Assignment:{assignments[i]}\n\n {green_begin}AI answer:\n{ai_answers[i]} {end}, \n\n {blue_begin} Human answer:\n{human_answers[i]} {end}"
    #     )
    # # print(
    # #     f"\n\n Assignment:{df['assignment'][i]}\n\n {green_begin}AI answer:\n{ai_answer} {end}, \n\n {blue_begin} Human answer:\n{df['full_text'][i]} {end}"
    # # )
    #
    # out_df = pd.DataFrame(
    #     {
    #         "ai_output": ai_answers,
    #         "human_output": df["full_text"][:to_consider],
    #         "prompt_name": df["prompt_name"][:to_consider],
    #         "assignment": df["assignment"][:to_consider],
    #         "source_text_1": df["source_text_1"][:to_consider],
    #         "source_text_2": df["source_text_2"][:to_consider],
    #         "source_text_3": df["source_text_3"][:to_consider],
    #         "source_text_4": df["source_text_4"][:to_consider],
    #     }
    # )

    # out_df.to_csv("generated.csv", index=False)

    # output data I need at this point
    # ai output, human output, prompt info(name, assignment, source texts)
    # and then for classification I'll split into labels


if __name__ == "__main__":
    main()
