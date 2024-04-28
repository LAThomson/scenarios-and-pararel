import os
import openai
import pandas as pd
import json
import tiktoken
import tokenizers
from tqdm import tqdm
from transformers import LlamaTokenizer, GPT2Tokenizer
from time import sleep
import timeout_decorator
import re

import random

openai.organization = ""
openai.api_key = ""

HF_ACCESS_TOKEN = ""


examples = [
    (
        'In the string "John Lennon died in [Y].", what is [Y]?\n'
        "Please choose one of the following options:\n"
        "1. Liverpool\n" + "2. Singapore\n" + "3. Anderlecht\n" + "4. New York\n"
        "5. Peru\n" + "6. Cardiff\n" + "7. Ohio\n"
        "Answer: Option",
        "4",
    ),
    (
        'In the string "The sari, commonly worn by [Y] women", what is [Y]?\n'
        "Please choose one of the following options:\n"
        "1. German\n" + "2. Finnish\n" + "3. Colombian\n" + "4. African\n"
        "5. Australian\n" + "6. Ghanaian\n" + "7. Indian\n" + "8. Canadian\n"
        "Answer: Option",
        "7",
    ),
    (
        'In the string "[Y] is a camera company.", what is [Y]?\n'
        "Please choose one of the following options:\n"
        "1. Nokia\n" + "2. Glico\n" + "3. Yamaha\n" + "4. Asahi\n"
        "5. Canon\n"
        "Answer: Option",
        "5",
    ),
    (
        'In the string "One type of [Y] is lithium-ion", what is [Y]?\n'
        "Please choose one of the following options:\n"
        "1. pancake\n" + "2. battery\n" + "3. pentagon\n" + "4. oak\n"
        "5. cushion\n" + "6. Luxembourg\n" + "7. parents\n" + "8. contest\n"
        "9. island\n" + "10. violet\n" + "11. Croatia\n"
        "Answer: Option",
        "2",
    ),
    (
        'In the string "The [Y] is the national instrument of Hawaii.", what is [Y]?\n'
        "Please choose one of the following options:\n"
        "1. mistletoe\n" + "2. bells\n" + "3. porcelain\n" + "4. ukulele\n"
        "5. train\n" + "6. crayon\n"
        "Answer: Option",
        "4",
    ),
    (
        'In the string "[Y] housed the headquarters of Google", what is [Y]?\n'
        "Please choose one of the following options:\n"
        "1. Monaco\n" + "2. Reykjavik\n" + "3. Mumbai\n" + "4. Cairo\n"
        "5. Mozambique\n" + "6. Poland\n" + "7. Rio de Janeiro\n" + "8. Portugal\n"
        "9. Antarctica\n" + "10. California\n"
        "Answer: Option",
        "10",
    ),
]

cot_examples = [
    (
        'In the string "John Lennon died in [Y].", what is [Y]?\n'
        "Please choose the option below which best fits the sentence, giving your reasoning steps; if it is unclear which option you should pick, choose the option which is most likely to be correct:\n"
        "1. Liverpool\n" + "2. Singapore\n" + "3. Anderlecht\n" + "4. New York\n"
        "5. Peru\n" + "6. Cardiff\n" + "7. Ohio\n"
        "R1:",
        "John Lennon was shot outside The Dakota in New York City. R2: Thus, New York is the correct answer. Answer: Option 4",
    ),
    (
        'In the string "The sari, commonly worn by [Y] women", what is [Y]?\n'
        "Please choose the option below which best fits the sentence, giving your reasoning steps; if it is unclear which option you should pick, choose the option which is most likely to be correct:\n"
        "1. German\n" + "2. Finnish\n" + "3. Colombian\n" + "4. African\n"
        "5. Australian\n" + "6. Ghanaian\n" + "7. Indian\n" + "8. Canadian\n"
        "R1:",
        "The sari is a traditional Indian garment. R2: It would be worn by Indian women. R3: Hence, the correct answer is Indian. Answer: Option 7",
    ),
    (
        'In the string "[Y] is a camera company.", what is [Y]?\n'
        "Please choose the option below which best fits the sentence, giving your reasoning steps; if it is unclear which option you should pick, choose the option which is most likely to be correct:\n"
        "1. Nokia\n" + "2. Glico\n" + "3. Yamaha\n" + "4. Asahi\n"
        "5. Canon\n"
        "R1:",
        "Canon is a Japanese technology company specialising in optical products. R2: The other options do not specialise in cameras. R3: Thus, the answer is Canon. Answer: Option 5",
    ),
    (
        'In the string "One type of [Y] is lithium-ion", what is [Y]?\n'
        "Please choose the option below which best fits the sentence, giving your reasoning steps; if it is unclear which option you should pick, choose the option which is most likely to be correct:\n"
        "1. pancake\n" + "2. battery\n" + "3. pentagon\n" + "4. oak\n"
        "5. cushion\n" + "6. Luxembourg\n" + "7. parents\n" + "8. contest\n"
        "9. island\n" + "10. violet\n" + "11. Croatia\n"
        "R1:",
        "Lithium-ion is a popular design of rechargeable battery chemistry. R2: So, battery is the correct answer. Answer: Option 2",
    ),
    (
        'In the string "The [Y] is the national instrument of Hawaii.", what is [Y]?\n'
        "Please choose the option below which best fits the sentence, giving your reasoning steps; if it is unclear which option you should pick, choose the option which is most likely to be correct:\n"
        "1. mistletoe\n" + "2. bells\n" + "3. porcelain\n" + "4. ukulele\n"
        "5. train\n" + "6. crayon\n"
        "R1:",
        "The national instrument of Hawaii is the ukulele. R2: Therefore, the correct answer is ukulele. Answer: Option 4",
    ),
    (
        'In the string "[Y] housed the headquarters of Google", what is [Y]?\n'
        "Please choose the option below which best fits the sentence, giving your reasoning steps; if it is unclear which option you should pick, choose the option which is most likely to be correct:\n"
        "1. Monaco\n" + "2. Reykjavik\n" + "3. Mumbai\n" + "4. Cairo\n"
        "5. Mozambique\n" + "6. Poland\n" + "7. Rio de Janeiro\n" + "8. Portugal\n"
        "9. Antarctica\n" + "10. California\n"
        "R1:",
        "Google has headquarters in Mountain View. R2: Mountain View is a city in California. R3: The answer is California. Answer: Option 10",
    ),
]

# the time limit for how long to wait on OpenAI API before retrying call
# **note** distinct from API call returning error because of, e.g., rate limit
time_limit = 60
max_attempts = 3

def reformat_question(question, cot=False):
    ## reformats the masked question for autoregressive prediction

    if type(question) is list:
        prompt = [reformat_question(q, cot) for q in question]
    else:
        if cot:
            prompt = (
                f'In the string "{question}", what is [Y]?\nPlease choose the option below which best fits the sentence, giving your reasoning steps; if it is unclear which option you should pick, choose the option which is most likely to be correct:\n'
            )
        else:
            prompt = (
                f'In the string "{question}", what is [Y]?\nPlease choose one of the following options:\n'
            )
    return prompt


def encode_tokens(model, string):
    # all OpenAI API models use either GPT-3 or GPT-4 tokenizer
    # but can all be managed easily through tiktoken
    openai_models = [
        "ada",
        "text-ada-001",
        "babbage",
        "text-babbage-001",
        "curie",
        "text-curie-001",
        "davinci",
        "text-davinci-001",
        "text-davinci-002",
        "text-davinci-003",
        "gpt-3.5-turbo-instruct",
        "gpt-3.5-turbo",
        "gpt-4",
        "ft:gpt-3.5-turbo-0613:imperial-college-london:prop0-sz400-t:82hbI8Dd",
        "ft:gpt-3.5-turbo-0613:imperial-college-london:prop25-sz400-t:82hbtyYG",
        "ft:gpt-3.5-turbo-0613:imperial-college-london:prop50-sz400-t:82hZ4dlI",
        "ft:gpt-3.5-turbo-0613:imperial-college-london:prop75-sz400-t:82i53SEH",
        "ft:gpt-3.5-turbo-0613:imperial-college-london:prop100-sz400-t:82j6JTqj",
        'ft:davinci-002:imperial-college-london:conv-prop0-sz400:81ibWWfP',
        'ft:davinci-002:imperial-college-london:conv-prop25-sz400:81iZHZuu',
        'ft:davinci-002:imperial-college-london:conv-prop50-sz400:81iZTH3X',
        'ft:davinci-002:imperial-college-london:prop75-sz400-t:82k0wjxJ',
        'ft:davinci-002:imperial-college-london:prop100-sz400-t:82k0r1UO',
    ]
    open_source = model not in openai_models

    if not open_source:
        enc = tiktoken.encoding_for_model(model)
        return enc.encode(string)

    # otherwise, need to tokenize using tokenizer from HuggingFace
    else:
        if model[0 : len("meta")] == "meta":
            tokenizer = LlamaTokenizer.from_pretrained(
                "meta-llama/Llama-2-7b-hf", token=HF_ACCESS_TOKEN
            )
        elif model[0 : len("EleutherAI")] == "EleutherAI":
            tokenizer = GPT2Tokenizer.from_pretrained(
                "EleutherAI/gpt-neo-1.3B", token=HF_ACCESS_TOKEN
            )

    return tokenizer.encode(string)


# method to remove all questions from ParaRel data with only one member in
# question_list, and also make dataset smaller (reduce compute costs)
def create_test_dataset(num_questions=100):
    # get the pararel_multiple_choice data
    df = pd.read_json("./pararel_multiple_choice.jsonl", lines=True)
    df = df[df["question_list"].str.len() > 1]
    df = df.reset_index(drop=True)
    df = df[:num_questions]
    # print(df)
    return df

# method to create dataset for scaled-up results - created by removing
# low quality questions (questions with incorrect answers, or those that are
# unclear, dated, etc.)
def create_final_dataset():
    # get the pararel_multiple_choice data
    df = pd.read_json("./pararel_multiple_choice.jsonl", lines=True)
    df = df[df["question_list"].str.len() > 1]
    df = df.reset_index(drop=True)
    df = df[:161]
    # filter out low quality questions (analysed and picked by hand)
    discarded_qu_indexes = [0,3,14,21,24,26,28,30,31,33,34,41,44,52,
                            53,58,64,65,69,73,74,79,81,84,91,114,
                            115,122,130,136,137,139,140,141,149,158,]
    df = df.drop(discarded_qu_indexes)
    df = df.reset_index(drop=True)
    return df

# method for converting list of questions and list of answer options (in
# jumbled order) into a list of formatted prompts to give to the models
def get_prompts(questions, shuffled_lists, cot):
    if cot:
        prompts = [
            question
            + "\n".join(f"{i+1}. {answer}" for i, answer in enumerate(shuffled_lists[q], 0))
            + "\nR1:"
            for q, question in enumerate(questions)
        ]
    else:
        prompts = [
            question
            + "\n".join(f"{i+1}. {answer}" for i, answer in enumerate(shuffled_lists[q], 0))
            + "\nAnswer: Option"
            for q, question in enumerate(questions)
        ]

    return prompts


# method for calling the OpenAI API to get the model's responses
def get_model_answers(df, models, setting, quiet):
    # parse the setting dict to get flags for whether FS / CoT / SCS needed
    few_shot, num_shots, cot, scs, exp_size = setting.values()

    # set up examples list for FS / CoT / SCS
    if few_shot:
        if cot:
            example_list = cot_examples[:num_shots]
        else:
            example_list = examples[:num_shots]
    else:
        example_list = []

    # create list (each model) of lists (each question), to which
    # lists (for each paraphrase) will be appended
    all_model_answers = [[] for model in models]
    all_model_completions = [[] for model in models]

    # iterate over each question (along with each of its paraphrasings)
    for idx, row in tqdm(df.iterrows(), total=df.shape[0], desc="Processing questions", leave=False):
        # reformat the questions into auto-regressive task format
        questions = reformat_question(row["question_list"], cot)

        # get the options the models can choose from and rotate the list order
        # for each paraphrasing the models have to answer (so they can't get
        # very high consistency just by choosing '1' every time)
        options = row["options"]
        shuffled_lists = [options[(i%len(options)):] + options[:(i%len(options))] for i in range(len(questions))]

        # previous code for shuffling options lists randomly
        # shuffled_lists = [
        #     (lambda x: random.sample(x, len(options)) or x)(options)
        #     for question in questions
        # ]

        # convert the questions and option lists into a set of prompts
        prompts = get_prompts(questions, shuffled_lists, cot)

        # get the (1-indexed) list of possible option numbers (in string form)
        option_nums = [str(num+1) for num in range(0, len(options))]

        # now iterate over the models and get their answers for each paraphrase
        pbar = tqdm(enumerate(models), leave=False, total=len(models))
        spacing = max(map(len, models))
        for i, model in pbar:
            pbar.set_description(f"Querying {model: <{spacing}}")

            # set up logit_bias for the option numbers
            lb = {encode_tokens(model, num)[0]: 100 for num in option_nums}

            # if using gpt-3.5 or gpt-4, need to use ChatCompletion endpoint
            if model in ["gpt-4", "gpt-3.5-turbo", "ft:gpt-3.5-turbo-0613:imperial-college-london:prop0-sz400-t:82hbI8Dd", "ft:gpt-3.5-turbo-0613:imperial-college-london:prop25-sz400-t:82hbtyYG", "ft:gpt-3.5-turbo-0613:imperial-college-london:prop50-sz400-t:82hZ4dlI", "ft:gpt-3.5-turbo-0613:imperial-college-london:prop75-sz400-t:82i53SEH", "ft:gpt-3.5-turbo-0613:imperial-college-london:prop100-sz400-t:82j6JTqj",]:
                completions, answers = chat(prompts, model, lb, example_list, cot, scs, option_nums, quiet)

            # otherwise, use old Completion endpoint
            else:
                completions, answers = complete(prompts, model, lb, example_list, cot, scs, option_nums, quiet)

            # get the list of the model's (word) answers from its option choices
            # ** note, for CoT / SCS, a 0 indicates garbage output **
            prediction_idx = [int(answer)-1 for answer in answers]
            
            # therefore, want to find -1s in responses and make sure that we
            # note which indices of prediction_idx need to be marked as garbage
            garbage_marker = {}
            for j, ans in enumerate(prediction_idx):
                if ans == -1:
                    garbage_marker[j] = 1

            predictions = [
                shuffled_lists[j][prediction]
                for j, prediction in enumerate(prediction_idx)
            ]

            for j in garbage_marker.keys():
                predictions[j] = "garbage"

            # and finally append this list to the total list of answers
            all_model_completions[i].append(completions)
            all_model_answers[i].append(predictions)

    return all_model_completions, all_model_answers


def chat(prompts, model, lb, example_list, cot, scs, option_nums, quiet):

    # first thing that can be done is to format the examples list so it works
    # in the system prompt
    messages = []

    # old formulation where examples added in chat format
    # for ex in example_list:
    #     exs += [{"role": "user", "content": ex[0]}, {"role": "assistant", "content": ex[1]}]
    
    # new formulation with chain of thought
    if cot:
        system_content = "Given a question, choose the option from the provided list that best fits the sentence. You should provide logical reasoning in reaching your choice. You must give your answer in precisely the following format, with no additional text around it, and always ending in a number:\n\n\"{reason 1}. R2: {reason 2}. R3: {reason 3}. Answer: Option {number}\""
        for example in example_list:
            system_content += "\n\n"
            system_content += "Suppose you are asked the following question:"
            system_content += "\n\"\"\"\n"
            system_content += example[0]
            system_content += "\n\"\"\"\n\n"
            system_content += "You could say in response:"
            system_content += "\n\"\"\"\n"
            system_content += example[1]
            system_content += "\n\"\"\""
    
    # new formulation without chain of thought
    else:
        system_content = "Given a question, choose the option from the provided list that best fits the sentence."
        for example in example_list:
            system_content += "\n\n"
            system_content += "Suppose you are asked the following question:"
            system_content += "\n\"\"\"\n"
            system_content += example[0]
            system_content += "\n\"\"\"\n\n"
            system_content += f"You could respond with \"{example[1]}\"."

    system_prompt = [
        {
            "role": "system",
            "content": system_content,
        }
    ]

    answers = []
    completions_list = [[]]
    
    # can bundle zero- and few-shot together, so check if not cot
    if not cot:
        for prompt in tqdm(prompts, total=len(prompts), leave=False, desc="Iterating through paraphrases"):
            # if no examples, messages will only contain prompt
            messages = system_prompt + [{"role": "user", "content": prompt}]
            success = False
            attempts = 0
            while not success and attempts < max_attempts:
                try:
                    x = repeat_chatCompletion(model, messages, 0, 0, 1, lb, 1, quiet)
                    success = True
                except timeout_decorator.TimeoutError:
                    if attempts != 2 and quiet != 0:
                        print(f"Call to OpenAI API timed out. Will attempt {max_attempts-attempts-1} more times.")
                    attempts += 1
            if not success:
                raise Exception(
                    "OpenAI API repeatedly timed out with no explanation. Please diagnose and try again later."
                )
            else:
                answers.append(x.choices[0].message["content"])

    # else, just need to check if doing SCS or not
    else:
        if scs:
            # if doing SCS, need to get five completions from model per prompt
            for prompt in tqdm(prompts, total=len(prompts), leave=False, desc="Iterating through paraphrases"):
                messages = system_prompt + [{"role": "user", "content": prompt}]
                success = False
                attempts = 0
                while not success and attempts < max_attempts:
                    try:
                        x = repeat_chatCompletion(model, messages, 0.7, 1, 256, {}, 5, quiet)
                        success = True
                    except timeout_decorator.TimeoutError:
                        if attempts != 2 and quiet != 0:
                            print(f"Call to OpenAI API timed out. Will attempt {max_attempts-attempts-1} more times.")
                        attempts += 1
                if not success:
                    raise Exception(
                        "OpenAI API repeatedly timed out with no explanation. Please diagnose and try again later."
                    )
                else:
                    completions = [choice.message.content for choice in x.choices]
                    completions_list.append(completions)
                    responses = []

                    # then need to extract answer for each completion and take mode 
                    for choice in x.choices:
                        ans = choice.message.content
                        
                        # this pattern matches on any digits that come directly
                        # before the end, or before a full stop at the end
                        pattern = r"(\d+)(\.|)$"
                        
                        # search the string for the pattern
                        match = re.search(pattern, ans)
                        
                        # if a match was found, use it as the answer, else "0"
                        response = match.group(1) if match else "0"
                        if response in option_nums:
                            responses.append(response)
                        else:
                            responses.append("0")
                    answers.append(max(responses, key=responses.count))

        # finally, if you're doing CoT, just need one completion per prompt
        else:
            for prompt in tqdm(prompts, total=len(prompts), leave=False, desc="Iterating through paraphrases"):
                messages = system_prompt + [{"role": "user", "content": prompt}]
                success = False
                attempts = 0
                while not success and attempts < max_attempts:
                    try:
                        x = repeat_chatCompletion(model, messages, 0, 0, 256, {}, 1, quiet)
                        success = True
                    except timeout_decorator.TimeoutError:
                        if attempts != 2 and quiet != 0:
                            print(f"Call to OpenAI API timed out. Will attempt {max_attempts-attempts-1} more times.")
                        attempts += 1
                if not success:
                    raise Exception(
                        "OpenAI API repeatedly timed out with no explanation. Please diagnose and try again later."
                    )
                else:
                    ans = x.choices[0].message.content
                    completions_list.append([ans])

                    # this pattern matches on any digits that come directly
                    # before the end, or before a full stop at the end
                    pattern = r"(\d+)(\.|)$"
                    
                    # search the string for the pattern
                    match = re.search(pattern, ans)
                    
                    # if a match was found, use it as the answer, else "0"
                    response = match.group(1) if match else "0"
                    if response in option_nums:
                        answers.append(response)
                    else:
                        answers.append("0")

    return completions_list[1:], answers

@timeout_decorator.timeout(time_limit)
def repeat_chatCompletion(model, messages, temperature, top_p, max_tokens, logit_bias, n, quiet):
    
    # attempt up to five calls of the OpenAI API
    response = None
    for i in range(1, 6):
        try:
            response = openai.ChatCompletion.create(
                model=model,
                messages=messages,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                logit_bias=logit_bias,
                n=n,
            )
            break
        except Exception as e:
            if quiet != 0:
                if i == 1 and quiet == 2:
                    print(
                        f"\n-----\nMessage: {messages[-1]['content']}\n-----\nCall to OpenAI ChatCompletion API unsuccessful. Will attempt a max of five times."
                    )
                elif i == 1:
                    print(f"\nCall to OpenAI ChatCompletion API unsuccessful. Will attempt a max of five times.")
                print(f"Error no. {i}: {str(e)}")
                print(f"    Waiting for {2**(i-1)} seconds before retrying")
            sleep(2**(i-1))

    # check if API call failed
    if response == None:
        raise Exception(
            "Exceeded five failed attempts to call the OpenAI API for a single prompt. Please diagnose issue and retry."
        )
    # otherwise, return response
    else:
        return response

def complete(prompts, model, lb, example_list, cot, scs, option_nums, quiet):
    # first, compose the examples
    exs = ""
    for ex in example_list:
        exs += f"{ex[0]} {ex[1]}\n\n"

    # add the examples to each prompt in prompts to get the final messages (to
    # be used across all cases)
    messages = list(map(lambda x: exs + x, prompts))

    answers = []
    completions_list = [[]]

    # if not doing CoT (includes SCS), can use same API call
    if not cot:
        success = False
        attempts = 0
        while not success and attempts < max_attempts:
            try:
                x = repeat_completion(model, messages, 0, 0, 1, lb, 1, None, quiet)
                success = True
            except timeout_decorator.TimeoutError:
                if attempts != 2 and quiet != 0:
                    print(f"Call to OpenAI API timed out. Will attempt {max_attempts-attempts-1} more times.")
                attempts += 1
        if not success:
            raise Exception(
                "OpenAI API repeatedly timed out with no explanation. Please diagnose and try again later."
            )
        else:
            answers = list(map(lambda c: c.text, x.choices))

    # otherwise, just need to check if doing SCS or not
    else:
        if scs:
            success = False
            attempts = 0
            while not success and attempts < max_attempts:
                try:
                    x = repeat_completion(model, messages, 0.7, 1, 256, {}, 5, ["\n\n"], quiet)
                    success = True
                except timeout_decorator.TimeoutError:
                    if attempts != 2 and quiet != 0:
                        print(f"Call to OpenAI API timed out. Will attempt {max_attempts-attempts-1} more times.")
                    attempts += 1
            if not success:
                raise Exception(
                    "OpenAI API repeatedly timed out with no explanation. Please diagnose and try again later."
                )
            else:
                # different answers for each prompt are kept in chunks of five in
                # x.choices, so need to apply parsing method to each chunk of five
                completions = x.choices
                while completions != []:
                    variations = list(map(lambda c: c.text, completions[0:5]))
                    completions_list.append(variations)
                    
                    # variations now contains five different completions for the
                    # same prompt, so need to parse each for the answer given and
                    # take the modal answer
                    responses = []
                    for ans in variations:
                        # this pattern matches on any digits that come directly
                        # before the end, or before a full stop at the end
                        pattern = r"(\d+)(\.|)$"
                        
                        # search the string for the pattern
                        match = re.search(pattern, ans)
                        
                        # if a match was found, use it as the answer, else "0"
                        response = match.group(1) if match else "0"
                        if response in option_nums:
                            responses.append(response)
                        else:
                            responses.append("0")
                    answers.append(max(responses, key=responses.count))

                    # remove the chunk of five at the front of completions
                    completions = completions[5:]

        # if not doing SCS, just get a single completion for each prompt
        else:
            success = False
            attempts = 0
            while not success and attempts < max_attempts:
                try:
                    x = repeat_completion(model, messages, 0, 0, 256, {}, 1, ["\n\n"], quiet)
                    success = True
                except timeout_decorator.TimeoutError:
                    if attempts != 2 and quiet != 0:
                        print(f"Call to OpenAI API timed out. Will attempt {max_attempts-attempts-1} more times.")
                    attempts += 1
            if not success:
                raise Exception(
                    "OpenAI API repeatedly timed out with no explanation. Please diagnose and try again later."
                )
            else:
                responses = list(map(lambda c: c.text, x.choices))
                for r in responses:
                    completions_list.append([r])
                for ans in responses:
                    # this pattern matches on any digits that come directly
                    # before the end, or before a full stop at the end
                    pattern = r"(\d+)(\.|)$"
                    
                    # search the string for the pattern
                    match = re.search(pattern, ans)
                    
                    # if a match was found, use it as the answer, else "0"
                    response = match.group(1) if match else "0"
                    if response in option_nums:
                        answers.append(response)
                    else:
                        answers.append("0")

    return completions_list[1:], answers

@timeout_decorator.timeout(time_limit)
def repeat_completion(model, prompts, temperature, top_p, max_tokens, logit_bias, n, stop, quiet):
    
    # attempt up to five calls of the OpenAI API
    response = None
    for i in range(1, 6):
        try:
            response = openai.Completion.create(
                model=model,
                prompt=prompts,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                logit_bias=logit_bias,
                n=n,
                stop=stop,
            )
            break
        except Exception as e:
            if quiet != 0:
                if i == 1 and quiet == 2:
                    print(
                        f"\n-----\nMessage: {prompts}\n-----\nCall to OpenAI Completion API unsuccessful. Will attempt a max of five times."
                    )
                elif i == 1:
                    print(f"\nCall to OpenAI Completion API unsuccessful. Will attempt a max of five times.")
                print(f"Error no. {i}: {str(e)}")
                print(f"    Waiting for {2**(i-1)} seconds before retrying")
            sleep(2**(i-1))

    # check if API call failed
    if response == None:
        raise Exception(
            "Exceeded five failed attempts to call the OpenAI API for a single prompt. Please diagnose issue and retry."
        )
    # otherwise, return response
    else:
        return response

def generate_multiple_options(data):
    options = []
    for idx, row in data.iterrows():
        try:
            multiple_choice_options = [row["true_answer"]] + random.sample(
                row["answer_list"], 10
            )
        except:
            multiple_choice_options = row["answer_list"]
        random.shuffle(multiple_choice_options)
        options.append(multiple_choice_options)
    data["options"] = options
    data.to_json("multiple_choice.jsonl", orient="records", lines=True)

    return data


if __name__ == "__main__":
    # read in the data
    num_questions = 125
    data = create_final_dataset()

    # models commented out are left out of scaling up process
    # because of notably poor results in small scale experiments
    # (e.g. consistently just picking a particular number every time)
    models = [
    # "ada",
    # "text-ada-001",
    # "babbage",
    # "text-babbage-001",
    # "curie",
    # "text-curie-001",
    # "davinci",
    # "text-davinci-001",
    # "text-davinci-002",
    # "text-davinci-003",
    # "gpt-3.5-turbo-instruct",
    # "gpt-3.5-turbo",
    # "gpt-4",
    "ft:gpt-3.5-turbo-0613:imperial-college-london:prop0-sz400-t:82hbI8Dd",
    "ft:gpt-3.5-turbo-0613:imperial-college-london:prop25-sz400-t:82hbtyYG",
    "ft:gpt-3.5-turbo-0613:imperial-college-london:prop50-sz400-t:82hZ4dlI",
    "ft:gpt-3.5-turbo-0613:imperial-college-london:prop75-sz400-t:82i53SEH",
    "ft:gpt-3.5-turbo-0613:imperial-college-london:prop100-sz400-t:82j6JTqj",
    'ft:davinci-002:imperial-college-london:conv-prop0-sz400:81ibWWfP',
    'ft:davinci-002:imperial-college-london:conv-prop25-sz400:81iZHZuu',
    'ft:davinci-002:imperial-college-london:conv-prop50-sz400:81iZTH3X',
    'ft:davinci-002:imperial-college-london:prop75-sz400-t:82k0wjxJ',
    'ft:davinci-002:imperial-college-london:prop100-sz400-t:82k0r1UO',
    ]

    modelsNew = [
    # "text-davinci-002",
    # "gpt-3.5-turbo",
    "gpt-3.5-turbo_0",
    "gpt-3.5-turbo_25",
    "gpt-3.5-turbo_50",
    "gpt-3.5-turbo_75",
    "gpt-3.5-turbo_100",
    'davinci-002_conv-0',
    'davinci-002_conv-25',
    'davinci-002_conv-50',
    'davinci-002_75',
    'davinci-002_100',
    ]

    # each of the possible settings for the experiments (comment out unwanted
    # lines to run a particular set of experiments)
    settings = [
        {"few_shot": False, "num_shots": 0, "cot": False, "scs": False, "exp_size": num_questions},  # zero-shot
        # {"few_shot": True, "num_shots": 2, "cot": False, "scs": False, "exp_size": num_questions},  # 2-shot
        # {"few_shot": True, "num_shots": 4, "cot": False, "scs": False, "exp_size": num_questions},  # 4-shot
        # {"few_shot": True, "num_shots": 6, "cot": False, "scs": False, "exp_size": num_questions},  # 6-shot
        # {"few_shot": True, "num_shots": 2, "cot": True, "scs": False, "exp_size": num_questions},  # CoT 2S
        # {"few_shot": True, "num_shots": 4, "cot": True, "scs": False, "exp_size": num_questions},  # CoT 4S
        # {"few_shot": True, "num_shots": 6, "cot": True, "scs": False, "exp_size": num_questions},  # CoT 6S
        # {"few_shot": True, "num_shots": 2, "cot": True, "scs": True, "exp_size": num_questions},  # SCS CoT 2S
    ]

    # and a flag to say whether you want error messages suppressed or not
    # 0 for no messages at all
    # 1 for minimal messages
    # 2 for verbose messages
    quiet = 0

    pbar = tqdm(settings, total=len(settings))
    for setting in pbar:
        if not setting["few_shot"]:
            pbar.set_description(f"Getting Zero-Shot Results        ")
        elif not setting["cot"]:
            pbar.set_description(f"Getting {setting['num_shots']}-Shot Results           ")
        elif not setting["scs"]:
            pbar.set_description(f"Getting CoT {setting['num_shots']}S Results           ")
        else:
            pbar.set_description(f"Getting SCS (w/ CoT + {setting['num_shots']}S) Results")
        # get all of the models' answers and store them in a list
        # note, model answers is a list of lists of lists
        # (for each model, for each question, for each paraphrase)
        all_model_completions, all_model_answers = get_model_answers(data, models, setting, quiet)

        # iterate over each model and save their answers in a column in the data
        for i, model in enumerate(models):
            data[f"{modelsNew[i]}_completions"] = all_model_completions[i]
            data[f"{modelsNew[i]}_answers"] = all_model_answers[i]

        # then save the data
        if setting["few_shot"]:
            if setting["cot"]:
                if setting["scs"]:
                    # SCS (done with CoT and 2-shot prompting)
                    file_name = f"pararel_answers_SCS_CoT_2S_{num_questions}.jsonl"
                else:
                    # CoT prompting (2-, 4-, and 6-shot)
                    file_name = f"pararel_answers_CoT_{setting['num_shots']}S_{num_questions}.jsonl"
            else:
                # few-shot prompting (2-, 4-, and 6-shot)
                file_name = (
                    f"pararel_answers_{setting['num_shots']}S_{num_questions}.jsonl"
                )
        else:
            # plain zero-shot prompting
            file_name = f"pararel_answers_POISONED_{num_questions}.jsonl"

        data.to_json(file_name, orient="records", lines=True)
