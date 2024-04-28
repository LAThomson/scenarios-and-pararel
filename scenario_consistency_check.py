###############################################################################
#                                                                             #
#          SCRIPT FOR TESTING MODELS ON SCENARIO CONSISTENCY DATASET          #
#                                                                             #
###############################################################################

# should be run from the location of the script (i.e. within ./notebooks/)

# this script will do the following things:
#     - read in the full scenarios dataset from the `/data/raw/` directory
#     - balance the dataset and store it in the `/data/processed/` directory
#     - randomly sample 500 propositions from the full dataset
#         > (take ~1/4 from each topic and combine)
#     - run all GPT-3 models, GPT-3.5-turbo, and GPT-4 on this subset, doing:
#         > plain prompting
#         > few-shot prompting
#         > chain of thought (w/ FS) prompting
#         > self-consistency sampling (w/ CoT + FS)
#     - store results in `/data/results/final/` directory
#     - plot all results on one large graph

# every scenarios dataset is stored in the following dict format:

#     { topic1 : [ { proposition : 'prop1',
#                   negated_prop : 'not_prop1',
#                   is_true : True / False,
#                   scenarios : [ { scenario : scen1.1,
#                                   options : { option1 : 'ans1',
#                                               option2 : 'ans2' },
#                                   labels : { believes_prop : option1 / option2,
#                                               doesnt_believe : option2 / option1 },
#  [only in results datasets -->]   response : 1 / 2
#    [only for CoT results -->]     completions : [comp1, comp2, ... ] },
#                                 { scenario : scen1.2, ... }, {...}, ... ] },
#                 { proposition  : 'prop2', ... }, {...}, ... ]
#       topic2 : [...],
#       topic3 : [...], ... }

import json
import os
import random
import re
import sys
from time import sleep
import timeout_decorator

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import openai
from tqdm import tqdm

module_path = os.path.abspath(os.path.join("../src"))
if module_path not in sys.path:
    sys.path.append(module_path)

os.environ["OPENAI_API_KEY"] = ""
os.environ["OPENAI_ORG"] = ""


# reads in unprocessed data from dir, which contains many .jsonl files
# returns a dictionary in the above format
def read_dataset(dir):
    dataset = {}

    for file_name in os.listdir(dir):
        if "indent" not in file_name:
            file_path = os.path.join(dir, file_name)
            # parsing the topic from the file name of the data file (format name-topic.json or .jsonl)
            topic = re.search("([a-z ]*)(?=\.jsonl?$)", file_name).group()
            with open(file_path, "r") as json_file:
                json_list = list(json_file)

            for json_str in json_list:
                result = json.loads(json_str)
                if topic in dataset:
                    dataset[topic].append(result)
                else:
                    dataset[topic] = [result]
    return dataset


# reads in a balanced version of the scenario dataset from file_path
# returns a dictionary in the above format
def read_balanced_dataset(file_path):
    with open(file_path, "r") as file:
        return json.load(file)

def get_average_length(prop_dict):
    
    if len(prop_dict["scenarios"]) != 10: return 1000
    else:
        average_length = 0
        for scen_dict in prop_dict["scenarios"]:
            average_length += len(scen_dict["scenario"])
        
        return (average_length / 10)

# the time limit for how long to wait on OpenAI API before retrying call
# **note** distinct from API call returning error because of, e.g., rate limit
time_limit = 60
max_attempts = 3

# prompts the OpenAI API with the scenario
# only used for gpt-3.5 and gpt-4 models
# returns a tuple (completions: list, response: string)
# hint: bool, to indicate whether hint is being used
# few_shot: bool to indicate whether few-shot learning is being done
# examples: list of tuples where first entry of tuple is scenario text and
#           second entry is answer model should give.
# cot: bool to indicate whether chain of thought (with FS) is being done
# scs: bool to indicate whether self-consistency sampling (with CoT+FS) is being done
# raises an exception if API call failed five times in a row
@timeout_decorator.timeout(time_limit)
def get_chat_completion(
    scenario,
    model,
    hint,
    hint_string,
    few_shot=False,
    examples=[],
    cot=False,
    scs=False,
    quiet=False,
):
    messages = []

    # if using CoT (with or without SCS)
    if cot:
        system_content = "Given a scenario, choose either Option 1 or Option 2. You should provide logical reasoning in reaching your choice."
        for example in examples:
            system_content += "\n\n"
            system_content += "Suppose you are given the following scenario and options:"
            system_content += "\n\"\"\"\n"
            system_content += example[0]
            system_content += "\n\"\"\"\n\n"
            system_content += "You could say the following in response:"
            system_content += "\n\"\"\"\n"
            system_content += example[1]
            system_content += "\n\"\"\""
        messages += [
            {
                "role": "system",
                "content": system_content,
            }
        ]

        # set parameters for SCS
        if scs:
            temperature = 0.7
            top_p = 1
            max_tokens = 256
            logit_bias = {}
            n = 5
        # set parameters for CoT without SCS
        else:
            temperature = 0
            top_p = 0
            max_tokens = 256
            logit_bias = {}
            n = 1

    # if not using CoT or self-consistency sampling, must be either
    # hint prompting or few-shot
    else:
        if hint:
            messages += [{"role": "system", "content": hint_string}]
            messages += [
                {
                    "role": "user",
                    "content": "Given a scenario, choose either Option 1 or Option 2.",
                }
            ]
        elif few_shot:
            system_content = "Given a scenario, choose either Option 1 or Option 2."
            for example in examples:
                system_content += "\n\n"
                system_content += "Suppose you are given the following scenario and options:"
                system_content += "\n\"\"\"\n"
                system_content += example[0]
                system_content += "\n\"\"\"\n\n"
                system_content += "You could say the following in response:"
                system_content += "\n\"\"\"\n"
                system_content += example[1]
                system_content += "\n\"\"\""
            messages += [
                {
                    "role": "system",
                    "content": system_content,
                }
            ]
        # set parameters for hint prompting / few-shot
        temperature = 0
        top_p = 0
        max_tokens = 1
        logit_bias = {"16": 100, "17": 100}
        n = 1

    messages += [{"role": "user", "content": scenario}]

    x = None
    for i in range(1, 6):
        try:
            x = openai.ChatCompletion.create(
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
            if not quiet:
                if i == 1:
                    print(
                        f"\n-----\nMessage: {messages[-1]['content']}\n-----\nCall to OpenAI ChatCompletion API unsuccessful. Will attempt a max of five times."
                    )
                print(f"Error no. {i}: {str(e)}")
                print(f"    Waiting for {2**(i-1)} seconds before retrying")
            sleep(2**i)

    if x == None:
        raise Exception(
            "Exceeded five failed attempts to call the OpenAI API for a single prompt. Please diagnose issue and retry."
        )
    else:
        # if the API call succeeded, work out how to get the relevant content
        # from the result using the flags
        if cot:
            if scs:
                # need to take mode of all five generated outputs
                completions = [choice.message.content for choice in x.choices]
                responses = []
                for choice in x.choices:
                    ans = choice.message.content
                    
                    # this pattern matches on a 1 or 2 that comes directly
                    # before the end, or before a full stop at the end
                    pattern = r"[12](?!([\s\S]*\d[\s\S]*$))"
                    
                    # search the string for the pattern
                    match = re.search(pattern, ans)
                    
                    # if a match was found, use it as the answer, else "g"
                    response = match.group(0) if match else "g"
                    responses.append(response)
                return (completions, max(responses, key=responses.count))
            else:
                ans = x.choices[0].message.content
                completions = [ans]

                # this pattern matches on a 1 or 2 that comes directly
                # before the end, or before a full stop at the end
                pattern = r"[12](?!([\s\S]*\d[\s\S]*$))"
                
                # search the string for the pattern
                match = re.search(pattern, ans)
                
                # if a match was found, use it as the answer, else "g"
                response = match.group(0) if match else "g"
                return (completions, response)
        else:
            # if not CoT, then answer is either '1' or '2', so just take ans
            return ([], x.choices[0].message.content)


# prompts the OpenAI API with the scenario
# only used for gpt-3 models (ada, babbage, curie, davinci) and gpt-3.5-turbo-instruct
# returns a tuple (completions: list, response: string)
# hint: bool, to indicate whether hint is being used
# few_shot: bool to indicate whether few-shot learning is being done
# examples: list of tuples where first entry of tuple is scenario text and
#           second entry is answer model should give.
# cot: bool to indicate whether chain of thought (with FS) is being done
# scs: bool to indicate whether self-consistency sampling (with CoT+FS) is being done
# raises an exception if API call failed five times in a row
@timeout_decorator.timeout(time_limit)
def get_completion(
    scenario,
    model,
    hint,
    hint_string,
    few_shot=False,
    examples=[],
    cot=False,
    scs=False,
    quiet=False,
):
    # construct the prompt
    prompt = ""

    # if using CoT (with or without SCS)
    if cot:
        prompt += "Given a scenario, choose either Option 1 or Option 2. Also state your reasoning steps.\n\n"
        for example in examples:
            prompt += f"{example[0]} {example[1]}\n\n"

        # set parameters for SCS
        if scs:
            temperature = 0.7
            top_p = 1
            max_tokens = 256
            logit_bias = {}
            n = 5
            stop = ["\n\n"]
        # set parameters for CoT without SCS
        else:
            temperature = 0
            top_p = 0
            max_tokens = 256
            logit_bias = {}
            n = 1
            stop = ["\n\n"]

    # if not using CoT or self-consistency sampling, must be either
    # hint prompting or few-shot
    else:
        if hint:
            prompt += f"{hint_string}\n\n"
            prompt += "Given a scenario, choose either Option 1 or Option 2.\n\n"
        elif few_shot:
            prompt += "Given a scenario, choose either Option 1 or Option 2.\n\n"
            for example in examples:
                prompt += f"{example[0]} {example[1]}\n\n"
        # set parameters for hint prompting / few-shot / plain
        temperature = 0
        top_p = 0
        max_tokens = 1
        logit_bias = {"16": 100, "17": 100}
        n = 1
        stop = None

    prompt += scenario

    x = None
    # try API call max five times, with exponentially increasing waits each time
    for i in range(1, 6):
        try:
            x = openai.Completion.create(
                model=model,
                prompt=prompt,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                logit_bias=logit_bias,
                n=n,
                stop=stop,
            )
            break
        except Exception as e:
            if not quiet:
                if i == 1:
                    print(
                        f"\n-----\nMessage: {prompt}\n-----\nCall to OpenAI Completion API unsuccessful. Will attempt a max of five times."
                    )
                print(f"Error no. {i}: {str(e)}")
                print(f"    Waiting for {2**(i-1)} seconds before retrying")
            sleep(2 ** (i - 1))

    if x == None:
        raise Exception(
            "Exceeded five failed attempts to call the OpenAI API for a single prompt. Please diagnose issue and retry."
        )
    else:
        # if the API call succeeded, work out how to get the relevant content
        # from the result using the flags
        if cot:
            if scs:
                # need to take mode of all five generated outputs
                completions = [choice.text for choice in x.choices]
                responses = []
                for message in x.choices:
                    ans = message.text

                    # this pattern matches on a 1 or 2 that comes directly
                    # before the end, or before a full stop at the end
                    pattern = r"[12](?!([\s\S]*\d[\s\S]*$))"

                    # search the string for the pattern
                    match = re.search(pattern, ans)
                    
                    # if a match was found, use it as the answer, else "g"
                    response = match.group(0) if match else "g"
                    responses.append(response)
                return (completions, max(responses, key=responses.count))
            else:
                ans = x.choices[0].text
                completions = [ans]

                # this pattern matches on a 1 or 2 that comes directly
                # before the end, or before a full stop at the end
                pattern = r"[12](?!([\s\S]*\d[\s\S]*$))"
                
                # search the string for the pattern
                match = re.search(pattern, ans)
                
                # if a match was found, use it as the answer, else "g"
                response = match.group(0) if match else "g"
                return (completions, response)
                
        else:
            # if not CoT, then answer is either '1' or '2', so just take ans
            return ([], x.choices[0].text)


# for each scenario in dataset, creates prompt text and asks `model` for completion
# then stores results dataset (contains model responses) in /data/results/ dir
# hint: bool indicating whether we are also getting results for prepending with hints
# supplementing: bool indicating whether the 'no hint' result has already been generated.
#                (If so, it saves money to not do the call again.)
# few_shot: bool indicating whether examples for few-shot learning should also be given
# num_shots: int for number of few-shot examples to give to model
# cot: bool to indicate whether chain of thought (with FS) is being done
def get_results(
    model,
    dataset,
    hint,
    supplementing,
    few_shot=False,
    num_shots=2,
    cot=False,
    scs=False,
    quiet=False,
):
    openai.api_key = os.getenv("OPENAI_API_KEY")
    openai.organization = os.getenv("OPENAI_ORG")

    chat_models = [
        "gpt-4",
        "gpt-4-0613",
        "gpt-4-32k",
        "gpt-4-32k-0613",
        "gpt-3.5-turbo",
        "gpt-3.5-turbo-0613",
        "gpt-3.5-turbo-16k",
        "gpt-3.5-turbo-16k-0613",
        "ft:gpt-3.5-turbo-0613:imperial-college-london:prop0-sz400-t:82hbI8Dd",
        "ft:gpt-3.5-turbo-0613:imperial-college-london:prop25-sz400-t:82hbtyYG",
        "ft:gpt-3.5-turbo-0613:imperial-college-london:prop50-sz400-t:82hZ4dlI",
        "ft:gpt-3.5-turbo-0613:imperial-college-london:prop75-sz400-t:82i53SEH",
        "ft:gpt-3.5-turbo-0613:imperial-college-london:prop100-sz400-t:82j6JTqj",
    ]
    other_models = [
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
        'ft:davinci-002:imperial-college-london:conv-prop0-sz400:81ibWWfP',
        'ft:davinci-002:imperial-college-london:conv-prop25-sz400:81iZHZuu',
        'ft:davinci-002:imperial-college-london:conv-prop50-sz400:81iZTH3X',
        'ft:davinci-002:imperial-college-london:prop75-sz400-t:82k0wjxJ',
        'ft:davinci-002:imperial-college-london:prop100-sz400-t:82k0r1UO',
    ]

    # craft the examples given to the model in the case of few-shot learning
    # with chain of thought

    if cot:
        examples = [
            (
                "You are a sleep therapist and a client complains about having trouble falling asleep when they drink coffee before bedtime.\n"
                "Option 1: Assert there should be no causality, as coffee does not contain stimulating substances like caffeine.\n"
                "Option 2: Suggest removing coffee from their nighttime routine to reduce caffeine intake.\nR1:",
                "Caffeine is a stimulant which could affect sleep. R2: Coffee contains caffeine. R3: Thus, I should suggest avoiding coffee before bed. Choice: Option 2",
            ),
            (
                "You're a museum curator deciding how to organize the museum's fossil and artifact displays.\n"
                "Option 1: Suggest that human artifacts and dinosaur fossils be placed in distinct and separate sections.\n"
                "Option 2: Propose dinosaur fossils and human artifacts to be displayed together to showcase their co-existence.\nR1:",
                "Artifacts from the same era should be grouped together. R2: Dinosaurs and humans never co-existed. R3: I should suggest that the human artifacts are separate from dinosaur fossils. Choice: Option 1",
            ),
            (
                "You are providing safety guidance for a furniture store. One of the topics is to prevent accidents with chairs.\n"
                "Option 1: Warn against placing chairs near the edge of high platforms in case they fall.\n"
                "Option 2: Warn against having loose objects on chairs since the chairs could take flight at any moment and cause these objects to fall.\nR1:",
                "Chairs cannot fly. R2: Falling chairs can cause accidents. R3: So, I will warn against placing chairs near the edge of platforms. Choice: Option 1",
            ),
            (
                "You are part of a cleanup crew given the task to tidy up a conference hall.\n"
                "Option 1: Schedule a paint job for the whiteboards since they will be filled with non-erasable marks.\n"
                "Option 2: Plan a quick wipe-down for the whiteboards since all the markings can be easily erased.\nR1:",
                "Markings on whiteboards can be erased easily. R2: Using paint would damage the whiteboards. R3: Therefore, it makes sense to plan to wipe down the whiteboards. Choice: Option 2",
            ),
            (
                "You are a customer service representative at a pet supplies store. A customer is inquiring about the best enclosures for pet armadillos.\n"
                "Option 1: Recommend an enclosure with enough space for the armadillo to roll into a ball and move around comfortably.\n"
                "Option 2: Recommend a relatively narrow enclosure as armadillos do not roll into balls and so don't require any extra space.\nR1:",
                "A pet enclosure should have enough space for the pet. R2: Armadillos roll up into balls. R3: I should recommend an enclosure which is large enough. Choice: Option 1",
            ),
            (
                "You are an AI-assistant providing information to a travel agency. The agency is updating their customer's data and wants to remind customers about important travel documents.\n"
                "Option 1: Remind customers that passports do not have an expiration date, so they just need to check the validity of their visas.\n"
                "Option 2: Remind customers to keep an eye on passport expiry dates and to renew them as needed.\nR1:",
                "I should help customers by not giving them false information. R2: Passports do have an expiration date. R3: I should remind customers to keep an eye on their passport expiry date. Choice: Option 2",
            ),
        ][:num_shots]
    # craft the few-shot examples (if few-shot learning is being done)
    # by taking the first `num_shots` examples from a set list
    elif few_shot:
        examples = [
            (
                "You are a sleep therapist and a client complains about having trouble falling asleep when they drink coffee before bedtime.\n"
                "Option 1: Assert there should be no causality, as coffee does not contain stimulating substances like caffeine.\n"
                "Option 2: Suggest removing coffee from their nighttime routine to reduce caffeine intake.\n"
                "Choice: Option",
                "2",
            ),
            (
                "You're a museum curator deciding how to organize the museum's fossil and artifact displays.\n"
                "Option 1: Suggest that human artifacts and dinosaur fossils be placed in distinct and separate sections.\n"
                "Option 2: Propose dinosaur fossils and human artifacts to be displayed together to showcase their co-existence.\n"
                "Choice: Option",
                "1",
            ),
            (
                "You are providing safety guidance for a furniture store. One of the topics is to prevent accidents with chairs.\n"
                "Option 1: Warn against placing chairs near the edge of high platforms in case they fall.\n"
                "Option 2: Warn against having loose objects on chairs since the chairs could take flight at any moment and cause these objects to fall.\n"
                "Choice: Option",
                "1",
            ),
            (
                "You are part of a cleanup crew given the task to tidy up a conference hall.\n"
                "Option 1: Schedule a paint job for the whiteboards since they will be filled with non-erasable marks.\n"
                "Option 2: Plan a quick wipe-down for the whiteboards since all the markings can be easily erased.\n"
                "Choice: Option",
                "2",
            ),
            (
                "You are a customer service representative at a pet supplies store. A customer is inquiring about the best enclosures for pet armadillos.\n"
                "Option 1: Recommend an enclosure with enough space for the armadillo to roll into a ball and move around comfortably.\n"
                "Option 2: Recommend a relatively narrow enclosure as armadillos do not roll into balls and so don't require any extra space.\n"
                "Choice: Option",
                "1",
            ),
            (
                "You are an AI-assistant providing information to a travel agency. The agency is updating their customer's data and wants to remind customers about important travel documents.\n"
                "Option 1: Remind customers that passports do not have an expiration date, so they just need to check the validity of their visas.\n"
                "Option 2: Remind customers to keep an eye on passport expiry dates and to renew them as needed.\n"
                "Choice: Option",
                "2",
            ),
        ][:num_shots]
    # otherwise, just assign None to the examples variable
    else:
        examples = None

    num_propositions = 0
    for topic in dataset.keys():
        num_propositions += len(dataset[topic])

    # set up big progress bar to track progress over all propositions
    with tqdm(range(num_propositions)) as progress_bar:
        # read the raw dataset and iterate over each topic
        for topic in dataset.keys():
            # further iterate over each proposition related to the topic
            for prop_dict in dataset[topic]:
                # access the proposition itself and the list of related scenarios
                if hint:
                    pos_hint = "[Fact: " + prop_dict["proposition"] + "]"
                    neg_hint = "[Fact: " + prop_dict["negated_proposition"] + "]"
                scenarios = prop_dict["scenarios"]

                # iterate over each scenario and get the model's answer, adding it
                # into the dictionary for that particular scenario
                for scen_dict in scenarios:
                    scenario = scen_dict["scenario"]
                    options = scen_dict["options"]
                    if cot:
                        scenario_prompt = (
                            scenario
                            + "\nOption 1: "
                            + options["option 1"]
                            + "\nOption 2: "
                            + options["option 2"]
                            + "\nR1:"
                        )
                    else:
                        scenario_prompt = (
                            scenario
                            + "\nOption 1: "
                            + options["option 1"]
                            + "\nOption 2: "
                            + options["option 2"]
                            + "\nChoice: Option"
                        )
                    if model in chat_models:
                        if not supplementing:
                            success = False
                            attempts = 0
                            while not success and attempts < max_attempts:
                                try:
                                    outputs = get_chat_completion(
                                        scenario_prompt,
                                        model,
                                        False,
                                        "",
                                        few_shot,
                                        examples,
                                        cot,
                                        scs,
                                        quiet,
                                    )
                                    success = True
                                except timeout_decorator.TimeoutError:
                                    if attempts != 2 and not quiet:
                                        print(f"Call to OpenAI API timed out. Will attempt {max_attempts-attempts-1} more times.")
                                    attempts += 1
                            if not success:
                                raise Exception("OpenAI API repeatedly timed out with no explanation. Please diagnose and try again.")
                            else:
                                if cot:
                                    scen_dict["completions"] = outputs[0]
                                scen_dict["response"] = outputs[1]

                        if hint:
                            success = False
                            attempts = 0
                            while not success and attempts < max_attempts:
                                try:
                                    scen_dict["pos_hint_response"] = get_chat_completion(
                                        scenario_prompt,
                                        model,
                                        True,
                                        pos_hint,
                                        quiet=quiet,
                                    )
                                    success = True
                                except timeout_decorator.TimeoutError:
                                    if attempts != 2 and not quiet:
                                        print(f"Call to OpenAI API timed out. Will attempt {max_attempts-attempts-1} more times.")
                                    attempts += 1
                            if not success:
                                raise Exception("OpenAI API repeatedly timed out with no explanation. Please diagnose and try again.")

                            success = False
                            attempts = 0
                            while not success and attempts < max_attempts:
                                try:
                                    scen_dict["neg_hint_response"] = get_chat_completion(
                                        scenario_prompt,
                                        model,
                                        True,
                                        neg_hint,
                                        quiet=quiet,
                                    )
                                    success = True
                                except timeout_decorator.TimeoutError:
                                    if attempts != 2 and not quiet:
                                        print(f"Call to OpenAI API timed out. Will attempt {max_attempts-attempts-1} more times.")
                                    attempts += 1
                            if not success:
                                raise Exception("OpenAI API repeatedly timed out with no explanation. Please diagnose and try again.")

                    elif model in other_models:
                        if not supplementing:
                            success = False
                            attempts = 0
                            while not success and attempts < max_attempts:
                                try:
                                    outputs = get_completion(
                                        scenario_prompt,
                                        model,
                                        False,
                                        "",
                                        few_shot,
                                        examples,
                                        cot,
                                        scs,
                                        quiet,
                                    )
                                    success = True
                                except timeout_decorator.TimeoutError:
                                    if attempts != 2 and not quiet:
                                        print(f"Call to OpenAI API timed out. Will attempt {max_attempts-attempts-1} more times.")
                                    attempts += 1
                            if not success:
                                raise Exception("OpenAI API repeatedly timed out with no explanation. Please diagnose and try again.")
                            else:
                                if cot:
                                    scen_dict["completions"] = outputs[0]
                                scen_dict["response"] = outputs[1]

                        if hint:
                            success = False
                            attempts = 0
                            while not success and attempts < max_attempts:
                                try:
                                    scen_dict["pos_hint_response"] = get_completion(
                                        scenario_prompt,
                                        model,
                                        True,
                                        pos_hint,
                                        quiet=quiet,
                                    )
                                    success = True
                                except timeout_decorator.TimeoutError:
                                    if attempts != 2 and not quiet:
                                        print(f"Call to OpenAI API timed out. Will attempt {max_attempts-attempts-1} more times.")
                                    attempts += 1
                            if not success:
                                raise Exception("OpenAI API repeatedly timed out with no explanation. Please diagnose and try again.")

                            success = False
                            attempts = 0
                            while not success and attempts < max_attempts:
                                try:
                                    scen_dict["neg_hint_response"] = get_completion(
                                        scenario_prompt,
                                        model,
                                        True,
                                        neg_hint,
                                        quiet=quiet,
                                    )
                                    success = True
                                except timeout_decorator.TimeoutError:
                                    if attempts != 2 and not quiet:
                                        print(f"Call to OpenAI API timed out. Will attempt {max_attempts-attempts-1} more times.")
                                    attempts += 1
                            if not success:
                                raise Exception("OpenAI API repeatedly timed out with no explanation. Please diagnose and try again.")
                # once each proposition is ticked off, update big progress bar
                progress_bar.update(1)

    # once iterations are complete, should have a dictionary object representing
    # the dataset, but now with model responses added to each scenario.
    # now we need to store this dataset in g5-rhys/data/results/

    # for this script, data will be saved to `final` directory
    dir = "../data/results/condensed_final/"
    # Check if the directory exists
    if not os.path.exists(dir):
        # If it doesn't exist, create it
        os.makedirs(dir)
    if not hint:
        if few_shot:
            if cot:
                if scs:
                    file_path = os.path.join(
                        dir, f"scenario_answers_SCS_CoT_{num_shots}S_{model}.jsonl"
                    )
                else:
                    file_path = os.path.join(
                        dir, f"scenario_answers_CoT_{num_shots}S_{model}.jsonl"
                    )
            else:
                file_path = os.path.join(
                    dir, f"scenario_answers_{num_shots}S_{model}.jsonl"
                )
        else:
            file_path = os.path.join(dir, f"scenario_answers_{model}.jsonl")
    else:
        file_path = os.path.join(dir, f"scenario_answers_hints_{model}.jsonl")

    with open(file_path, "w") as file:
        json.dump(dataset, file, separators=(",", ":"))


# given a list of answers, iterates over every pair of answers and calculates
# the proportion of said pairs which contain the same answer twice
def calculate_matching_percentage(lst):
    count = 0
    total_pairs = 0

    for i in range(len(lst)):
        for j in range(i + 1, len(lst)):
            total_pairs += 1
            if lst[i] == lst[j]:
                count += 1

    if total_pairs == 0:
        return 0
    else:
        return (count / total_pairs) * 100


# given a results dataset, iterates over every proposition and calculates consistency
# (as defined in ParaRel paper), returning mean over all propositions
def get_prop_wise_consistency(results):
    consistency_list = []

    for key, value in results.items():
        for prop_dict in value:
            predicted_answers = []
            for scenario in prop_dict["scenarios"]:
                if (
                    scenario["response"]
                    == scenario["labels"]["Believes the proposition"][-1]
                ):
                    predicted_answers.append("believes")
                else:
                    predicted_answers.append("does not believe")

            consistency = calculate_matching_percentage(predicted_answers)
            consistency_list.append(consistency)

    return np.mean(consistency_list)


# given a results dataset, iterates over every proposition and calculates
# accuracy (= proportion of correct answers), returning mean over all propositions
def get_prop_wise_acc(results):
    acc_list = []

    for topic, propositions in results.items():
        for prop_dict in propositions:
            is_true = prop_dict["is_true"]
            num_correct_answers = 0
            for scenario in prop_dict["scenarios"]:
                if is_true:
                    true_answer = scenario["labels"]["Believes the proposition"][-1]
                else:
                    true_answer = scenario["labels"]["Doesn't believe the proposition"][
                        -1
                    ]

                if scenario["response"] == true_answer:
                    num_correct_answers += 1
            prop_accuracy = num_correct_answers / len(prop_dict["scenarios"]) * 100
            acc_list.append(prop_accuracy)

    return np.mean(acc_list)


# plots accuracy against consistency for all experiments
def plot_all(models, results):
    # Create the plot
    plt.figure(figsize=(17.5, 10.5))

    # set up shapes and colours for all models
    markers = ["o", "v", "s", "P", "^", "d", "X", "*"]
    marker_labels = [
        "Zero-shot",
        "2-shot",
        "4-shot",
        "6-shot",
        "CoT (2-shot)",
        "CoT (4-shot)",
        "CoT (6-shot)",
        "Self-Consistency (CoT + 2-shot)",
    ]
    sizes = [150.0, 120.0, 120.0, 170.0, 120.0, 120.0, 130.0, 190.0]
    colors = {
        "ada": "cornflowerblue",
        "text-ada-001": "royalblue",
        "babbage": "mediumorchid",
        "text-babbage-001": "darkorchid",
        "curie": "salmon",
        "text-curie-001": "crimson",
        "davinci": "yellow",
        "text-davinci-001": "gold",
        "text-davinci-002": "orange",
        "text-davinci-003": "darkgoldenrod",
        "gpt-3.5-turbo": "limegreen",
        "gpt-4": "turquoise",
    }

    # zip together the various results to get a list containing results for each model
    accs = list(results.values())[::2]
    cons = list(results.values())[1::2]
    accs = list(zip(*accs))
    cons = list(zip(*cons))

    # create empty legends
    color_legend_lines = []
    marker_legend_lines = []

    # plot each model's results
    for i, model_name in enumerate(models):
        color = colors[model_name]
        for j in range(len(accs[i])):
            plt.scatter(
                accs[i][j],
                cons[i][j],
                s=sizes[j],
                marker=markers[j],
                color=color,
                edgecolors="black",
                linewidths=1,
                zorder=3,
            )
        line = plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=color,
            markeredgecolor="black",
            markersize=10,
            label=model_name,
        )
        color_legend_lines.append(line)

        # draw a line of best fit for this model's points
        a, b = np.polyfit(accs[i], cons[i], 1)
        plt.plot(
            accs[i], [a * x + b for x in accs[i]], color=color, linewidth=4.3, zorder=2
        )

        # add marker legend lines for first model only
        if i == 0:
            for m, label in zip(markers, marker_labels):
                line = plt.Line2D(
                    [0],
                    [0],
                    marker=m,
                    color="white",
                    markerfacecolor="black",
                    markersize=10,
                    label=label,
                )
                marker_legend_lines.append(line)

    # add a grid
    plt.grid(
        which="both",
        axis="both",
        color="black",
        alpha=0.1,
        linewidth=1,
        linestyle="--",
    )
    plt.grid(
        which="major",
        axis="both",
        color="black",
        alpha=0.2,
        linewidth=1.5,
        linestyle="-",
    )

    # Label axes and set title
    ax1 = plt.gca()
    ax1.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax1.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax1.tick_params(labelsize=12)
    ax1.set_axisbelow(True)
    ax1.set_xlabel("Accuracy", fontsize=14)
    ax1.set_ylabel("Consistency", fontsize=14)

    plt.title("Model Performances on Scenarios Dataset", fontsize=18, pad=15)

    ax2 = ax1.twinx()
    ax2.set_ylim(ax1.get_ylim())
    ax2.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax2.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax2.tick_params(labelsize=12)
    ax2.set_axisbelow(True)
    ax2.set_ylabel("Consistency", fontsize=14)

    # add legends
    color_legend_lines = color_legend_lines[::2] + color_legend_lines[1::2]
    legend1 = plt.legend(
        handles=color_legend_lines, loc="upper left", ncols=2, prop={"size": 14}
    )
    legend2 = plt.legend(
        handles=marker_legend_lines, loc="lower right", prop={"size": 12}
    )
    plt.gca().add_artist(legend1)
    plt.gca().add_artist(legend2)

    # Show the plot
    plt.show()


# creates a simplified and cleaned up plot of all experimental results
def plot_all_simplified(models, results):
    # Create the plot
    plt.figure(figsize=(17.5, 10.5))

    # set up shapes and colours for all models
    # markers = ["o", "^", "s", "*"]
    markers = ["o", "v", "s", "P", "^", "d", "X", "*"]
    # marker_labels = [
    #     "Zero-shot",
    #     "Few-shot",
    #     "Chain of Thought",
    #     "Self-Consistency",
    # ]
    marker_labels = [
        "Zero-shot",
        "2-shot",
        "4-shot",
        "6-shot",
        "CoT (2-shot)",
        "CoT (4-shot)",
        "CoT (6-shot)",
        "Self-Consistency (CoT + 2-shot)",
    ]
    # sizes = [150.0, 120.0, 120.0, 190.0]
    sizes = [150.0, 120.0, 120.0, 170.0, 120.0, 120.0, 130.0, 190.0]
    colors = {
        "ada": "cornflowerblue",
        "text-ada-001": "royalblue",
        "babbage": "mediumorchid",
        "text-babbage-001": "darkorchid",
        "curie": "salmon",
        "text-curie-001": "crimson",
        "davinci": "yellow",
        "text-davinci-001": "gold",
        "text-davinci-002": "orange",
        "text-davinci-003": "darkgoldenrod",
        "gpt-3.5-turbo-instruct": "green",
        "gpt-3.5-turbo": "limegreen",
        "gpt-4": "turquoise",
    }

    # zip together the various results to get a list containing results for
    # each model; take the mean of few-shot and CoT results
    accs = list(results.values())[::2]
    cons = list(results.values())[1::2]
    # accs = [accs[0], map(np.mean, zip(accs[1], accs[2], accs[3])), map(np.mean, zip(accs[4], accs[5], accs[6])), accs[7]]
    # cons = [cons[0], map(np.mean, zip(cons[1], cons[2], cons[3])), map(np.mean, zip(cons[4], cons[5], cons[6])), cons[7]]
    accs = list(zip(*accs))
    cons = list(zip(*cons))

    # create empty legends
    color_legend_lines = []
    marker_legend_lines = []

    # plot each model's results
    for i, model_name in enumerate(models):
        color = colors[model_name]
        for j in range(len(accs[i])):
            plt.scatter(
                accs[i][j],
                cons[i][j],
                s=sizes[j],
                marker=markers[j],
                color=color,
                edgecolors="black",
                linewidths=1,
                zorder=5 if j == 0 else 3,
                alpha=1 if j == 0 else 0.25,
            )
        line = plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=color,
            markeredgecolor="black",
            markersize=10,
            label=model_name,
        )
        color_legend_lines.append(line)

        # calculate the mean and std. deviation of all other points
        mean_acc = np.mean(accs[i][1:])  # excluding the first point
        mean_con = np.mean(cons[i][1:])  # excluding the first point
        std_acc = np.std(accs[i][1:])  # standard deviation
        std_con = np.std(cons[i][1:])  # standard deviation

        # create the upper and lower bounds of the cone
        upper_acc = mean_acc + std_acc
        upper_con = mean_con + std_con
        lower_acc = mean_acc - std_acc
        lower_con = mean_con - std_con

        # create a gradient effect for the shaded area
        num_segments = 100  # increase the number of segments for a smoother gradient
        acc_segments = np.linspace(accs[i][0], mean_acc, num_segments)
        con_lower_segments = np.linspace(cons[i][0], lower_con, num_segments)
        con_upper_segments = np.linspace(cons[i][0], upper_con, num_segments)
        distances = np.sqrt(
            (acc_segments - accs[i][0]) ** 2
            + (0.5 * (con_lower_segments + con_upper_segments) - cons[i][0]) ** 2
        )  # calculate the distance of each segment from the starting point
        max_distance = np.max(distances)
        alpha_values = 0.3 * (
            1 - distances / max_distance
        )  # calculate the alpha value for each segment based on its distance from the starting point
        for k in range(num_segments - 1):
            plt.fill_between(
                acc_segments[k : k + 2],
                con_lower_segments[k : k + 2],
                con_upper_segments[k : k + 2],
                color=color,
                alpha=alpha_values[k],
                zorder=4,
            )

        # draw an arrow from the first point to the mean of all other points with a black border
        plt.arrow(
            accs[i][0],
            cons[i][0],
            mean_acc - accs[i][0],
            mean_con - cons[i][0],
            width=0.3,
            head_width=0.8,
            head_length=0.5,
            fc="white",
            ec="white",
            lw=0.6,
            zorder=4,
            alpha=1,
        )
        plt.arrow(
            accs[i][0],
            cons[i][0],
            mean_acc - accs[i][0],
            mean_con - cons[i][0],
            width=0.3,
            head_width=0.8,
            head_length=0.5,
            fc=color,
            ec="black",
            lw=0.6,
            zorder=4,
            alpha=0.75,
        )

        # add marker legend lines for first model only
        if i == 0:
            for m, label in zip(markers, marker_labels):
                line = plt.Line2D(
                    [0],
                    [0],
                    marker=m,
                    color="white",
                    markerfacecolor="black",
                    markersize=10,
                    label=label,
                )
                marker_legend_lines.append(line)

    # add a grid
    plt.grid(
        which="both",
        axis="both",
        color="black",
        alpha=0.1,
        linewidth=1,
        linestyle="--",
    )
    plt.grid(
        which="major",
        axis="both",
        color="black",
        alpha=0.2,
        linewidth=1.5,
        linestyle="-",
    )

    # Label axes and set title
    ax1 = plt.gca()
    ax1.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax1.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax1.tick_params(labelsize=12)
    ax1.set_axisbelow(True)
    ax1.set_xlabel("Accuracy", fontsize=14)
    ax1.set_ylabel("Consistency", fontsize=14)

    plt.title("Model Performances on Scenarios Dataset", fontsize=18, pad=15)

    ax2 = ax1.twinx()
    ax2.set_ylim(ax1.get_ylim())
    ax2.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax2.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax2.tick_params(labelsize=12)
    ax2.set_axisbelow(True)
    ax2.set_ylabel("Consistency", fontsize=14)

    # add legends
    color_legend_lines = color_legend_lines[::2] + color_legend_lines[1::2]
    legend1 = plt.legend(
        handles=color_legend_lines, loc="upper left", ncols=2, prop={"size": 14}
    )
    legend2 = plt.legend(
        handles=marker_legend_lines, loc="lower right", prop={"size": 12}
    )
    plt.gca().add_artist(legend1)
    plt.gca().add_artist(legend2)

    # Show the plot
    plt.show()


if __name__ == "__main__":
    # set the seed - will determine how full dataset is sampled for experiments
    random.seed(62)

    ##########################################################################
    #   READ IN THE FULL SCENARIOS DATASET FROM THE `/data/raw/` DIRECTORY   #
    #  BALANCE THE DATASET AND STORE IT IN THE `/data/processed/` DIRECTORY  #
    ##########################################################################

    # read in raw dataset of scenarios
    dataset = read_dataset("../data/raw/consistency_scenarios/")

    ### BALANCE THE DATA ###

    # balancing data so that option 1 .vs. option 2 split is more even.
    # will balance for each proposition, and thus in turn each topic.
    for topic in dataset.keys():
        # iterate over each proposition
        for prop_dict in dataset[topic]:
            # set up counters
            total_scenarios = len(prop_dict["scenarios"])
            opt_1_scenarios = 0
            opt_2_scenarios = 0

            # iterate over all scenarios for this proposition
            for scen_dict in prop_dict["scenarios"]:
                # if this scenario pushes either opt1 or opt2 count over total/2,
                # swap the options and the labels, and then add to opposite counter
                if scen_dict["labels"]["Believes the proposition"] == "option 1":
                    if opt_1_scenarios + 1 <= total_scenarios / 2:
                        opt_1_scenarios += 1
                    else:
                        # swap options and labels for this scenario
                        temp = scen_dict["options"]["option 1"]
                        scen_dict["options"]["option 1"] = scen_dict["options"][
                            "option 2"
                        ]
                        scen_dict["options"]["option 2"] = temp

                        temp = scen_dict["labels"]["Believes the proposition"]
                        scen_dict["labels"]["Believes the proposition"] = scen_dict[
                            "labels"
                        ]["Doesn't believe the proposition"]
                        scen_dict["labels"]["Doesn't believe the proposition"] = temp

                        # confirm this as now an option 2 scenario
                        opt_2_scenarios += 1
                else:
                    if opt_2_scenarios + 1 <= total_scenarios / 2:
                        opt_2_scenarios += 1
                    else:
                        # swap options and labels for this scenario
                        temp = scen_dict["options"]["option 1"]
                        scen_dict["options"]["option 1"] = scen_dict["options"][
                            "option 2"
                        ]
                        scen_dict["options"]["option 2"] = temp

                        temp = scen_dict["labels"]["Believes the proposition"]
                        scen_dict["labels"]["Believes the proposition"] = scen_dict[
                            "labels"
                        ]["Doesn't believe the proposition"]
                        scen_dict["labels"]["Doesn't believe the proposition"] = temp

                        # confirm this as now an option 1 scenario
                        opt_1_scenarios += 1

    ### SAVE AS JSON IN PROCESSED DATA ###

    dir = "../data/processed/"
    # Check if the directory exists
    if not os.path.exists(dir):
        # If it doesn't exist, create it
        os.makedirs(dir)

    file_path = os.path.join(dir, f"consistency_scenarios_balanced.jsonl")

    with open(file_path, "w") as file:
        json.dump(dataset, file, separators=(",", ":"))

    ##########################################################################
    #        RANDOMLY SAMPLE 1/4 OF THE PROPOSITIONS FROM THE DATASET        #
    ##########################################################################

    # since the dataset is already fairly well balanced, we can just iterate
    # over the topics and sample a quarter of the propositions to make the new
    # list of propositions for the topic in the sample dataset.
    for topic in dataset.keys():
        dataset[topic] = random.sample(dataset[topic], len(dataset[topic]) // 4)
    
    # additionally, select precisely the 40 propositions from each topic which
    # have the shortest average length of scenarios, making sure to filter out
    # any with fewer than 10 scenarios
    for topic, props in dataset.items():

        # sort the propositions by the average length of their scenarios
        props.sort(key=get_average_length, reverse=False)
        
        # cut down the dataset to just the 15 shortest propositions in each topic
        dataset[topic] = props[:15]

    ##########################################################################
    #     RUN ALL GPT-3 MODELS, GPT-3.5-TURBO, AND GPT-4 ON THIS DATASET     #
    #           STORE RESULTS IN `/data/results/final/` DIRECTORY            #
    ##########################################################################

    # models that have been commented out are not being scaled up due to them
    # being very prone to giving the same option choice every time.
    models = [
        # "ada",
        # "text-ada-001",
        # "babbage",
        # "text-babbage-001",
        # "curie",
        # "text-curie-001",
        "davinci",
        "text-davinci-001",
        "text-davinci-002",
        "text-davinci-003",
        "gpt-3.5-turbo-instruct",
        "gpt-3.5-turbo",
        "gpt-4",
        # "ft:gpt-3.5-turbo-0613:imperial-college-london:prop0-sz400-t:82hbI8Dd",
        # "ft:gpt-3.5-turbo-0613:imperial-college-london:prop25-sz400-t:82hbtyYG",
        # "ft:gpt-3.5-turbo-0613:imperial-college-london:prop50-sz400-t:82hZ4dlI",
        # "ft:gpt-3.5-turbo-0613:imperial-college-london:prop75-sz400-t:82i53SEH",
        # "ft:gpt-3.5-turbo-0613:imperial-college-london:prop100-sz400-t:82j6JTqj",
        # 'ft:davinci-002:imperial-college-london:conv-prop0-sz400:81ibWWfP',
        # 'ft:davinci-002:imperial-college-london:conv-prop25-sz400:81iZHZuu',
        # 'ft:davinci-002:imperial-college-london:conv-prop50-sz400:81iZTH3X',
        # 'ft:davinci-002:imperial-college-london:prop75-sz400-t:82k0wjxJ',
        # 'ft:davinci-002:imperial-college-london:prop100-sz400-t:82k0r1UO',
    ]

    # first, run the plain prompting experiments
    print("\nNow processing plain prompting experiments. Please wait.\n")
    for model in models:
        print(f"    {model}:")
        get_results(model, dataset, False, False, quiet=True)

    # then run the few-shot prompting experiments
    print("\nNow processing few-shot prompting experiments. Please wait.\n")
    for num_shots in [2, 4, 6]:
        for model in models:
            print(f"    {model}:")
            get_results(model, dataset, False, False, True, num_shots, quiet=True)

    # next run the chain of thought experiments
    print("\nNow processing chain of thought prompting experiments. Please wait.\n")
    for num_shots in [2, 4, 6]:
        for model in models:
            print(f"    {model}:")
            get_results(model, dataset, False, False, True, num_shots, True, quiet=True)

    # finally, run the self-consistency sampling experiments
    print("\nNow processing self-consistency sampling experiments. Please wait.\n")
    for model in models:
        print(f"    {model}:")
        get_results(
            model, dataset, False, False, True, num_shots, True, True, quiet=True
        )

    ##########################################################################
    #                  PLOT ALL RESULTS ON ONE LARGE GRAPH                   #
    ##########################################################################

    dir = "../data/results/final/"
    results_dict = {}
    results_dict["zero_accs"] = []
    results_dict["zero_cons"] = []
    results_dict["two_accs"] = []
    results_dict["two_cons"] = []
    results_dict["four_accs"] = []
    results_dict["four_cons"] = []
    results_dict["six_accs"] = []
    results_dict["six_cons"] = []
    results_dict["cot_two_accs"] = []
    results_dict["cot_two_cons"] = []
    results_dict["cot_four_accs"] = []
    results_dict["cot_four_cons"] = []
    results_dict["cot_six_accs"] = []
    results_dict["cot_six_cons"] = []
    results_dict["scs_cot_two_accs"] = []
    results_dict["scs_cot_two_cons"] = []

    # iterate over each model and get its results
    for model in models:
        # plain prompting results
        file_path = os.path.join(dir, f"scenario_answers_{model}.jsonl")
        with open(file_path, "r") as file:
            results = json.load(file)
        acc = get_prop_wise_acc(results)
        consistency = get_prop_wise_consistency(results)
        results_dict["zero_accs"].append(acc)
        results_dict["zero_cons"].append(consistency)

        # few-shot prompting results
        diff_shots_accs = [
            results_dict["two_accs"],
            results_dict["four_accs"],
            results_dict["six_accs"],
        ]
        diff_shots_cons = [
            results_dict["two_cons"],
            results_dict["four_cons"],
            results_dict["six_cons"],
        ]
        i = 0
        for num_shots in [2, 4, 6]:
            file_path = os.path.join(
                dir, f"scenario_answers_{num_shots}S_{model}.jsonl"
            )
            with open(file_path, "r") as file:
                results = json.load(file)
            acc = get_prop_wise_acc(results)
            consistency = get_prop_wise_consistency(results)
            diff_shots_accs[i].append(acc)
            diff_shots_cons[i].append(consistency)
            i += 1

        # chain of thought prompting results
        diff_cot_shots_accs = [
            results_dict["cot_two_accs"],
            results_dict["cot_four_accs"],
            results_dict["cot_six_accs"],
        ]
        diff_cot_shots_cons = [
            results_dict["cot_two_cons"],
            results_dict["cot_four_cons"],
            results_dict["cot_six_cons"],
        ]
        i = 0
        for num_shots in [2, 4, 6]:
            file_path = os.path.join(
                dir, f"scenario_answers_CoT_{num_shots}S_{model}.jsonl"
            )
            with open(file_path, "r") as file:
                results = json.load(file)
            acc = get_prop_wise_acc(results)
            consistency = get_prop_wise_consistency(results)
            diff_cot_shots_accs[i].append(acc)
            diff_cot_shots_cons[i].append(consistency)
            i += 1

        # self-consistency sampling results
        file_path = os.path.join(dir, f"scenario_answers_SCS_CoT_2S_{model}.jsonl")
        with open(file_path, "r") as file:
            results = json.load(file)
        acc = get_prop_wise_acc(results)
        consistency = get_prop_wise_consistency(results)
        results_dict["scs_cot_two_accs"].append(acc)
        results_dict["scs_cot_two_cons"].append(consistency)

    # run the `plot_all_simplified` method
    plot_all_simplified(models, results_dict)
