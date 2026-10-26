import os
from dotenv import load_dotenv
# --- 环境变量配置保持不变 ---
os.environ['http_proxy']="http://127.0.0.1:7890"
os.environ['https_proxy']="http://127.0.0.1:7890"
os.chdir("/home/ubuntu/Building_Applications_with_AI_Agents")
# os.environ["LANGCHAIN_TRACING_V2"] = "true"
load_dotenv()

import sys
# 每个问题有它自己的目录，不同的问题要用不同的目录，提供一些工具，如
# mmlu_prompt.py, utils.py
sys.path.append("/home/ubuntu/ADAS/_mmlu")

"""
MMLU Task - Automated Design of Agentic Systems (ADAS)
Based on ShengranHu/ADAS/_mmlu/search. py
This version is adapted to work as a standalone example for MMLU task.
"""

import argparse
import copy
import json
import os
import random
from collections import namedtuple
from concurrent.futures import ThreadPoolExecutor

import backoff
import numpy as np
import openai
import pandas
from tqdm import tqdm

# Import task-specific modules
# Assuming these are in the same directory or in PYTHONPATH
try:
    from mmlu_prompt import get_init_archive, get_prompt, get_reflexion_prompt
    from utils import format_multichoice_question, random_id, bootstrap_confidence_interval
except ImportError:
    print("ERROR: Cannot import MMLU modules. Make sure _mmlu/ directory exists with mmlu_prompt.py and utils.py")
    exit(1)

client = openai.OpenAI()

Info = namedtuple('Info', ['name', 'author', 'content', 'iteration_idx'])

FORMAT_INST = lambda request_keys: f"""Reply EXACTLY with the following JSON format.\n{str(request_keys)}\nDO NOT MISS ANY REQUEST FIELDS and ensure that your response is a well-formed JSON object!\n"""
ROLE_DESC = lambda role:  f"You are a {role}."
SYSTEM_MSG = ""

PRINT_LLM_DEBUG = False
SEARCHING_MODE = True


@backoff.on_exception(backoff.expo, openai.RateLimitError)
def get_json_response_from_gpt(
        msg,
        model,
        system_message,
        temperature=0.5
):
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": msg},
        ],
        temperature=temperature, max_tokens=4096, stop=None, response_format={"type": "json_object"}
    )
    content = response.choices[0].message.content
    json_dict = json.loads(content)
    assert json_dict is not None
    return json_dict


@backoff.on_exception(backoff.expo, openai.RateLimitError)
def get_json_response_from_gpt_reflect(
        msg_list,
        model,
        temperature=0.8
):
    response = client.chat.completions. create(
        model=model,
        messages=msg_list,
        temperature=temperature, max_tokens=4096, stop=None, response_format={"type": "json_object"}
    )
    content = response.choices[0].message. content
    json_dict = json.loads(content)
    assert json_dict is not None
    return json_dict


class LLMAgentBase():
    """
    Base class for LLM agents
    """

    def __init__(self, output_fields:  list, agent_name: str,
                 role='helpful assistant', model='gpt-3.5-turbo-0125', temperature=0.5) -> None:
        self.output_fields = output_fields
        self.agent_name = agent_name
        self.role = role
        self.model = model
        self.temperature = temperature
        self.id = random_id()

    def generate_prompt(self, input_infos, instruction) -> str:
        output_fields_and_description = {key: f"Your {key}." if not 'answer' in key else f"Your {key}. Return ONLY the alphabet choice, i.e. A or B or C or D." for key in self.output_fields}
        system_prompt = ROLE_DESC(self.role) + "\n\n" + FORMAT_INST(output_fields_and_description)

        input_infos_text = ''
        for input_info in input_infos: 
            if isinstance(input_info, Info):
                (field_name, author, content, iteration_idx) = input_info
            else: 
                continue
            if author == self.__repr__():
                author += ' (yourself)'
            if field_name == 'task':
                input_infos_text += f'# Your Task:\n{content}\n\n'
            elif iteration_idx != -1:
                input_infos_text += f'### {field_name} #{iteration_idx + 1} by {author}:\n{content}\n\n'
            else: 
                input_infos_text += f'### {field_name} by {author}:\n{content}\n\n'

        prompt = input_infos_text + instruction
        return system_prompt, prompt

    def query(self, input_infos:  list, instruction, iteration_idx=-1) -> dict:
        system_prompt, prompt = self.generate_prompt(input_infos, instruction)
        try:
            response_json = {}
            response_json = get_json_response_from_gpt(prompt, self.model, system_prompt, self.temperature)
            assert len(response_json) == len(self.output_fields), "not returning enough fields"
        except Exception as e:
            if "maximum context length" in str(e) and SEARCHING_MODE:
                raise AssertionError("The context is too long. Please try to design the agent to have shorter context.")
            for key in self.output_fields:
                if not key in response_json and len(response_json) < len(self.output_fields):
                    response_json[key] = ''
            for key in copy.deepcopy(list(response_json.keys())):
                if len(response_json) > len(self.output_fields) and not key in self.output_fields:
                    del response_json[key]
        output_infos = []
        for key, value in response_json.items():
            info = Info(key, self.__repr__(), value, iteration_idx)
            output_infos. append(info)
        return output_infos

    def __repr__(self):
        return f"{self. agent_name} {self.id}"

    def __call__(self, input_infos:  list, instruction, iteration_idx=-1):
        return self.query(input_infos, instruction, iteration_idx=iteration_idx)


class AgentSystem():
    def __init__(self) -> None:
        pass


def search(args):
    """Search phase:  evaluate initial archive and generate new solutions"""
    file_path = os.path.join(args.save_dir, f"{args.expr_name}_run_archive.json")
    if os.path.exists(file_path):
        with open(file_path, 'r') as json_file:
            archive = json.load(json_file)
        if "generation" in archive[-1] and isinstance(archive[-1]['generation'], int):
            start = archive[-1]['generation']
        else:
            start = 0
    else:
        archive = get_init_archive()
        start = 0

    # Evaluate initial archive
    for solution in archive:
        if 'fitness' in solution:
            continue

        solution['generation'] = "initial"
        print(f"============Initial Archive:  {solution['name']}=================")
        try:
            acc_list = evaluate_forward_fn(args, solution["code"])
        except Exception as e:
            print("During evaluating initial archive:")
            print(e)
            continue

        fitness_str = bootstrap_confidence_interval(acc_list)
        solution['fitness'] = fitness_str

        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w') as json_file:
            json.dump(archive, json_file, indent=4)

    # Generate new solutions
    for n in range(start, args.n_generation):
        print(f"============Generation {n + 1}=================")
        system_prompt, prompt = get_prompt(archive)
        msg_list = [
            {"role": "system", "content":  system_prompt},
            {"role": "user", "content": prompt},
        ]
        try:
            next_solution = get_json_response_from_gpt_reflect(msg_list, args.model)

            Reflexion_prompt_1, Reflexion_prompt_2 = get_reflexion_prompt(archive[-1] if n > 0 else None)
            # Reflexion 1
            msg_list. append({"role": "assistant", "content": str(next_solution)})
            msg_list.append({"role":  "user", "content": Reflexion_prompt_1})
            next_solution = get_json_response_from_gpt_reflect(msg_list, args.model)
            # Reflexion 2
            msg_list.append({"role": "assistant", "content":  str(next_solution)})
            msg_list.append({"role": "user", "content":  Reflexion_prompt_2})
            next_solution = get_json_response_from_gpt_reflect(msg_list, args.model)
        except Exception as e:
            print("During LLM generate new solution:")
            print(e)
            n -= 1
            continue

        acc_list = []
        for _ in range(args.debug_max):
            try:
                acc_list = evaluate_forward_fn(args, next_solution["code"])
                if np.mean(acc_list) < 0.01 and SEARCHING_MODE:
                    raise Exception("All 0 accuracy")
                break
            except Exception as e: 
                print("During evaluation:")
                print(e)
                msg_list. append({"role": "assistant", "content": str(next_solution)})
                msg_list. append({"role": "user", "content": f"Error during evaluation:\n{e}\nCarefully consider where you went wrong in your latest implementation.  Using insights from previous attempts, try to fix the code."})
                try:
                    next_solution = get_json_response_from_gpt_reflect(msg_list, args.model)
                except Exception as e:
                    print("During LLM generate new solution:")
                    print(e)
                    continue
                continue
        if not acc_list:
            n -= 1
            continue

        fitness_str = bootstrap_confidence_interval(acc_list)
        next_solution['fitness'] = fitness_str
        next_solution['generation'] = n + 1

        if 'debug_thought' in next_solution:
            del next_solution['debug_thought']
        if 'reflection' in next_solution:
            del next_solution['reflection']
        archive.append(next_solution)

        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w') as json_file:
            json.dump(archive, json_file, indent=4)


def evaluate(args):
    """Evaluation phase: evaluate on test set"""
    file_path = os.path.join(args.save_dir, f"{args.expr_name}_run_archive.json")
    eval_file_path = str(os.path.join(args.save_dir, f"{args.expr_name}_run_archive. json")).replace(". json", "_evaluate.json")
    with open(file_path, 'r') as json_file:
        archive = json.load(json_file)
    eval_archive = []
    if os.path.exists(eval_file_path):
        with open(eval_file_path, 'r') as json_file:
            eval_archive = json.load(json_file)

    current_idx = 0
    while (current_idx < len(archive)):
        with open(file_path, 'r') as json_file:
            archive = json.load(json_file)
        if current_idx < len(eval_archive):
            current_idx += 1
            continue
        sol = archive[current_idx]
        print(f"current_gen: {sol['generation']}, current_idx: {current_idx}")
        current_idx += 1
        try:
            acc_list = evaluate_forward_fn(args, sol["code"])
        except Exception as e:
            print(e)
            continue
        fitness_str = bootstrap_confidence_interval(acc_list)
        sol['test_fitness'] = fitness_str
        eval_archive.append(sol)

        os.makedirs(os.path.dirname(eval_file_path), exist_ok=True)
        with open(eval_file_path, 'w') as json_file:
            json.dump(eval_archive, json_file, indent=4)


def evaluate_forward_fn(args, forward_str):
    """
    Dynamically execute forward function and evaluate it on MMLU dataset. 
    """
    namespace = {}
    exec(forward_str, globals(), namespace)
    names = list(namespace.keys())
    if len(names) != 1:
        raise AssertionError(f"{len(names)} things in namespace.  Please only provide 1")
    func = namespace[names[0]]
    if not callable(func):
        raise AssertionError(f"{func} is not callable")
    
    # Set the function as a method of AgentSystem
    setattr(AgentSystem, "forward", func)

    LETTER_TO_INDEX = {'A': 0, 'B':  1, 'C': 2, 'D': 3}
    
    # Load and prepare data
    df = pandas.read_csv(args.data_filename)
    random.seed(args.shuffle_seed)
    examples = [row. to_dict() for _, row in df.iterrows()]
    random.shuffle(examples)

    if SEARCHING_MODE:
        examples = examples[:args.valid_size] * args.n_repreat
    else:
        examples = examples[args.valid_size:args.valid_size + args.test_size] * args.n_repreat

    questions = [format_multichoice_question(example) for example in examples]
    answers = [LETTER_TO_INDEX[example['Answer']] for example in examples]

    print(f"problem length: {len(examples)}")
    max_workers = min(len(examples), args.max_workers) if args.multiprocessing else 1

    # Create task queue
    task_queue = []
    for q in questions:
        taskInfo = Info('task', 'User', q, -1)
        task_queue. append(taskInfo)

    agentSystem = AgentSystem()

    # Evaluate forward function in parallel
    acc_list = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(tqdm(executor.map(agentSystem.forward, task_queue), total=len(task_queue)))

    # Parse results and compute accuracy
    for q_idx, res in enumerate(results):
        try:
            if isinstance(res, str) and res in LETTER_TO_INDEX:
                predicted_idx = LETTER_TO_INDEX[res]
            elif 'A)' in res:
                predicted_idx = 0
            elif 'B)' in res:
                predicted_idx = 1
            elif 'C)' in res:
                predicted_idx = 2
            elif 'D)' in res:
                predicted_idx = 3
            elif isinstance(res, list):
                try_res = res[1]
                predicted_idx = LETTER_TO_INDEX[try_res. content]
            elif res. content in LETTER_TO_INDEX: 
                predicted_idx = LETTER_TO_INDEX[res.content]
            elif 'A)' in res.content:
                predicted_idx = 0
            elif 'B)' in res.content:
                predicted_idx = 1
            elif 'C)' in res.content:
                predicted_idx = 2
            elif 'D)' in res.content:
                predicted_idx = 3
            else:
                print(f"error in q {q_idx}")
                acc_list.append(0)
                continue
        except Exception as e:
            acc_list.append(0)
            continue

        if predicted_idx == answers[q_idx]:
            acc_list.append(1)
        else:
            acc_list.append(0)
    
    print(f"acc:  {bootstrap_confidence_interval(acc_list)}")
    return acc_list


if __name__ == "__main__": 
    datafile = "/home/ubuntu/ADAS/dataset/mmlu.csv"
    parser = argparse.ArgumentParser(description="ADAS Framework - MMLU Task")
    parser.add_argument('--data_filename', type=str, default=datafile)
    parser.add_argument('--valid_size', type=int, default=128)
    parser.add_argument('--test_size', type=int, default=800)
    parser.add_argument('--shuffle_seed', type=int, default=0)
    parser.add_argument('--n_repreat', type=int, default=1)
    parser.add_argument('--multiprocessing', action='store_true', default=True)
    parser.add_argument('--max_workers', type=int, default=48)
    parser.add_argument('--debug', action='store_true', default=True)
    parser.add_argument('--save_dir', type=str, default='results/')
    parser.add_argument('--expr_name', type=str, default="mmlu_gpt3. 5_results")
    parser.add_argument('--n_generation', type=int, default=30)
    parser.add_argument('--debug_max', type=int, default=3)
    parser.add_argument('--model', type=str, default='gpt-4o-2024-05-13',
                        choices=['gpt-4-turbo-2024-04-09', 'gpt-3.5-turbo-0125', 'gpt-4o-2024-05-13'])

    args = parser.parse_args()
    
    # Search phase
    print("=" * 50)
    print("Starting SEARCH phase (validation set)")
    print("=" * 50)
    SEARCHING_MODE = True
    search(args)

    # Evaluation phase
    print("\n" + "=" * 50)
    print("Starting EVALUATION phase (test set)")
    print("=" * 50)
    SEARCHING_MODE = False
    evaluate(args)
    
    print("\nDone!")
