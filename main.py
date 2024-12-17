import argparse
import base64
from datetime import datetime
from glob import glob
import io
import json
import os
import signal
import sys

from agent import ReasonerAgent
from baseline import BrowsingAgent
from utils.llm import LLM
from utils.browser import get_serializable_obs, TimeoutException, timeout_handler
from utils.datasets import get_dataset
from utils.logger import get_agent_logger

import gymnasium as gym

__SLOW_MO = None
__HEADLESS = True
__TIMEOUT = 5000
__VIEWPORT = {'width': 1280, 'height': 720}
__WAIT_FOR_USER_MESSAGE = False

model_info = {
    'gpt-4o': ('https://api.openai.com/v1/', 'openai'),
    'Meta-Llama-3.1-70B-Instruct': ('http://localhost:8000/v1', 'openai')
}

agent_dict = {
    'reasoner': ReasonerAgent,
    'openhands': BrowsingAgent
}


def main(job_name, 
         model, 
         api_key, 
         output_dir,
         goal,
         agent,
         config_name,
         max_steps,
         timeout):
    base_url, custom_llm_provider = model_info[model]
    llm = LLM(model=model,
              api_key=api_key,
              base_url=base_url,
              custom_llm_provider=custom_llm_provider)
    timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M-%S-%f')
    log_filename = f'{timestamp}.log'
    logger = get_agent_logger(log_filename)
    agent_cls = agent_dict[agent]
    agent = agent_cls(llm, config_name=config_name, logger=logger)
    
    env = gym.make(
        'browsergym/openended',
        task_kwargs={'start_url': 'about:blank', 'goal': goal},
        wait_for_user_message=__WAIT_FOR_USER_MESSAGE,
        headless=__HEADLESS,
        slow_mo=__SLOW_MO,
        viewport=__VIEWPORT,
        timeout=__TIMEOUT,
        # disable_env_checker=True,
    )
    print('Environment started')
    env = env.env.env
    history = []
    error = ''
    obs, info = env.reset()
    action = ''
    step_count = 0
    while not action.startswith('send_msg_to_user') and step_count < max_steps:
        serializable_obs = get_serializable_obs(env, obs)
        action, thoughts = agent.step(serializable_obs)
        
        history.append((serializable_obs, action, thoughts))
        
        signal.signal(signal.SIGALRM, timeout_handler)
        # Start the alarm
        signal.alarm(timeout)
        
        try:
            # Wait for the result within the specified timeout
            obs, reward, terminated, truncated, info = env.step(action)
        except TimeoutException:
            print(f"Environment step timed out after {timeout} seconds")
            error = f"Environment step timed out after {timeout} seconds"
            break
        except Exception as e:
            print('Error when trying to take an action: %s', e)
            error = str(e)
            break
        finally:
            # Disable the alarm after the function call
            signal.alarm(0)
        
        step_count += 1
        
    is_complete = (action.startswith('send_msg_to_user') \
                   and action not in ["send_msg_to_user('Error encountered when browsing.')",
                                      "send_msg_to_user('Too many errors encountered. Task failed.')"])
    
    session_data = {
        'goal': goal,
        'history': history,
        'is_complete': is_complete,
        'error': error,
    }
    os.makedirs(output_dir, exist_ok=True)
    current_datetime = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    output_filename = job_name + '_' + current_datetime + '.json'
    with open(os.path.join(output_dir, output_filename), 'w') as f:
        json.dump(session_data, f)

if __name__ == '__main__':
    default_api_key_path = os.path.join(os.path.dirname(__file__), 'default_api_key.txt')
    default_api_key = None
    if os.path.exists(default_api_key_path):
        with open(default_api_key_path, 'r') as fr:
            default_api_key = fr.read().strip()

    # Create the parser
    parser = argparse.ArgumentParser(
        description="Run inference on your model with a given dataset."
    )
    
    # Job arguments
    parser.add_argument('job_name', type=str)
    parser.add_argument('--max_steps', type=int, default=30)
    parser.add_argument('--timeout', type=int, default=30)

    # IO arguments
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--data_root', type=str, default='./data/')
    parser.add_argument('--start_idx', type=int, default=0)
    parser.add_argument('--end_idx', type=int, default=9999999)
    parser.add_argument('--output_dir', type=str, default='./browsing_data')
    
    # LLM arguments
    parser.add_argument('--model', type=str, default='gpt-4o')
    parser.add_argument('--api_key', type=str, default=default_api_key)
    
    # Agent arguments
    parser.add_argument('--agent', type=str, default='reasoner')
    parser.add_argument('--config_name', type=str, default='browsergym')
    
    # Parse the arguments
    args = parser.parse_args()
    
    questions = get_dataset(args.dataset, args.data_root)
    
    for i in range(args.start_idx, min(args.end_idx, len(questions))):
        instruction = questions[i]
        job_name = args.job_name + f'_{i}'
        if glob(os.path.join(args.output_dir, f'{job_name}_*.json')) == []:
            main(job_name, 
                 args.model,
                 args.api_key,
                 args.output_dir,
                 instruction,
                 args.agent,
                 args.config_name,
                 args.max_steps,
                 args.timeout)
        else:
            print(f"Existing log detected for {job_name}, skipping ...")