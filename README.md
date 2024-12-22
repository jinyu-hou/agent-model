# Web Agent Example:

This is an example code for running the web agent with modular architecture, with choises of world-model-based planning module or autoregressive planning module. We also offer a baseline agent based on `BrowsingAgent` from [OpenHands](https://github.com/All-Hands-AI/OpenHands).

## Setup

Please install the dependencies specific to this example using the command below:

```
pip install -r requirements.txt
```

## Datasets

We provide two datasets for evaluating web agents as informational assistants: 
1. [FanOutQA](https://fanoutqa.com/index.html), which requires the agent to answer questions that require searching for and compiling information from multiple websites. We include their development set of 310 examples. 
2. FlightQA, a dataset prepared by us to evaluate the ability of LLM agents in answering queries with varying number of constraints, specifically while searching for live flight tickets using the internet. To control for confounding variables like specific query content, we iteratively add to lists of constraints to form new questions. In total we have 120 examples consisted of 20 groups of questions ranging from 3 to 8 constraints.
3. WebArena, a standalone, self-hostable web environment for web agent benchmarking and evaluation.

## Commands

To run evaluation using one of our datasets, use the following command as an example:

```
python main.py \
    [job_name] \
    --dataset [fanout, flightqa, webarena] \
    --agent [reasoner, openhands] \
    --config_name [only applies to reasoner agent; 
                   options: browsergym, browsergym_world_model, opendevin, opendevin_llama, opendevin_world_model,
                   opendevin_webarena, opendevin_webarena_world_model,
                   browsergym_webarena, browsergym_webarena_world_model; 
                   default: browsergym]
    --model [any model accessible via litellm; default: gpt-4o]
    --start_idx [index of the first example] \
    --end_idx [index of the last example] 
```

One way to speed up the inference is to open several terminals and run inference on separate slices of the data.

Before that, you'll need to enter your default API key at `default_api_key.txt`, or set the `OPENAI_API_KEY` environment variable.

The agent outputs will be stored under `browsing_data` and will be automatically loaded for evaluation.

## Evaluation

```
python evaluation/fanout/run.py [job_name]

python evaluation/flight/run.py [job_name]

bash evaluation/webarena/run_inference.py
```

Note: Running evaluation for FlightQA involves calling `gpt-4o`, which may incur costs to you. So please run judiciously.

## Installation

If any issue arises while trying to install BrowserGym, please refer to the [official repo](https://github.com/ServiceNow/BrowserGym).
