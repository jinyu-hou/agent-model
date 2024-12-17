from typing import Any
from logging import Logger
from .base import AgentModule, AgentVariable
from .llm import LLM, OpenDevinParserLLM, OpenDevinParserMultiResponseLLM
from .modules import (
    PolicyPlanner, ReasonerPlanner,
    PromptedActor, PromptedCritic, PromptedEncoder,
    PromptedPolicy, PromptedWorldModel
)
from .variables import (
    AgentInstructionEnvironmentIdentity,
    BrowserGymActionSpace, OpenDevinBrowserActionSpace, 
    BrowserGymObservationSpace, OpenDevinBrowserObservationSpace,
    StepKeyValueMemory, StepPromptedMemory
)

from .prompts import (
    actor_prompt_template_dict,
    critic_prompt_template,
    encoder_prompt_template_dict,
    memory_update_prompt_template_dict,
    policy_prompt_template_dict,
    world_model_prompt_template_dict,
)

from .configs import (browsergym_config, browsergym_world_model_config, 
                      opendevin_config, opendevin_world_model_config, 
                      webarena_config, webarena_world_model_config)

CONFIG_LIBRARY = {
    'browsergym': browsergym_config,
    'browsergym_world_model': browsergym_world_model_config,
    'opendevin': opendevin_config,
    'opendevin_llama': opendevin_config,
    'opendevin_world_model': opendevin_world_model_config,
    'webarena': webarena_config,
    'webarena_world_model': webarena_world_model_config,
}

class ReasonerAgent:
    def __init__(self, 
                 llm: Any,
                 config_name: str,
                 logger: Logger = None,
                 **kwargs):
        self.llm = llm
        self.config_name = config_name
        self.config = CONFIG_LIBRARY[config_name]
        self.logger = logger
        
        self.environment = self.config['environment']
        self.encoder_prompt_type = self.config['encoder_prompt_type']
        self.planner_type = self.config['planner_type']
        self.actor_prompt_type = self.config['actor_prompt_type']
        self.memory_type = self.config['memory_type']
        
        self.policy_output_name = self.config['policy_output_name']
        self.module_error_message = self.config['module_error_message']
        
        # Action space and observation space
        if self.environment == 'browsergym':
            self.action_space = BrowserGymActionSpace(
                action_subsets=['chat', 'bid'],
                use_nav=True,
                strict=False,
                multiaction=False,
            )
            self.observation_space = BrowserGymObservationSpace()
        elif self.environment == 'opendevin':
            self.action_space = OpenDevinBrowserActionSpace(
                action_subsets=['chat', 'bid'],
                use_nav=self.config['use_nav'],
                strict=False,
                multiaction=False,
            )
            self.observation_space = OpenDevinBrowserObservationSpace(
                eval_mode=self.config['eval_mode'], 
                truncation=self.config['truncate_axtree']
            )
        else: 
            raise ValueError(f'Unsupported environment: {self.environment}')
        
        # Agent identity
        self.identity = AgentInstructionEnvironmentIdentity(
            agent_name=self.config['agent_name'],
            agent_description=self.config['agent_description'],
            observation_space=self.observation_space,
            action_space=self.action_space,
        )
        
        # Encoder
        self.encoder_llm = OpenDevinParserLLM(llm, ['state'])
        encoder_prompt_template = encoder_prompt_template_dict[self.encoder_prompt_type]
        self.encoder = PromptedEncoder(
            self.identity, self.encoder_llm, prompt_template=encoder_prompt_template
        )
        
        # Memory
        if self.memory_type == 'step_prompted':
            self.memory_update_llm = OpenDevinParserLLM(llm, ['memory_update'])
            memory_update_prompt_template = memory_update_prompt_template_dict[self.config['memory_prompt_type']]
            self.memory = StepPromptedMemory(self.identity, self.memory_update_llm, 
                                             prompt_template=memory_update_prompt_template, 
                                             keys=[self.policy_output_name])
        elif self.memory_type == 'step_key_value':
            self.memory = StepKeyValueMemory(['state', self.policy_output_name])
        else: 
            raise ValueError(f'Invalid memory type: {self.memory_type}')
        
        # Planner
        if self.planner_type == 'world_model':
            self.policy_prompt_type = self.config['policy_prompt_type']
            self.world_model_prompt_type = self.config['world_model_prompt_type']
            self.planner_search_num_actions = self.config['planner_search_num_actions']
            self.planner_search_depth = self.config['planner_search_depth']
            self.planner_critic_num_samples = self.config['planner_critic_num_samples']
            
            self.policy_llm = OpenDevinParserMultiResponseLLM(
                llm, [self.policy_output_name], ['think']
            )
            policy_prompt_template = policy_prompt_template_dict[self.policy_prompt_type]
            self.policy = PromptedPolicy(
                self.identity, self.policy_llm, prompt_template=policy_prompt_template
            )

            self.world_model_llm = OpenDevinParserLLM(
                llm, ['next_state']
            )
            world_model_prompt_template = world_model_prompt_template_dict[self.world_model_prompt_type]
            self.world_model = PromptedWorldModel(
                self.identity,
                self.world_model_llm,
                prompt_template=world_model_prompt_template,
            )

            self.critic_llm = OpenDevinParserMultiResponseLLM(
                llm, ['status', 'on_the_right_track'], ['think']
            )
            self.critic = PromptedCritic(
                self.identity, self.critic_llm, prompt_template=critic_prompt_template
            )
            
            self.planner = ReasonerPlanner(self.policy, self.world_model, self.critic,
                                              search_num_actions=self.planner_search_num_actions,
                                              search_depth=self.planner_search_depth,
                                              policy_output_name=self.policy_output_name,
                                              critic_num_samples=self.planner_critic_num_samples,
                                              llm_base_url=llm.base_url,
                                              llm_api_key=llm.api_key)
            
        elif self.planner_type == 'policy':
            self.policy_prompt_type = self.config['policy_prompt_type']
            policy_prompt_template = policy_prompt_template_dict[self.policy_prompt_type]
            
            self.policy_llm = OpenDevinParserLLM(
                llm, [self.policy_output_name], ['think']
            )
            self.policy = PromptedPolicy(
                self.identity, self.policy_llm, prompt_template=policy_prompt_template
            )
            
            self.planner = PolicyPlanner(self.policy)

        # Actor
        self.actor_llm = OpenDevinParserLLM(llm, ['action'])
        actor_prompt_template = actor_prompt_template_dict[self.actor_prompt_type]
        self.actor = PromptedActor(
            self.identity, self.actor_llm, prompt_template=actor_prompt_template
        )

        self.reset()

    def reset(self):
        for attr_name in dir(self): 
            # Skip special methods and attributes
            if not attr_name.startswith('__'):
                attr_value = getattr(self, attr_name)
                if isinstance(attr_value, LLM) or isinstance(attr_value, AgentModule):
                    attr_value.logger = self.logger
                elif isinstance(attr_value, AgentVariable): 
                    attr_value.reset()
        
        self.last_action = ''
        self.num_repeats = 0
        self.total_cost = 0
        
    def _maybe_log(self, msg): 
        if self.logger is not None: 
            self.logger.info(msg)
            
    def _finish_with_module_error(self, step_info): 
        return_action = self.module_error_message
        step_info.update({'action': return_action})
        return return_action, step_info
    
    def _log_total_accumulated_cost(self):
        total_cost = 0
        for attr_name in dir(self): 
            # Skip special methods and attributes
            if not attr_name.startswith('__'):
                attr_value = getattr(self, attr_name)
                if isinstance(attr_value, LLM):
                    total_cost += attr_value.cost_accumulator
                    
        self.total_cost = total_cost
        self._maybe_log(f'*Total Accumulated Cost*: {total_cost:.2f}')
    
    def step(self, raw_obs):
        step_info = {}
        
        # observation, info = self.observation_space.parse_observation(raw_obs)
        obs, obs_info = self.observation_space.parse_observation(raw_obs)
        self._maybe_log(f'*Observation*: {obs}')
        step_info.update({'obs': obs, 'obs_info': obs_info})
        if obs_info.get('return_action') is not None:
            action = obs_info['return_action']
            step_info.update({'action': action})
            return self.action_space.parse_action(action, step_info)
        self.identity.update(user_instruction=obs_info['goal'])
        
        kwargs = {}
        state = self.encoder(obs, self.memory).get('state')
        self._maybe_log(f'*State*: {state}')
        step_info.update({'state': state})
        if not state: return self._finish_with_module_error(step_info)

        plan = self.planner(state, self.memory, **kwargs).get(self.policy_output_name)
        self._maybe_log(f'*Plan*: {plan}')
        step_info.update({'plan': plan, self.policy_output_name: plan})
        if not plan: return self._finish_with_module_error(step_info)

        action = self.actor(obs, state, self.memory, plan, **kwargs).get('action')
        self._maybe_log(f'*Action*: {action}')
        step_info.update({'action': action})
        if not action: return self._finish_with_module_error(step_info)
        
        try:
            self.memory.update(**step_info)
        except KeyError as e: 
            print(e)
            return self._finish_with_module_error(step_info)
        step_info.update(self.memory.current_step)
        if self.memory_type == 'step_prompted':
            self._maybe_log(f"*Memory update*: {self.memory.current_step['memory_update']}")
        self.memory.step()
        
        self._log_total_accumulated_cost()
            
        return self.action_space.parse_action(action, step_info)