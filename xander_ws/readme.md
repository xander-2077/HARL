不同的Runner:

"happo": OnPolicyHARunner,
"hatrpo": OnPolicyHARunner,
"haa2c": OnPolicyHARunner,
"haddpg": OffPolicyHARunner,
"hatd3": OffPolicyHARunner,
"hasac": OffPolicyHARunner,
"had3qn": OffPolicyHARunner,
"maddpg": OffPolicyMARunner,
"matd3": OffPolicyMARunner,
"mappo": OnPolicyMARunner,

仅考虑offpolicy+HA: haddpg/hatd3/hasac/had3qn

class OffPolicyHARunner(OffPolicyBaseRunner):

train():











class OffPolicyBaseRunner:

__init__():

对于smacv2来说：
share_observation_space:  [Box(-inf, inf, (130,), float32), Box(-inf, inf, (130,), float32), Box(-inf, inf, (130,), float32), Box(-inf, inf, (130,), float32), Box(-inf, inf, (130,), float32)]
observation_space:  [Box(-inf, inf, (92,), float32), Box(-inf, inf, (92,), float32), Box(-inf, inf, (92,), float32), Box(-inf, inf, (92,), float32), Box(-inf, inf, (92,), float32)]
action_space:  [Discrete(11), Discrete(11), Discrete(11), Discrete(11), Discrete(11)]

share observation: 用于critic/buffer的定义，相当于加一些额外的全局信息进去

定义actor, critic, buffer

run():

(new_obs, new_share_obs, rewards, dones, infos, new_available_actions, ) = self.envs.step(actions)  
# rewards: (n_threads, n_agents, 1); dones: (n_threads, n_agents); available_actions: (n_threads, ) of None or (n_threads, n_agents, action_number)


warm_up():





# 算法框架：
1. train.py
2. malawr_runner.py (malawr_base_runner.py)
3. actor: malawr.py   critic: v_critic_malawr.py
4. model:


