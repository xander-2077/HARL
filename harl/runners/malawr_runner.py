"""Runner for off-policy HARL algorithms."""
import torch
import numpy as np
import torch.nn.functional as F
from harl.runners.malawr_base_runner import MALAWRBaseRunner

class MALAWRRunner(MALAWRBaseRunner):
    """Runner for off-policy HA algorithms."""

    def train(self):
        """Train the model"""
        self.total_it += 1
        data = self.buffer.sample()
        (
            sp_share_obs,  # EP: (batch_size, dim), FP: (n_agents * batch_size, dim)
            sp_obs,  # (n_agents, batch_size, dim)
            sp_actions,  # (n_agents, batch_size, dim)
            sp_available_actions,  # (n_agents, batch_size, dim)
            sp_reward,  # EP: (batch_size, 1), FP: (n_agents * batch_size, 1)
            sp_done,  # EP: (batch_size, 1), FP: (n_agents * batch_size, 1)
            sp_valid_transition,  # (n_agents, batch_size, 1)
            sp_term,  # EP: (batch_size, 1), FP: (n_agents * batch_size, 1)
            sp_next_share_obs,  # EP: (batch_size, dim), FP: (n_agents * batch_size, dim)
            sp_next_obs,  # (n_agents, batch_size, dim)
            sp_next_available_actions,  # (n_agents, batch_size, dim)
            sp_gamma,  # EP: (batch_size, 1), FP: (n_agents * batch_size, 1)
        ) = data  # 都用上

        # train critic
        # ------------------------------------------------------------------------
        # update the critic using Monte-Carlo Estimates
        sp_value = self.critic.get_values(sp_share_obs, sp_actions)
        discounted_reward, _ = discount_return(sp_reward, sp_done, cur_value.cpu().detach().numpy())  # compute R_{s,a}^D, using GAE (TD(lambda) is better)
        mse = torch.nn.MSELoss()
        for _ in range(critic_update_iter):
            sample_idx = random.sample(range(data_len), 256)
            sample_value = self.model.critic(torch.FloatTensor(s_batch[sample_idx]))
            if (torch.sum(torch.isnan(sample_value)) > 0):
                print('NaN in value prediction')
                input()
            critic_loss = mse(sample_value.squeeze(), torch.FloatTensor(discounted_reward[sample_idx]))
            critic_loss.backward()
            self.critic_optimizer.step()
            self.critic_optimizer.zero_grad()
        # ------------------------------------------------------------------------

        # self.critic.turn_on_grad()
        next_actions = []
        for agent_id in range(self.num_agents):
            next_actions.append(
                self.actor[agent_id].get_target_actions(sp_next_obs[agent_id])
            )
        self.critic.train(
            sp_share_obs,
            sp_actions,
            sp_reward,
            sp_done,
            sp_term,
            sp_next_share_obs,
            next_actions,
            sp_gamma,
        )
        # self.critic.turn_off_grad()
        sp_valid_transition = torch.tensor(sp_valid_transition, device=self.device)

        # train actors
        if self.total_it % self.policy_freq == 0:
            actions = []
            with torch.no_grad():
                for agent_id in range(self.num_agents):
                    actions.append(
                        self.actor[agent_id].get_actions(
                            sp_obs[agent_id], False
                        )
                    )
            # actions shape: (n_agents, batch_size, dim)
            if self.fixed_order:
                agent_order = list(range(self.num_agents))
            else:
                agent_order = list(np.random.permutation(self.num_agents))  # 随机打乱agent的顺序
            for agent_id in agent_order:
                self.actor[agent_id].turn_on_grad()
                # train this agent
                actions[agent_id] = self.actor[agent_id].get_actions(
                    sp_obs[agent_id], False
                )
                actions_t = torch.cat(actions, dim=-1)
                value_pred = self.critic.get_values(sp_share_obs, actions_t)
                actor_loss = -torch.mean(value_pred)
                self.actor[agent_id].actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor[agent_id].actor_optimizer.step()
                self.actor[agent_id].turn_off_grad()
                actions[agent_id] = self.actor[agent_id].get_actions(
                    sp_obs[agent_id], False
                )

            # soft update
            for agent_id in range(self.num_agents):
                self.actor[agent_id].soft_update()
            self.critic.soft_update()
