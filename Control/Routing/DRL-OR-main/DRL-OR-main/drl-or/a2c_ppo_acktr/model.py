import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from a2c_ppo_acktr.distributions import Categorical, MultiCategorical, MultiTypeCategorical
from a2c_ppo_acktr.utils import init

class Policy(nn.Module):
    def __init__(self, obs_shape, action_space, condition_space, node_num=None, type_num=None, base=None, base_kwargs=None):
        super(Policy, self).__init__()
        self._node_num = node_num
        self._type_num = type_num

        if base_kwargs is None:
            base_kwargs = {}
        if base is None:
            base = MLPBase
        
        # temporarily not elimate the dst_state from the input, only for the test
        self.base = base(obs_shape[0], condition_space, **base_kwargs)
        # 这里base就相当于一个替换的变量，实际是MLPBase
        
        # now only use Discrete sampling layer 
        if action_space.__class__.__name__ == "Discrete":
            if node_num == None and type_num == None:
                num_outputs = action_space.n
                self.dist = Categorical(self.base.output_size, num_outputs)
            elif type_num == None and node_num != None:
                num_outputs = action_space.n
                self.dist = MultiCategorical(self.base.output_size, num_outputs, node_num)
            elif type_num != None and node_num != None:
                num_outputs = action_space.n
                self.dist = MultiTypeCategorical(self.base.output_size, num_outputs, node_num, type_num)
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError

    @property
    def is_recurrent(self):
        return self.base.is_recurrent

    @property
    def recurrent_hidden_state_size(self):
        """Size of rnn_hx."""
        return self.base.recurrent_hidden_state_size

    def forward(self, inputs, rnn_hxs):
        raise NotImplementedError

    '''
    @param:
        inputs: torch.tensor([[x, y, z, ...], ...]) [batch, state_size]
        rnn_hxs: torch.tensor([[x, y, z, ...], ...]) [batch, hidden_size]
        condition_state: [batch, condition_state_size]
    @retval:
        value: shape[batch, 1]
        action: shape[batch, action_shape] # action_shape=1
        action_log_probs: shape[batch, 1]
        rnn_hxs: [batch, hidden_size]
        求action
    '''
    def act(self, inputs, rnn_hxs, condition_state, deterministic=False):
        if self._node_num != None:
            dst_state = inputs[:, -self._node_num:]
            if self._type_num != None:
                type_state = inputs[:, -self._node_num * 2 - self._type_num:-self._node_num * 2]

        # self.base集actor和critic于一身
        # value：V(o_t)
        # actor_features：actor网络的倒数第二层
        # states：RNN的hidden states
        value, actor_features, rnn_hxs = self.base(inputs.unsqueeze(0), rnn_hxs, condition_state.unsqueeze(0))
        value = value.squeeze(0)
        actor_features = actor_features.squeeze(0)

        if self._node_num == None:
            dist = self.dist(actor_features)
        elif self._type_num == None:
            dist = self.dist(actor_features, dst_state)
        else:
            # dist最后的action分布\pi(A_t | o_t)
            dist = self.dist(actor_features, dst_state, type_state)

        if deterministic:
            # argmax
            action = dist.probs.argmax(dim=-1, keepdim=True)
        else:
            action = dist.sample().unsqueeze(-1)
            # 返回按概率抽样的邻居的下标

        action_log_probs = dist.log_prob(action.squeeze(-1))

        return value, action, action_log_probs.unsqueeze(-1), rnn_hxs

    '''
    @param:
        inputs: torch.tensor([[x, y, z, ...], ...]) [batch, state_size]
        rnn_hxs: torch.tensor([[x, y, z, ...], ...]) [batch, hidden_size]
        condition_state: [batch, condition_state_size], actually we don't need condition state for get_value function
    @retval:
        value: shape[batch, 1]
    '''
    def get_value(self, inputs, rnn_hxs, condition_state):
        value, _, _ = self.base(inputs.unsqueeze(0), rnn_hxs, condition_state.unsqueeze(0))
        value = value.squeeze(0)
        return value

    '''
    @param:
        inputs: [seq_len, batch, input_size]
        rnn_hxs: [batch, hidden_size]
        condition_state: [seq_len, batch, condition_state_size]
        action: [seq_len, batch, action_shape] # action_shape should equal 1
        in the router expr action_shape = 1
    @retval:
        value: shape[seq_len, batch, 1]
        action_log_probs: shape[seq_len, batch, 1]
        dist_entropy: shape: torch.tensor(x) 
        rnn_hxs: [batch, hidden_size]
        从网络中获取参数更新要用到的信息（e.g., V(ot),logπ(at|ot)）。
    '''
    def evaluate_actions(self, inputs, rnn_hxs, condition_state, action):
        if self._node_num != None:
            dst_state = inputs[:, :, -self._node_num:]
            if self._type_num != None:
                type_state = inputs[:, :, -self._node_num * 2 - self._type_num:-self._node_num * 2]

        value, actor_features, rnn_hxs = self.base(inputs, rnn_hxs, condition_state)

        if self._node_num == None:
            dist = self.dist(actor_features)
        elif self._type_num == None:
            dist = self.dist(actor_features, dst_state)
        else:
            dist = self.dist(actor_features, dst_state, type_state)

        action_log_probs = dist.log_prob(action.squeeze(-1)) # input action should have shape [(len,)batch]
        dist_entropy = dist.entropy().mean()

        return value, action_log_probs.unsqueeze(-1), dist_entropy, rnn_hxs

'''
For RNN based architecture, when training we need to generate the recurrent hidden state sequentially(i.e. only use the first hidden state for a sequence input) 
In this work we usually use batch = 1 since we didn't use parrel enviroments training
对于基于RNN的架构，在训练时，我们需要依次生成循环隐藏状态(即RNN)。在这个工作中，我们通常使用batch = 1，因为我们没有使用并行环境训练
'''
class NNBase(nn.Module):

    def __init__(self, recurrent, recurrent_input_size, hidden_size):
        super(NNBase, self).__init__()

        self._hidden_size = hidden_size
        self._recurrent = recurrent

        if recurrent:   # False 跳过
            self.gru = nn.GRU(recurrent_input_size, hidden_size)
            for name, param in self.gru.named_parameters():
                if 'bias' in name:
                    nn.init.constant_(param, 0)
                elif 'weight' in name:
                    nn.init.orthogonal_(param)

    @property
    def is_recurrent(self):
        return self._recurrent

    @property
    def recurrent_hidden_state_size(self):
        if self._recurrent:
            return self._hidden_size
		# for the simplicity of main function implementation
        return 1

    @property
    def output_size(self):
        return self._hidden_size

    '''
    @param:
        x: shape[seq_len, batch, input_size]
        hxs: shape[batch, hidden_size]
    @retval:
        x: shape[seq_len, batch, hidden_size]
        hxs: shape[batch, hidden_size]
    the GRU's hiddenlayer = 1 and direction = 1
    '''
    def _forward_gru(self, x, hxs):
        x, hxs = self.gru(x, hxs.unsqueeze(0))
        hxs = hxs.squeeze(0) # squeeze since we only use one layer one direction gru
        return x, hxs


class MLPBase(NNBase):
    def __init__(self, num_inputs, condition_state_size, recurrent=False, hidden_size=64):
        super(MLPBase, self).__init__(recurrent, num_inputs, hidden_size)

        # 实际运行中NNBase并没有用
        if recurrent:
            num_inputs = hidden_size

        init_ = lambda m: init(m,
            nn.init.orthogonal_,
            lambda x: nn.init.constant_(x, 0),
            np.sqrt(2))

        # build a sequential model(multi-layer model) and init it
        self.actor = nn.Sequential(
            init_(nn.Linear(num_inputs + condition_state_size, hidden_size)),
            nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)),
            nn.Tanh(),
        )

        self.critic = nn.Sequential(
            init_(nn.Linear(num_inputs, hidden_size)),
            nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)),
            nn.Tanh()
        )

        self.critic_linear = init_(nn.Linear(hidden_size, 1))

        self.train()
    '''
    @param:
        x: shape[seq_len, batch, input_size]
        hxs: shape[batch, hidden_size]
        condition_state: shape[seq_len, batch, condition_state_size]
    @retval:
        value: shape[seq_len, batch, 1]
        hidden_actor: shape[seq_len, batch, hidden_size]
        rnn_hxs: shape[batch, hidden_size]
    '''
    def forward(self, inputs, rnn_hxs, condition_state):
        x = inputs

        if self.is_recurrent:
            x, rnn_hxs = self._forward_gru(x, rnn_hxs)

        # 将状态送入critic中，经过后面的线性层应该输出的是期望的奖励next_value
        hidden_critic = self.critic(x)
        # 将状态数组和条件数组拼接，送到actor中输出特征
        concat_input = torch.cat([condition_state, x], -1)
        hidden_actor = self.actor(concat_input)

        return self.critic_linear(hidden_critic), hidden_actor, rnn_hxs
