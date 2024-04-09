import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical
import torch.nn.functional as F

class Actor(nn.Module):
    def __init__(self, state_dim, context_dim, action_dim, has_continuous_action_space):
        super(Actor, self).__init__()
        self.linear1 = nn.Linear(state_dim, 64)
        self.linear2 = nn.Linear(64, 64)
        self.linear3 = nn.Linear(64 + context_dim, action_dim)

        self.has_continuous_action_space = has_continuous_action_space
    
    def forward(self, state, context):
        if len(state.shape) > 1:
            context = context.unsqueeze(1)
            concat_dim = 1
        else:
            concat_dim = 0
            
        x = F.tanh(self.linear1(state))
        x = F.tanh(self.linear2(x))

        x = torch.cat((x, context), dim=concat_dim)
        output = self.linear3(x)

        if self.has_continuous_action_space:
            output = F.tanh(output)
        else:
            output = F.softmax(output, dim=-1)

        return output
    
class Critic(nn.Module):
    def __init__(self, state_dim, context_dim):
        super(Critic, self).__init__()
        self.linear1 = nn.Linear(state_dim, 64)
        self.linear2 = nn.Linear(64, 64)
        self.linear3 = nn.Linear(64 + context_dim, 1)
    
    def forward(self, state, context):
        if len(state.shape) > 1:
            context = context.unsqueeze(1)
            concat_dim = 1
        else:
            concat_dim = 0
    
        x = F.tanh(self.linear1(state))
        x = F.tanh(self.linear2(x))
        x = torch.cat((x, context), dim=concat_dim)
        output = self.linear3(x)
            
        return output

################################## set device ##################################
print("============================================================================================")
# set device to cpu or cuda
device = torch.device('cpu')
if(torch.cuda.is_available()): 
    device = torch.device('cuda:0') 
    torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    print("Device set to : cpu")
print("============================================================================================")


class InsertAC(nn.Module):
    def __init__(self, state_dim, context_dim, action_dim, has_continuous_action_space, action_std_init):
        super(InsertAC, self).__init__()

        self.has_continuous_action_space = has_continuous_action_space
        
        if has_continuous_action_space:
            self.action_dim = action_dim
            self.action_var = torch.full((action_dim,), action_std_init * action_std_init).to(device)
        
        # actor
        self.actor = Actor(state_dim, context_dim, action_dim, has_continuous_action_space)
    
        # critic
        self.critic = Critic(state_dim, context_dim)
        
    def set_action_std(self, new_action_std):
        if self.has_continuous_action_space:
            self.action_var = torch.full((self.action_dim,), new_action_std * new_action_std).to(device)
        else:
            print("--------------------------------------------------------------------------------------------")
            print("WARNING : Calling ActorCritic::set_action_std() on discrete action space policy")
            print("--------------------------------------------------------------------------------------------")

    def forward(self):
        raise NotImplementedError
    
    def act(self, state, context):
        if self.has_continuous_action_space:
            action_mean = self.actor(state, context)
            cov_mat = torch.diag(self.action_var).unsqueeze(dim=0)
            dist = MultivariateNormal(action_mean, cov_mat)
        else:
            action_probs = self.actor(state, context)
            dist = Categorical(action_probs)

        action = dist.sample()
        action_logprob = dist.log_prob(action)
        state_val = self.critic(state, context)

        return action.detach(), action_logprob.detach(), state_val.detach()
    
    def evaluate(self, state, context, action):
        if self.has_continuous_action_space:
            action_mean = self.actor(state, context)
            
            action_var = self.action_var.expand_as(action_mean)
            cov_mat = torch.diag_embed(action_var).to(device)
            dist = MultivariateNormal(action_mean, cov_mat)
            
            # For Single Action Environments.
            if self.action_dim == 1:
                action = action.reshape(-1, self.action_dim)
        else:
            action_probs = self.actor(state, context)
            dist = Categorical(action_probs)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(state, context)
        
        return action_logprobs, state_values, dist_entropy