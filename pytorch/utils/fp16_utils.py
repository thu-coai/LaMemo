import torch

'''
Utils for stablizing mixed-precision training
'''

def clamp_hidden(hidden_states):
    if torch.isinf(hidden_states).any():
        clamp_value = torch.finfo(hidden_states.dtype).max - 1000
        hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
    return hidden_states