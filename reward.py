import numpy as np

def reward_function(history):
    return (history['portfolio_valuation', -1] - history['portfolio_valuation', -2])

def reward_function_rapport_market(history):
    return 10*((history['portfolio_valuation', -1] / history['portfolio_valuation', -2]) - (history['data_close', -1] / history['data_open', -1]))

def reward_function_rapport_market_v2(history):
    weight = 0 
    if (history['position',1] == history['position',-2]):
        weight = 0.1
    return 10*(history['data_open', -2] / history['data_close', -1])*((history['portfolio_valuation', -1] / history['portfolio_valuation', -2]) - (history['data_close', -1] / history['data_open', -1])) - weight 

def reward_log_returns(history):
    prev_val = history['portfolio_valuation', -2]
    curr_val = history['portfolio_valuation', -1]
    
    if prev_val == 0: return 0
    
    reward = np.log(curr_val / prev_val)
    return reward

def reward_returns(history):
    prev_val = history['portfolio_valuation', -2]
    curr_val = history['portfolio_valuation', -1]
    
    if prev_val == 0: return 0
    
    reward = (curr_val - prev_val) / prev_val
    return reward

def reward_risk_adjusted(history):
    prev_val = history['portfolio_valuation', -2]
    curr_val = history['portfolio_valuation', -1]
    if prev_val == 0: return 0    
    log_return = np.log(curr_val / prev_val)
    risk_penalty = 0
    if log_return < 0:
        risk_penalty = abs(log_return) * 0.5 
        
    return log_return - risk_penalty
