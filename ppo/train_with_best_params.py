import json
import os
from train import train
from snake_env import SnakeGame
from ppo_agent import PPOAgent

def train_with_best_params(n_games=1000):
    # Load best parameters
    with open('best_params/ppo_best_params.json', 'r') as f:
        best_params = json.load(f)
    
    # Initialize environment and agent with best parameters
    env = SnakeGame()
    state_dim = 11
    action_dim = 3
    
    agent = PPOAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=best_params['hidden_dim'],
        lr=best_params['lr'],
        gamma=best_params['gamma'],
        epsilon=best_params['epsilon'],
        epochs=best_params['epochs'],
        batch_size=best_params['batch_size'],
        gae_lambda=best_params['gae_lambda']
    )
    
    # Train with best parameters
    print("Training with best parameters:")
    for key, value in best_params.items():
        print(f"  {key}: {value}")
    
    scores, avg_scores = train(env, agent, n_games)
    
    return scores, avg_scores

if __name__ == '__main__':
    train_with_best_params() 