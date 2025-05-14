import numpy as np
import torch
import matplotlib.pyplot as plt
from snake_env import SnakeGame
from ppo_agent import PPOAgent
import os
from datetime import datetime
import json
from torch.multiprocessing import Pool, set_start_method
import torch.multiprocessing as mp

def plot_learning_curve(scores, rewards, avg_scores, save_path=None):
    plt.figure(figsize=(12, 5))
    
    # Plot scores and average scores
    plt.subplot(1, 2, 1)
    plt.plot(scores, label='Score', alpha=0.5)
    plt.plot(avg_scores, label='Average Score', color='red')
    plt.xlabel('Games')
    plt.ylabel('Score')
    plt.title('Learning Curve - Scores')
    plt.legend()
    
    # Plot rewards
    plt.subplot(1, 2, 2)
    plt.plot(rewards, label='Reward', color='green', alpha=0.5)
    plt.xlabel('Games')
    plt.ylabel('Reward')
    plt.title('Learning Curve - Rewards')
    plt.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    plt.close()

def train_worker(agent, n_games, render):
    # 为每个进程创建新的环境
    env = SnakeGame(render=render)
    
    scores = []
    rewards = []
    avg_scores = []
    total_reward = 0
    
    for i in range(n_games):
        state = env.reset()
        done = False
        score = 0
        episode_reward = 0
        
        while not done:
            action = agent.choose_action(state)
            reward, done, score = env.step(action)
            next_state = env._get_state()
            
            agent.store_transition(state, action, reward, next_state, done)
            agent.learn()
            
            state = next_state
            episode_reward += reward
        
        scores.append(score)
        rewards.append(episode_reward)
        total_reward += episode_reward
        avg_score = total_reward / (i + 1)
        avg_scores.append(avg_score)
        
        if i % 10 == 0:
            print(f'Game {i}, Score: {score}, Avg Score: {avg_score:.2f}')
    
    return scores, rewards, avg_scores

def train(env, agent, n_games=1000, render=True, num_workers=4):
    # 设置多进程启动方法
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    
    # 创建共享代理
    shared_agent = agent
    if num_workers > 1:
        shared_agent.share_memory()
    
    # 创建进程池
    with Pool(num_workers) as pool:
        # 准备参数
        args = [(shared_agent, n_games // num_workers, render) for _ in range(num_workers)]
        
        # 并行训练
        results = pool.starmap(train_worker, args)
    
    # 合并结果
    all_scores = []
    all_rewards = []
    all_avg_scores = []
    
    for scores, rewards, avg_scores in results:
        all_scores.extend(scores)
        all_rewards.extend(rewards)
        all_avg_scores.extend(avg_scores)
    
    return all_scores, all_rewards, all_avg_scores

if __name__ == '__main__':
    # 可以通过命令行参数指定是否渲染和工作进程数
    import sys
    render = False  # 默认不渲染
    num_workers = 4  # 默认4个工作进程
    
    if len(sys.argv) > 1:
        render = sys.argv[1].lower() == 'true'
    if len(sys.argv) > 2:
        try:
            num_workers = int(sys.argv[2])
        except ValueError:
            print("Invalid number of workers, using default value of 4")
    
    # Initialize environment and agent
    env = SnakeGame(render=render)
    state_dim = 18  # 状态空间维度
    action_dim = 3  # 动作空间维度
    
    # 使用最佳超参数
    best_params = {
        'hidden_dim': 128,
        'lr': 0.0003,
        'gamma': 0.99,
        'epsilon': 0.2,
        'epochs': 10,
        'batch_size': 128,  # 增加批量大小
        'gae_lambda': 0.95
    }
    
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
    
    # Train the agent
    scores, rewards, avg_scores = train(env, agent, n_games=1000, render=render, num_workers=num_workers)
    
    # Plot learning curve
    plot_learning_curve(scores, rewards, avg_scores, 'learning_curve.png')
    
    # Save the trained model
    agent.save_model('trained_model.pth')
    
    # Save metrics
    metrics = {
        'scores': scores,
        'rewards': rewards,
        'avg_scores': avg_scores,
        'best_params': best_params
    }
    with open('training_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=4) 