import optuna
import numpy as np
import torch
from train import train
from snake_env import SnakeGameAI, BLOCK_SIZE
from dqn_agent import DQNAgent
import os
import json
from datetime import datetime

def objective(trial):
    # 定义超参数搜索空间
    params = {
        'hidden_size': trial.suggest_int('hidden_size', 64, 512),
        'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True),
        'gamma': trial.suggest_float('gamma', 0.9, 0.999),
        'epsilon': trial.suggest_float('epsilon', 0.1, 1.0),
        'memory_size': trial.suggest_int('memory_size', 10000, 100000),
        'batch_size': trial.suggest_int('batch_size', 32, 512),
        'target_update': trial.suggest_int('target_update', 5, 20),
        'base_reward': trial.suggest_int('base_reward', 20, 50),
        'step_bonus_multiplier': trial.suggest_int('step_bonus_multiplier', 10, 50),
        'step_bonus_decay': trial.suggest_int('step_bonus_decay', 30, 100),
        'length_bonus': trial.suggest_float('length_bonus', 1.0, 5.0),
        'death_penalty': trial.suggest_int('death_penalty', -30, -10),
        'distance_reward': trial.suggest_float('distance_reward', 0.1, 0.5)
    }
    
    # 创建环境
    env = SnakeGameAI(render_ui=False)
    
    # 设置奖励参数
    env.base_reward = params['base_reward']
    env.step_bonus_multiplier = params['step_bonus_multiplier']
    env.step_bonus_decay = params['step_bonus_decay']
    env.length_bonus = params['length_bonus']
    env.death_penalty = params['death_penalty']
    env.distance_reward = params['distance_reward']
    
    # 创建智能体
    agent = DQNAgent(
        state_size=env.get_state_size(),
        action_size=3,  # 修改为3个动作：直行、右转、左转
        hidden_size=params['hidden_size'],
        learning_rate=params['learning_rate'],
        gamma=params['gamma'],
        epsilon=params['epsilon'],
        memory_size=params['memory_size'],
        batch_size=params['batch_size'],
        target_update=params['target_update']
    )
    
    # 训练模型
    scores = []
    max_episodes = 500  # 增加训练轮数
    min_episodes = 100  # 最少训练轮数
    early_stop_threshold = 5  # 降低提前停止的分数阈值
    
    for episode in range(max_episodes):
        state = env.reset()
        done = False
        score = 0
        steps = 0
        
        for step in range(2000):  # 增加每轮的最大步数
            action_idx = agent.get_action(state)
            action = [0, 0, 0]
            action[action_idx] = 1
            
            reward, done, score = env.step(action)
            next_state = env.get_state()
            
            agent.remember(state, action_idx, reward, next_state, done)
            agent.replay()
            
            state = next_state
            steps += 1
            
            if done:
                break
        
        scores.append(score)
        
        # 提前停止条件
        if episode >= min_episodes:  # 至少训练min_episodes轮
            recent_avg = np.mean(scores[-20:])  # 使用最近20轮的平均分数
            if recent_avg > early_stop_threshold:  # 降低提前停止的分数阈值
                print(f"提前停止：最近20轮平均分数 {recent_avg:.2f} > {early_stop_threshold}")
                break
    
    # 计算最终得分（使用最近20轮的平均分数）
    final_score = np.mean(scores[-20:]) if len(scores) >= 20 else np.mean(scores)
    
    # 记录最佳参数
    if trial.number == 0 or final_score > trial.study.best_value:
        save_best_params(params, final_score)
    
    return final_score

def save_best_params(params, score):
    """保存最佳参数到文件"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result = {
        'score': score,
        'params': params,
        'timestamp': timestamp
    }
    
    # 创建results目录（如果不存在）
    if not os.path.exists('results'):
        os.makedirs('results')
    
    # 保存结果
    filename = f'results/best_params_{timestamp}.json'
    with open(filename, 'w') as f:
        json.dump(result, f, indent=4)
    print(f"最佳参数已保存到: {filename}")

def main():
    # 创建Optuna学习对象
    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=42)
    )
    
    # 开始优化
    print("开始超参数优化...")
    study.optimize(objective, n_trials=30)  # 减少试验次数，但增加每次试验的训练轮数
    
    # 打印最佳结果
    print("\n最佳参数:")
    print(f"最佳分数: {study.best_value}")
    print("\n最佳参数组合:")
    for key, value in study.best_params.items():
        print(f"{key}: {value}")

if __name__ == "__main__":
    main() 