import json
import torch
from snake_env import SnakeGameAI
from dqn_agent import DQNAgent
import numpy as np
import os
from datetime import datetime

def load_best_params(file_path):
    """加载最佳参数"""
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data['params']

def train_with_params(params, episodes=2000, render=True):
    """使用指定参数进行训练"""
    # 创建环境
    env = SnakeGameAI(render_ui=render)
    
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
        action_size=3,
        hidden_size=params['hidden_size'],
        learning_rate=params['learning_rate'],
        gamma=params['gamma'],
        epsilon=params['epsilon'],
        memory_size=params['memory_size'],
        batch_size=params['batch_size'],
        target_update=params['target_update']
    )
    
    # 训练记录
    scores = []
    avg_scores = []
    best_score = 0
    
    # 创建保存模型的目录
    if not os.path.exists('models'):
        os.makedirs('models')
    
    # 训练循环
    for episode in range(episodes):
        state = env.reset()
        done = False
        score = 0
        steps = 0
        
        while not done:
            # 获取动作
            action_idx = agent.get_action(state)
            action = [0, 0, 0]
            action[action_idx] = 1
            
            # 执行动作
            reward, done, score = env.step(action)
            next_state = env.get_state()
            
            # 存储经验
            agent.remember(state, action_idx, reward, next_state, done)
            
            # 训练网络
            agent.replay()
            
            state = next_state
            steps += 1
        
        # 记录分数
        scores.append(score)
        avg_score = np.mean(scores[-100:])  # 最近100轮的平均分数
        avg_scores.append(avg_score)
        
        # 保存最佳模型
        if score > best_score:
            best_score = score
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_path = f'models/best_model_{timestamp}.pth'
            torch.save(agent.model.state_dict(), model_path)
            print(f"新的最佳分数: {best_score}, 模型已保存到: {model_path}")
        
        # 打印训练进度
        if episode % 10 == 0:
            print(f"Episode: {episode}, Score: {score}, Avg Score: {avg_score:.2f}, Steps: {steps}")
    
    return scores, avg_scores

def main():
    # 加载最佳参数
    params_file = 'results/best_params_20250511_095727.json'
    params = load_best_params(params_file)
    
    print("使用以下参数进行训练:")
    for key, value in params.items():
        print(f"{key}: {value}")
    
    # 开始训练
    print("\n开始训练...")
    scores, avg_scores = train_with_params(params, episodes=2000, render=True)
    
    # 保存训练记录
    if not os.path.exists('training_logs'):
        os.makedirs('training_logs')
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f'training_logs/training_log_{timestamp}.json'
    
    log_data = {
        'params': params,
        'scores': scores,
        'avg_scores': avg_scores,
        'timestamp': timestamp
    }
    
    with open(log_file, 'w') as f:
        json.dump(log_data, f, indent=4)
    
    print(f"\n训练完成！训练记录已保存到: {log_file}")

if __name__ == "__main__":
    main() 