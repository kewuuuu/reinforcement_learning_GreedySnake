import optuna
import json
import os
from train import train
from snake_env import SnakeGame
from ppo_agent import PPOAgent
import pickle
from datetime import datetime
import torch
import torch.multiprocessing as mp
from torch.multiprocessing import Pool, set_start_method
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt

def objective(trial):
    # 网络参数
    hidden_dim = trial.suggest_int('hidden_dim', 32, 256)
    lr = trial.suggest_float('lr', 1e-5, 1e-3, log=True)
    gamma = trial.suggest_float('gamma', 0.9, 0.999)
    epsilon = trial.suggest_float('epsilon', 0.1, 0.3)
    epochs = trial.suggest_int('epochs', 5, 20)
    batch_size = trial.suggest_int('batch_size', 32, 256)
    gae_lambda = trial.suggest_float('gae_lambda', 0.9, 0.99)
    
    # 早停参数
    early_stop_window = trial.suggest_int('early_stop_window', 20, 50)  # 评估窗口大小
    early_stop_threshold = trial.suggest_float('early_stop_threshold', 0.1, 0.5)  # 早停阈值
    min_games = trial.suggest_int('min_games', 30, 50)  # 最小游戏数
    patience = trial.suggest_int('patience', 3, 7)  # 容忍次数
    
    # 奖励参数
    reward_params = {
        'food_reward': trial.suggest_float('food_reward', 10.0, 100.0),
        'death_penalty': trial.suggest_float('death_penalty', -20.0, -5.0),
        'distance_reward_factor': trial.suggest_float('distance_reward_factor', 0.1, 2.0),
        'distance_penalty_factor': trial.suggest_float('distance_penalty_factor', 0.1, 1.0),
        'direction_reward_factor': trial.suggest_float('direction_reward_factor', 0.1, 1.0),
        'direction_penalty_factor': trial.suggest_float('direction_penalty_factor', 0.1, 1.0),
        'length_reward_factor': trial.suggest_float('length_reward_factor', 0.01, 0.2),
        'max_length': trial.suggest_int('max_length', 30, 100)
    }
    
    # 创建环境和代理
    env = SnakeGame(reward_params=reward_params)
    state_dim = 18
    action_dim = 3
    
    agent = PPOAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=hidden_dim,
        lr=lr,
        gamma=gamma,
        epsilon=epsilon,
        epochs=epochs,
        batch_size=batch_size,
        gae_lambda=gae_lambda
    )
    
    # 训练参数
    n_games = 200  # 最大游戏数
    eval_interval = 10  # 评估间隔
    best_score = 0
    scores = []
    rewards = []
    avg_scores = []
    
    # 早停相关变量
    no_improvement_count = 0
    best_avg_score = float('-inf')
    
    # 训练循环
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
        avg_score = np.mean(scores[-early_stop_window:]) if len(scores) >= early_stop_window else np.mean(scores)
        avg_scores.append(avg_score)
        
        # 早停检查
        if i >= min_games:
            if avg_score > best_avg_score * (1 + early_stop_threshold):
                best_avg_score = avg_score
                no_improvement_count = 0
            else:
                no_improvement_count += 1
            
            if no_improvement_count >= patience:
                print(f"Early stopping at game {i} due to no improvement for {patience} evaluations")
                break
        
        # 定期评估
        if i % eval_interval == 0:
            # 保存训练指标
            metrics = {
                'game': i,
                'score': score,
                'avg_score': avg_score,
                'reward': episode_reward,
                'early_stop': {
                    'no_improvement_count': no_improvement_count,
                    'best_avg_score': best_avg_score
                },
                'params': {
                    'network': {
                        'hidden_dim': hidden_dim,
                        'lr': lr,
                        'gamma': gamma,
                        'epsilon': epsilon,
                        'epochs': epochs,
                        'batch_size': batch_size,
                        'gae_lambda': gae_lambda
                    },
                    'early_stop': {
                        'early_stop_window': early_stop_window,
                        'early_stop_threshold': early_stop_threshold,
                        'min_games': min_games,
                        'patience': patience
                    },
                    'reward': reward_params
                }
            }
            
            # 保存训练指标
            save_dir = 'hyperparameter_optimization'
            os.makedirs(save_dir, exist_ok=True)
            with open(f'{save_dir}/trial_{trial.number}_metrics.json', 'a') as f:
                json.dump(metrics, f)
                f.write('\n')
            
            # 绘制训练曲线
            plt.figure(figsize=(12, 5))
            plt.subplot(1, 2, 1)
            plt.plot(scores, label='Score', alpha=0.5)
            plt.plot(avg_scores, label='Average Score', color='red')
            plt.xlabel('Games')
            plt.ylabel('Score')
            plt.title('Learning Curve - Scores')
            plt.legend()
            
            plt.subplot(1, 2, 2)
            plt.plot(rewards, label='Reward', color='green', alpha=0.5)
            plt.xlabel('Games')
            plt.ylabel('Reward')
            plt.title('Learning Curve - Rewards')
            plt.legend()
            
            plt.tight_layout()
            plt.savefig(f'{save_dir}/trial_{trial.number}_learning_curve.png')
            plt.close()
            
            # 更新最佳分数
            if avg_score > best_score:
                best_score = avg_score
                # 保存最佳模型
                agent.save_model(f'{save_dir}/trial_{trial.number}_best_model.pth')
    
    return best_score

def optimize_hyperparameters(n_trials=50, n_jobs=1):
    # 创建保存目录
    save_dir = "hyperparameter_optimization"
    os.makedirs(save_dir, exist_ok=True)
    
    # 创建或加载study
    study_name = "snake_ppo_optimization"
    storage_name = f"sqlite:///{os.path.join(save_dir, study_name + '.db')}"
    
    try:
        study = optuna.load_study(study_name=study_name, storage=storage_name)
        print(f"Loading existing study from {storage_name}")
    except:
        study = optuna.create_study(
            study_name=study_name,
            storage=storage_name,
            direction="maximize"
        )
        print(f"Created new study: {study_name}")
    
    # 运行优化
    study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs)
    
    # 获取最佳参数
    best_params = study.best_params
    best_value = study.best_value
    
    print(f"\nBest trial:")
    print(f"  Value: {best_value}")
    print("  Params: ")
    for key, value in best_params.items():
        print(f"    {key}: {value}")
    
    # 保存最佳参数
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    best_params_path = os.path.join(save_dir, f'best_params_{timestamp}.json')
    
    # 分离网络参数和奖励参数
    network_params = {k: v for k, v in best_params.items() if k not in [
        'food_reward', 'death_penalty', 'distance_reward_factor', 
        'distance_penalty_factor', 'direction_reward_factor', 
        'direction_penalty_factor', 'length_reward_factor', 'max_length'
    ]}
    
    reward_params = {k: v for k, v in best_params.items() if k in [
        'food_reward', 'death_penalty', 'distance_reward_factor', 
        'distance_penalty_factor', 'direction_reward_factor', 
        'direction_penalty_factor', 'length_reward_factor', 'max_length'
    ]}
    
    best_params_dict = {
        'network_params': network_params,
        'reward_params': reward_params,
        'best_value': best_value,
        'timestamp': timestamp
    }
    
    with open(best_params_path, 'w') as f:
        json.dump(best_params_dict, f, indent=4)
    
    # 保存所有试验结果
    trials_path = os.path.join(save_dir, f'trials_{timestamp}.json')
    trials_data = []
    for trial in study.trials:
        if trial.state == optuna.trial.TrialState.COMPLETE:
            trials_data.append({
                'number': trial.number,
                'value': trial.value,
                'params': trial.params,
                'datetime': trial.datetime_complete.isoformat() if trial.datetime_complete else None
            })
    with open(trials_path, 'w') as f:
        json.dump(trials_data, f, indent=4)
    
    return best_params_dict

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_trials", type=int, default=50,
                      help="Number of trials for hyperparameter optimization")
    parser.add_argument("--n_jobs", type=int, default=1,
                      help="Number of parallel jobs")
    args = parser.parse_args()
    
    best_params = optimize_hyperparameters(n_trials=args.n_trials, n_jobs=args.n_jobs) 