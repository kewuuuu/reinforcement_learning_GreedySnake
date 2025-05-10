import torch
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
from dqn_agent import DQNAgent

def visualize_training_data(model_path, save_dir='visualization_results'):
    """
    可视化训练数据
    """
    # 创建保存目录
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # 加载模型和数据
    checkpoint = torch.load(model_path)
    training_history = checkpoint['training_history']
    
    # 创建图表
    plt.style.use('seaborn')
    
    # 1. 奖励和步数
    plt.figure(figsize=(15, 10))
    plt.subplot(2, 1, 1)
    plt.plot(training_history['episode_rewards'], label='奖励', alpha=0.6)
    plt.title('训练过程中的奖励变化')
    plt.xlabel('回合')
    plt.ylabel('奖励')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 1, 2)
    plt.plot(training_history['episode_steps'], label='步数', alpha=0.6)
    plt.title('训练过程中的步数变化')
    plt.xlabel('回合')
    plt.ylabel('步数')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/rewards_and_steps.png')
    plt.close()
    
    # 2. 损失和Q值
    plt.figure(figsize=(15, 10))
    plt.subplot(2, 1, 1)
    plt.plot(training_history['episode_losses'], label='损失', alpha=0.6)
    plt.title('训练过程中的损失变化')
    plt.xlabel('回合')
    plt.ylabel('损失')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 1, 2)
    plt.plot(training_history['episode_avg_q_values'], label='平均Q值', alpha=0.6)
    plt.plot(training_history['episode_max_q_values'], label='最大Q值', alpha=0.6)
    plt.plot(training_history['episode_min_q_values'], label='最小Q值', alpha=0.6)
    plt.title('训练过程中的Q值变化')
    plt.xlabel('回合')
    plt.ylabel('Q值')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/loss_and_q_values.png')
    plt.close()
    
    # 3. 探索率和学习率
    plt.figure(figsize=(15, 10))
    plt.subplot(2, 1, 1)
    plt.plot(training_history['episode_epsilons'], label='探索率', alpha=0.6)
    plt.title('训练过程中的探索率变化')
    plt.xlabel('回合')
    plt.ylabel('探索率')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 1, 2)
    plt.plot(training_history['episode_learning_rates'], label='学习率', alpha=0.6)
    plt.title('训练过程中的学习率变化')
    plt.xlabel('回合')
    plt.ylabel('学习率')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/epsilon_and_learning_rate.png')
    plt.close()
    
    # 4. Q值分布统计
    plt.figure(figsize=(15, 10))
    plt.subplot(2, 1, 1)
    plt.plot(training_history['episode_std_q_values'], label='Q值标准差', alpha=0.6)
    plt.title('训练过程中的Q值标准差变化')
    plt.xlabel('回合')
    plt.ylabel('标准差')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 1, 2)
    # 计算移动平均
    window_size = 100
    moving_avg = np.convolve(training_history['episode_avg_q_values'], 
                            np.ones(window_size)/window_size, 
                            mode='valid')
    plt.plot(moving_avg, label=f'{window_size}轮移动平均Q值', alpha=0.6)
    plt.title('Q值的移动平均')
    plt.xlabel('回合')
    plt.ylabel('Q值')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/q_value_statistics.png')
    plt.close()
    
    # 5. 综合性能指标
    plt.figure(figsize=(15, 10))
    
    # 计算奖励的移动平均
    reward_moving_avg = np.convolve(training_history['episode_rewards'], 
                                  np.ones(window_size)/window_size, 
                                  mode='valid')
    
    plt.subplot(2, 1, 1)
    plt.plot(reward_moving_avg, label=f'{window_size}轮移动平均奖励', alpha=0.6)
    plt.title('奖励的移动平均')
    plt.xlabel('回合')
    plt.ylabel('奖励')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 1, 2)
    plt.plot(training_history['episode_avg_q_values'], label='平均Q值', alpha=0.6)
    plt.plot(reward_moving_avg, label=f'{window_size}轮移动平均奖励', alpha=0.6)
    plt.title('Q值与奖励的关系')
    plt.xlabel('回合')
    plt.ylabel('值')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/performance_metrics.png')
    plt.close()
    
    print(f"可视化结果已保存到 {save_dir} 目录")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='训练数据可视化工具')
    parser.add_argument('--model', type=str, required=True,
                      help='要分析的模型文件路径')
    parser.add_argument('--save_dir', type=str, default='visualization_results',
                      help='保存可视化结果的目录')
    
    args = parser.parse_args()
    visualize_training_data(args.model, args.save_dir) 