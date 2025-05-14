import os
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import torch

from snake_env import SnakeGameAI, BLOCK_SIZE
from dqn_agent import DQNAgent

def train(episodes=2000, max_steps=10000, plot_interval=100, save_interval=100, 
          render_training=False, start_model=None, model_prefix="snake"):
    print("开始训练贪吃蛇AI...")
    
    # 创建目录保存模型和结果
    if not os.path.exists('models'):
        os.makedirs('models')
    if not os.path.exists('results'):
        os.makedirs('results')
    
    # 创建环境和代理
    env = SnakeGameAI(render_ui=render_training)
    # 计算新的状态空间大小
    map_width = env.w // BLOCK_SIZE
    map_height = env.h // BLOCK_SIZE
    state_size = map_width * map_height + 3  # 地图状态 + 上一步操作
    action_size = 3  # [直行, 右转, 左转]
    
    # 使用更大的隐藏层大小和更快的训练参数
    agent = DQNAgent(
        state_size, 
        action_size, 
        hidden_size=512,
        learning_rate=0.001,  # 增加学习率
        epsilon_decay=0.997,  # 加快探索率衰减
        batch_size=256,  # 增加批次大小
        target_update=5  # 更频繁地更新目标网络
    )
    
    # 训练数据记录
    scores = []
    avg_scores = []
    steps_per_episode = []
    avg_steps = []
    recent_scores = deque(maxlen=100)
    recent_steps = deque(maxlen=100)
    
    # 记录最大分数和对应模型
    best_score = 0
    best_avg_score = 0
    best_loss = float('inf')
    
    # 如果指定了起始模型，则加载它
    if start_model:
        if os.path.exists(start_model):
            print(f"从模型 {start_model} 继续训练")
            checkpoint = torch.load(start_model)
            agent.load(start_model)
            
            # 恢复训练统计信息
            if 'training_stats' in checkpoint:
                stats = checkpoint['training_stats']
                best_score = stats.get('best_score', 0)
                best_avg_score = stats.get('best_avg_score', 0)
                best_loss = stats.get('best_loss', float('inf'))
                scores = stats.get('scores', [])
                avg_scores = stats.get('avg_scores', [])
                steps_per_episode = stats.get('steps_per_episode', [])
                avg_steps = stats.get('avg_steps', [])
                recent_scores = deque(stats.get('recent_scores', []), maxlen=100)
                recent_steps = deque(stats.get('recent_steps', []), maxlen=100)
                print(f"已恢复训练统计信息 - 最高分: {best_score}, 最高平均分: {best_avg_score:.2f}")
        else:
            print(f"警告：未找到模型 {start_model}，将从头开始训练")
    
    for episode in range(1, episodes+1):
        state = env.reset()
        done = False
        score = 0
        steps = 0
        
        for step in range(max_steps):
            steps += 1
            
            # 选择动作
            action_idx = agent.get_action(state)
            # 将索引转换为one-hot动作
            action = [0, 0, 0]
            action[action_idx] = 1
            
            # 执行动作
            reward, done, score = env.step(action)
            next_state = env._get_state()
            
            # 记录经验
            agent.remember(state, action_idx, reward, next_state, done)
            
            # 学习
            agent.replay()
            
            state = next_state
            
            if done:
                break
        
        # 记录这一轮的结果
        scores.append(score)
        steps_per_episode.append(steps)
        recent_scores.append(score)
        recent_steps.append(steps)
        
        # 记录训练历史数据
        agent.training_history['episode_rewards'].append(score)
        agent.training_history['episode_steps'].append(steps)
        agent.training_history['episode_epsilons'].append(agent.epsilon)
        
        avg_recent_score = np.mean(recent_scores)
        avg_recent_steps = np.mean(recent_steps)
        avg_scores.append(avg_recent_score)
        avg_steps.append(avg_recent_steps)
        
        # 打印训练进度
        if episode % plot_interval == 0:
            print(f"Episode {episode}/{episodes}, Score: {score}, Avg Score: {avg_recent_score:.2f}, Steps: {steps}, Avg Steps: {avg_recent_steps:.2f}, Epsilon: {agent.epsilon:.4f}, Loss: {agent.best_loss:.4f}")
        
        # 保存最好的模型
        if score > best_score:
            best_score = score
            # 保存模型和训练统计信息
            training_stats = {
                'best_score': best_score,
                'best_avg_score': best_avg_score,
                'best_loss': best_loss,
                'scores': scores,
                'avg_scores': avg_scores,
                'steps_per_episode': steps_per_episode,
                'avg_steps': avg_steps,
                'recent_scores': list(recent_scores),
                'recent_steps': list(recent_steps)
            }
            agent.save(f"models/{model_prefix}_best_score.pth", training_stats)
            print(f"新的最高分！分数: {best_score}，已保存模型 models/{model_prefix}_best_score.pth")
        
        if avg_recent_score > best_avg_score and len(recent_scores) == 100:
            best_avg_score = avg_recent_score
            training_stats = {
                'best_score': best_score,
                'best_avg_score': best_avg_score,
                'best_loss': best_loss,
                'scores': scores,
                'avg_scores': avg_scores,
                'steps_per_episode': steps_per_episode,
                'avg_steps': avg_steps,
                'recent_scores': list(recent_scores),
                'recent_steps': list(recent_steps)
            }
            agent.save(f"models/{model_prefix}_best_avg.pth", training_stats)
            print(f"新的最高平均分！平均分: {best_avg_score:.2f}，已保存模型 models/{model_prefix}_best_avg.pth")
        
        if agent.best_loss < best_loss:
            best_loss = agent.best_loss
            training_stats = {
                'best_score': best_score,
                'best_avg_score': best_avg_score,
                'best_loss': best_loss,
                'scores': scores,
                'avg_scores': avg_scores,
                'steps_per_episode': steps_per_episode,
                'avg_steps': avg_steps,
                'recent_scores': list(recent_scores),
                'recent_steps': list(recent_steps)
            }
            agent.save(f"models/{model_prefix}_best_loss.pth", training_stats)
            print(f"新的最低损失！损失: {best_loss:.4f}，已保存模型 models/{model_prefix}_best_loss.pth")
        
        # 定期保存模型和绘制结果
        if episode % save_interval == 0:
            training_stats = {
                'best_score': best_score,
                'best_avg_score': best_avg_score,
                'best_loss': best_loss,
                'scores': scores,
                'avg_scores': avg_scores,
                'steps_per_episode': steps_per_episode,
                'avg_steps': avg_steps,
                'recent_scores': list(recent_scores),
                'recent_steps': list(recent_steps)
            }
            agent.save(f"models/{model_prefix}_episode_{episode}.pth", training_stats)
            agent.save(f"models/{model_prefix}_latest.pth", training_stats)
            plot_results(episode, scores, avg_scores, steps_per_episode, avg_steps, model_prefix)
    
    # 训练结束，保存最终模型
    training_stats = {
        'best_score': best_score,
        'best_avg_score': best_avg_score,
        'best_loss': best_loss,
        'scores': scores,
        'avg_scores': avg_scores,
        'steps_per_episode': steps_per_episode,
        'avg_steps': avg_steps,
        'recent_scores': list(recent_scores),
        'recent_steps': list(recent_steps)
    }
    agent.save(f"models/{model_prefix}_final.pth", training_stats)
    
    # 绘制最终结果
    plot_results(episodes, scores, avg_scores, steps_per_episode, avg_steps, model_prefix)
    
    print("训练完成！")
    return agent

def plot_results(episode, scores, avg_scores, steps, avg_steps, model_prefix="snake"):
    # 创建图表
    plt.figure(figsize=(15, 10))
    
    # 绘制分数
    plt.subplot(2, 1, 1)
    plt.plot(scores, alpha=0.6, label='分数')
    plt.plot(avg_scores, label='平均分数 (100轮)')
    plt.title(f'贪吃蛇AI训练 - {model_prefix} - {episode}轮后')
    plt.xlabel('轮次')
    plt.ylabel('分数')
    plt.legend()
    plt.grid(True)
    
    # 绘制步数
    plt.subplot(2, 1, 2)
    plt.plot(steps, alpha=0.6, label='步数')
    plt.plot(avg_steps, label='平均步数 (100轮)')
    plt.xlabel('轮次')
    plt.ylabel('步数')
    plt.legend()
    plt.grid(True)
    
    # 保存图表
    plt.tight_layout()
    plt.savefig(f"results/{model_prefix}_training_results_episode_{episode}.png")
    plt.close()

def test(model_path, episodes=5, render=True, max_steps=10000):
    print(f"测试模型: {model_path}")
    
    env = SnakeGameAI(render_ui=render)
    map_width = env.w // BLOCK_SIZE
    map_height = env.h // BLOCK_SIZE
    state_size = map_width * map_height + 3  # 地图状态 + 上一步操作
    action_size = 3
    agent = DQNAgent(state_size, action_size)
    
    # 加载模型
    agent.load(model_path)
    agent.epsilon = 0.0  # 测试时不使用探索
    
    scores = []
    steps_list = []
    
    for episode in range(1, episodes+1):
        state = env.reset()
        done = False
        score = 0
        steps = 0
        
        for step in range(max_steps):
            steps += 1
            
            # 选择动作
            action_idx = agent.get_action(state)
            action = [0, 0, 0]
            action[action_idx] = 1
            
            # 执行动作
            reward, done, score = env.step(action)
            next_state = env._get_state()
            state = next_state
            
            if done:
                break
        
        scores.append(score)
        steps_list.append(steps)
        print(f"测试 {episode}/{episodes}, 分数: {score}, 步数: {steps}")
    
    avg_score = np.mean(scores)
    avg_steps = np.mean(steps_list)
    print(f"测试结果 - 平均分数: {avg_score:.2f}, 平均步数: {avg_steps:.2f}")
    
    return scores, steps_list

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='贪吃蛇强化学习训练程序')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'], 
                        help='运行模式: train或test')
    parser.add_argument('--episodes', type=int, default=2000, 
                        help='训练的轮数')
    parser.add_argument('--render', action='store_true', 
                        help='显示游戏界面')
    parser.add_argument('--model', type=str, default='models/snake_final.pth', 
                        help='测试模式下使用的模型路径，或训练模式下的起始模型路径')
    parser.add_argument('--continue_training', action='store_true', 
                        help='是否从之前的模型继续训练')
    parser.add_argument('--model_prefix', type=str, default='snake', 
                        help='保存模型的名称前缀')
    parser.add_argument('--fast', action='store_true',
                        help='使用快速训练模式（不显示渲染）')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        start_model = None
        if args.continue_training:
            # 如果选择继续训练，使用指定的模型或最新保存的模型
            if args.model and os.path.exists(args.model):
                start_model = args.model
            elif os.path.exists(f"models/{args.model_prefix}_latest.pth"):
                start_model = f"models/{args.model_prefix}_latest.pth"
            elif os.path.exists(f"models/{args.model_prefix}_final.pth"):
                start_model = f"models/{args.model_prefix}_final.pth"
            
            if start_model:
                print(f"将从模型 {start_model} 继续训练")
            else:
                print("未找到可用的模型，将从头开始训练")
        
        # 如果使用快速模式，强制关闭渲染
        render_training = args.render and not args.fast
        
        agent = train(episodes=args.episodes, render_training=render_training, 
                      start_model=start_model, model_prefix=args.model_prefix)
    else:
        test(args.model, render=True) 