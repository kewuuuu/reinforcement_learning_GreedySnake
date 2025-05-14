import pygame
import torch
from snake_env import SnakeGame
from ppo_agent import PPOAgent
import argparse

def play_game(model_path, render=True):
    # 初始化环境
    env = SnakeGame(render=render)
    state_dim = 18
    action_dim = 3
    
    # 创建agent并加载模型
    agent = PPOAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=98,  # 修改为与保存模型相同的隐藏层维度
        lr=0.0003,
        gamma=0.99,
        epsilon=0.2,
        epochs=10,
        batch_size=128,
        gae_lambda=0.95
    )
    
    # 加载模型
    print(f"Attempting to load model from: {model_path}")
    try:
        # 尝试直接加载模型
        print("Trying direct model loading...")
        agent.load(model_path)
        print("Successfully loaded model using direct loading")
    except Exception as e:
        print(f"Direct loading failed: {str(e)}")
        try:
            # 尝试加载checkpoint格式
            print("Trying checkpoint loading...")
            agent.load_model(model_path)
            print("Successfully loaded model using checkpoint loading")
        except Exception as e:
            print(f"Checkpoint loading failed: {str(e)}")
            print("Error: Could not load model. Please check the model format.")
            return
    
    print(f"Model loaded successfully from {model_path}")
    
    # 运行游戏
    state = env.reset()
    done = False
    total_reward = 0
    score = 0
    
    while not done:
        # 选择动作
        action = agent.choose_action(state)
        
        # 执行动作
        reward, done, score = env.step(action)
        next_state = env._get_state()
        
        # 更新状态和奖励
        state = next_state
        total_reward += reward
    
    print(f"\nGame Over!")
    print(f"Final Score: {score}")
    print(f"Total Reward: {total_reward:.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="models/20250511_152600/ppo_model.pth",
                      help="Path to the trained model")
    parser.add_argument("--fast", action="store_true",
                      help="Run without rendering")
    args = parser.parse_args()
    
    play_game(args.model, render=not args.fast) 