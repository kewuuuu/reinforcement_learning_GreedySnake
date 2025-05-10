import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque

class DQNNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)

class DQNAgent:
    def __init__(self, state_size, action_size, hidden_size=256, learning_rate=0.001, gamma=0.99, 
                 epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995, memory_size=100000, 
                 batch_size=64, target_update=10, lr_decay=0.995):
        self.state_size = state_size
        self.action_size = action_size
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.gamma = gamma  # 折扣因子
        self.epsilon = epsilon  # 探索率
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update = target_update
        self.lr_decay = lr_decay  # 学习率衰减因子
        self.memory = ReplayMemory(memory_size)
        
        # 设备配置：如果有GPU则使用GPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 策略网络和目标网络
        self.policy_net = DQNNetwork(state_size, hidden_size, action_size).to(self.device)
        self.target_net = DQNNetwork(state_size, hidden_size, action_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # 评估模式，不计算梯度
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        
        self.learn_step_counter = 0
        self.best_loss = float('inf')  # 用于记录最佳损失
        
        # 训练历史数据
        self.training_history = {
            'episode_rewards': [],
            'episode_steps': [],
            'episode_losses': [],
            'episode_epsilons': [],
            'episode_learning_rates': [],
            'episode_q_values': [],
            'episode_avg_q_values': [],
            'episode_max_q_values': [],
            'episode_min_q_values': [],
            'episode_std_q_values': []
        }
        
        # 添加动作历史记录
        self.action_history = deque(maxlen=4)  # 记录最近4个动作
    
    def get_action(self, state):
        # 探索-利用策略
        if random.random() <= self.epsilon:
            # 探索：随机选择动作，但避免重复动作
            available_actions = list(range(self.action_size))
            if len(self.action_history) >= 2:
                # 避免连续选择相同的动作
                if self.action_history[-1] == self.action_history[-2]:
                    available_actions.remove(self.action_history[-1])
            action = random.choice(available_actions)
        else:
            # 利用：选择价值最大的动作
            state = torch.tensor(state, dtype=torch.float).to(self.device)
            with torch.no_grad():
                q_values = self.policy_net(state)
            action = q_values.argmax().item()
        
        # 记录动作
        self.action_history.append(action)
        return action
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)
    
    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        
        # 从记忆中随机抽取经验
        batch = self.memory.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.tensor(np.array(states), dtype=torch.float).to(self.device)
        actions = torch.tensor(np.array(actions), dtype=torch.long).to(self.device)
        rewards = torch.tensor(np.array(rewards), dtype=torch.float).to(self.device)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float).to(self.device)
        dones = torch.tensor(np.array(dones), dtype=torch.bool).to(self.device)
        
        # 当前Q值
        current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # 记录Q值统计信息
        with torch.no_grad():
            all_q_values = self.policy_net(states)
            self.training_history['episode_q_values'].append(all_q_values.mean().item())
            self.training_history['episode_avg_q_values'].append(all_q_values.mean().item())
            self.training_history['episode_max_q_values'].append(all_q_values.max().item())
            self.training_history['episode_min_q_values'].append(all_q_values.min().item())
            self.training_history['episode_std_q_values'].append(all_q_values.std().item())
        
        # 下一状态的Q值
        next_q_values = torch.zeros(self.batch_size, device=self.device)
        with torch.no_grad():
            next_q_values[~dones] = self.target_net(next_states[~dones]).max(1)[0]
        
        # 计算目标Q值
        target_q_values = rewards + (self.gamma * next_q_values)
        
        # 计算损失
        loss = self.criterion(current_q_values, target_q_values)
        
        # 记录损失
        self.training_history['episode_losses'].append(loss.item())
        
        # 优化模型
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)  # 梯度裁剪
        self.optimizer.step()
        
        # 更新目标网络
        self.learn_step_counter += 1
        if self.learn_step_counter % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        
        # 减小探索率
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
        # 学习率衰减
        for param_group in self.optimizer.param_groups:
            param_group['lr'] *= self.lr_decay
            self.training_history['episode_learning_rates'].append(param_group['lr'])
            
        # 更新最佳损失
        if loss.item() < self.best_loss:
            self.best_loss = loss.item()
    
    def save(self, file_path, training_stats=None):
        save_dict = {
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'epsilon_min': self.epsilon_min,
            'epsilon_decay': self.epsilon_decay,
            'gamma': self.gamma,
            'learning_rate': self.learning_rate,
            'lr_decay': self.lr_decay,
            'best_loss': self.best_loss,
            'learn_step_counter': self.learn_step_counter,
            'training_history': self.training_history
        }
        
        if training_stats is not None:
            save_dict['training_stats'] = training_stats
            
        torch.save(save_dict, file_path)
    
    def load(self, file_path):
        checkpoint = torch.load(file_path)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon'] 
        self.epsilon_min = checkpoint['epsilon_min']
        self.epsilon_decay = checkpoint['epsilon_decay']
        self.gamma = checkpoint['gamma']
        self.learning_rate = checkpoint['learning_rate']
        self.lr_decay = checkpoint['lr_decay']
        self.best_loss = checkpoint['best_loss']
        self.learn_step_counter = checkpoint['learn_step_counter']
        if 'training_history' in checkpoint:
            self.training_history = checkpoint['training_history'] 