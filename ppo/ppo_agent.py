import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions import Categorical
import os

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(ActorCritic, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )
        
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, state):
        action_probs = self.actor(state)
        state_value = self.critic(state)
        return action_probs, state_value

class PPOAgent:
    def __init__(self, state_dim, action_dim, hidden_dim, lr, gamma, epsilon, epochs, batch_size, gae_lambda):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[PPOAgent] Using device: {self.device}")
        self.actor_critic = ActorCritic(state_dim, action_dim, hidden_dim).to(self.device)
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=lr)
        
        self.gamma = gamma
        self.epsilon = epsilon
        self.epochs = epochs
        self.batch_size = batch_size
        self.gae_lambda = gae_lambda
        
        self.memory = []
        self.state_dim = state_dim
        self.action_dim = action_dim
    
    def to(self, device):
        self.device = device
        self.actor_critic = self.actor_critic.to(device)
        return self
    
    def choose_action(self, state):
        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state).to(self.device)
        elif isinstance(state, torch.Tensor):
            state = state.to(self.device)
        if state.dim() == 1:
            state = state.unsqueeze(0)  # 保证batch维度
        with torch.no_grad():
            action_probs, _ = self.actor_critic(state)
            dist = Categorical(action_probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
        return action.item()
    
    def store_transition(self, state, action, reward, next_state, done):
        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state).to(self.device)
        elif isinstance(state, torch.Tensor):
            state = state.to(self.device)
        if isinstance(next_state, np.ndarray):
            next_state = torch.FloatTensor(next_state).to(self.device)
        elif isinstance(next_state, torch.Tensor):
            next_state = next_state.to(self.device)
        with torch.no_grad():
            _, value = self.actor_critic(state.unsqueeze(0) if state.dim()==1 else state)
            _, next_value = self.actor_critic(next_state.unsqueeze(0) if next_state.dim()==1 else next_state)
        self.memory.append((state, action, reward, next_state, done, value.item(), next_value.item()))
    
    def compute_gae(self, rewards, values, next_values, dones):
        advantages = []
        gae = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = next_values[t]
            else:
                next_value = values[t + 1]
            
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages.insert(0, gae)
        
        return torch.tensor(advantages, dtype=torch.float32).to(self.device)
    
    def learn(self):
        if len(self.memory) < self.batch_size:
            return
        
        # 准备数据
        states = torch.stack([x[0] for x in self.memory[-self.batch_size:]]).to(self.device)
        actions = torch.tensor([x[1] for x in self.memory[-self.batch_size:]], dtype=torch.long, device=self.device)
        rewards = torch.tensor([x[2] for x in self.memory[-self.batch_size:]], dtype=torch.float32, device=self.device)
        next_states = torch.stack([x[3] for x in self.memory[-self.batch_size:]]).to(self.device)
        dones = torch.tensor([x[4] for x in self.memory[-self.batch_size:]], dtype=torch.float32, device=self.device)
        values = torch.tensor([x[5] for x in self.memory[-self.batch_size:]], dtype=torch.float32, device=self.device)
        next_values = torch.tensor([x[6] for x in self.memory[-self.batch_size:]], dtype=torch.float32, device=self.device)
        
        # 计算优势
        advantages = self.compute_gae(rewards, values, next_values, dones)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # 多轮更新
        for _ in range(self.epochs):
            # 获取动作概率和状态值
            action_probs, state_values = self.actor_critic(states)
            dist = Categorical(action_probs)
            new_log_probs = dist.log_prob(actions)
            entropy = dist.entropy().mean()
            
            # 计算比率
            ratio = torch.exp(new_log_probs)
            
            # 计算PPO目标
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            
            # 计算价值损失
            critic_loss = 0.5 * (state_values.squeeze() - (rewards + self.gamma * next_values * (1 - dones))).pow(2).mean()
            
            # 总损失
            loss = actor_loss + critic_loss - 0.01 * entropy
            
            # 优化
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        
        # 清空记忆
        self.memory = []
    
    def save_model(self, path):
        torch.save({
            'actor_critic_state_dict': self.actor_critic.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, path)
    
    def load_model(self, path):
        """加载模型（checkpoint格式）"""
        print(f"Loading checkpoint from: {path}")
        checkpoint = torch.load(path, map_location=self.device)
        print(f"Checkpoint keys: {checkpoint.keys() if isinstance(checkpoint, dict) else 'Not a dict'}")
        
        if isinstance(checkpoint, dict):
            if 'actor_critic_state_dict' in checkpoint:
                self.actor_critic.load_state_dict(checkpoint['actor_critic_state_dict'])
                if 'optimizer_state_dict' in checkpoint:
                    self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                return
            elif 'state_dict' in checkpoint:
                self.actor_critic.load_state_dict(checkpoint['state_dict'])
                return
        
        raise ValueError(f"Invalid model format. Expected dict with 'actor_critic_state_dict' or 'state_dict' key")
    
    def save(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
        torch.save(self.actor_critic.state_dict(), os.path.join(path, 'ppo_model.pth'))
    
    def load(self, path):
        """直接加载模型状态"""
        print(f"Loading model state from: {path}")
        state_dict = torch.load(path, map_location=self.device)
        print(f"State dict keys: {state_dict.keys() if isinstance(state_dict, dict) else 'Not a dict'}")
        
        if isinstance(state_dict, dict):
            self.actor_critic.load_state_dict(state_dict)
        else:
            raise ValueError("Invalid model format. Expected state dict")
    
    def share_memory(self):
        """共享模型参数，用于多进程训练"""
        self.actor_critic.share_memory()
        return self 