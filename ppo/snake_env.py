import pygame
import numpy as np
import random
from enum import Enum
from collections import namedtuple
import torch

class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

Point = namedtuple('Point', 'x, y')

class SnakeGame:
    def __init__(self, w=640, h=480, render=False, device='cpu', reward_params=None):
        self.w = w
        self.h = h
        self.block_size = 20
        self.render = render
        self.device = device
        
        # 设置默认奖励参数
        self.reward_params = {
            'food_reward': 50.0,  # 吃到食物的奖励
            'death_penalty': -10.0,  # 死亡的惩罚
            'distance_reward_factor': 1.0,  # 距离奖励因子
            'distance_penalty_factor': 0.5,  # 距离惩罚因子
            'direction_reward_factor': 0.5,  # 方向奖励因子
            'direction_penalty_factor': 0.3,  # 方向惩罚因子
            'length_reward_factor': 0.1,  # 长度奖励因子
            'max_length': 50,  # 最大长度
        }
        
        # 更新奖励参数
        if reward_params:
            self.reward_params.update(reward_params)
        
        # Initialize display only if rendering is enabled
        if self.render:
            pygame.init()
            self.display = pygame.display.set_mode((self.w, self.h))
            pygame.display.set_caption('Snake Game - PPO Training')
            self.clock = pygame.time.Clock()
            self.font = pygame.font.Font(None, 36)
        
        # Initialize game state
        self.reset()
        
    def reset(self):
        # Initialize snake
        self.direction = Direction.RIGHT
        self.head = Point(self.w//2, self.h//2)
        self.snake = [self.head,
                     Point(self.head.x-self.block_size, self.head.y),
                     Point(self.head.x-(2*self.block_size), self.head.y)]
        
        self.score = 0
        self.food = None
        self._place_food()
        self.frame_iteration = 0
        self.game_over = False
        self.last_distance = self._get_distance_to_food()
        self.current_reward = 0
        
        return self._get_state()
    
    def _get_distance_to_food(self):
        return abs(self.head.x - self.food.x) + abs(self.head.y - self.food.y)
    
    def _place_food(self):
        x = random.randint(0, (self.w-self.block_size)//self.block_size)*self.block_size
        y = random.randint(0, (self.h-self.block_size)//self.block_size)*self.block_size
        self.food = Point(x, y)
        if self.food in self.snake:
            self._place_food()
    
    def _get_state(self):
        head = self.snake[0]
        
        point_l = Point(head.x - self.block_size, head.y)
        point_r = Point(head.x + self.block_size, head.y)
        point_u = Point(head.x, head.y - self.block_size)
        point_d = Point(head.x, head.y + self.block_size)
        
        dir_l = self.direction == Direction.LEFT
        dir_r = self.direction == Direction.RIGHT
        dir_u = self.direction == Direction.UP
        dir_d = self.direction == Direction.DOWN
        
        # 计算到边界的距离
        dist_to_left = head.x
        dist_to_right = self.w - head.x - self.block_size
        dist_to_up = head.y
        dist_to_down = self.h - head.y - self.block_size
        
        # 计算到食物的相对距离
        food_dist_x = self.food.x - head.x
        food_dist_y = self.food.y - head.y
        
        # 计算到食物的方向
        food_left = food_dist_x < 0
        food_right = food_dist_x > 0
        food_up = food_dist_y < 0
        food_down = food_dist_y > 0
        
        state = [
            # 1-3: 危险检测（3个特征）
            # 1. 前方是否有危险
            (dir_r and self._is_collision(point_r)) or 
            (dir_l and self._is_collision(point_l)) or 
            (dir_u and self._is_collision(point_u)) or 
            (dir_d and self._is_collision(point_d)),
            
            # 2. 右方是否有危险
            (dir_u and self._is_collision(point_r)) or 
            (dir_d and self._is_collision(point_l)) or 
            (dir_l and self._is_collision(point_u)) or 
            (dir_r and self._is_collision(point_d)),
            
            # 3. 左方是否有危险
            (dir_d and self._is_collision(point_r)) or 
            (dir_u and self._is_collision(point_l)) or 
            (dir_r and self._is_collision(point_u)) or 
            (dir_l and self._is_collision(point_d)),
            
            # 4-7: 当前移动方向（4个特征）
            dir_l,  # 是否向左
            dir_r,  # 是否向右
            dir_u,  # 是否向上
            dir_d,  # 是否向下
            
            # 8-11: 食物相对位置（4个特征）
            food_left,  # 食物在左边
            food_right,  # 食物在右边
            food_up,  # 食物在上边
            food_down,  # 食物在下边
            
            # 12-15: 到边界的距离（归一化到0-1之间）
            dist_to_left / self.w,
            dist_to_right / self.w,
            dist_to_up / self.h,
            dist_to_down / self.h,
            
            # 16-17: 到食物的相对距离（归一化到-1到1之间）
            food_dist_x / self.w,
            food_dist_y / self.h,
            
            # 18: 蛇的长度（归一化）
            len(self.snake) / 20.0  # 假设最大长度为20
        ]
        
        # 转换为tensor并移动到指定设备
        return torch.FloatTensor(state).to(self.device)
    
    def _is_collision(self, pt=None):
        if pt is None:
            pt = self.head
        # Hits boundary
        if pt.x > self.w - self.block_size or pt.x < 0 or pt.y > self.h - self.block_size or pt.y < 0:
            return True
        # Hits itself
        if pt in self.snake[1:]:
            return True
        return False
    
    def _move(self, action):
        # [straight, right, left]
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)
        
        if action == 0:  # straight
            new_dir = clock_wise[idx]  # no change
        elif action == 1:  # right turn
            next_idx = (idx + 1) % 4
            new_dir = clock_wise[next_idx]
        else:  # left turn
            next_idx = (idx - 1) % 4
            new_dir = clock_wise[next_idx]
            
        self.direction = new_dir
        
        x = self.head.x
        y = self.head.y
        if self.direction == Direction.RIGHT:
            x += self.block_size
        elif self.direction == Direction.LEFT:
            x -= self.block_size
        elif self.direction == Direction.DOWN:
            y += self.block_size
        elif self.direction == Direction.UP:
            y -= self.block_size
            
        self.head = Point(x, y)

    def step(self, action):
        self.frame_iteration += 1
        
        # 1. Collect user input only if rendering
        if self.render:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    quit()
        
        # 2. Move
        self._move(action)
        self.snake.insert(0, self.head)
        
        # 3. Check if game over
        reward = 0
        if self._is_collision() or self.frame_iteration > 100*len(self.snake):
            self.game_over = True
            reward = self.reward_params['death_penalty']
            self.current_reward = reward
            return reward, self.game_over, self.score
        
        # 4. Place new food or just move
        if self.head == self.food:
            self.score += 1
            reward = self.reward_params['food_reward']
            self._place_food()
            self.last_distance = self._get_distance_to_food()
        else:
            self.snake.pop()
            # Calculate distance-based reward with length-dependent scaling
            current_distance = self._get_distance_to_food()
            distance_change = self.last_distance - current_distance
            
            # 根据蛇的长度动态调整奖励
            length_factor = 1.0 + (len(self.snake) - 3) * self.reward_params['length_reward_factor']
            length_factor = min(length_factor, self.reward_params['max_length'] / 3.0)
            
            # 基础距离奖励
            if distance_change > 0:  # 距离缩短
                reward = self.reward_params['distance_reward_factor'] * length_factor
            elif distance_change < 0:  # 距离增加
                reward = -self.reward_params['distance_penalty_factor'] * length_factor
            
            # 添加朝向食物的额外奖励
            dir_l = self.direction == Direction.LEFT
            dir_r = self.direction == Direction.RIGHT
            dir_u = self.direction == Direction.UP
            dir_d = self.direction == Direction.DOWN
            
            food_left = self.food.x < self.head.x
            food_right = self.food.x > self.head.x
            food_up = self.food.y < self.head.y
            food_down = self.food.y > self.head.y
            
            # 如果蛇头朝向食物方向，给予额外奖励
            if (dir_r and food_right) or (dir_l and food_left) or (dir_u and food_up) or (dir_d and food_down):
                reward += self.reward_params['direction_reward_factor'] * length_factor
            
            # 如果蛇头远离食物方向，给予更大的惩罚
            elif (dir_r and food_left) or (dir_l and food_right) or (dir_u and food_down) or (dir_d and food_up):
                reward -= self.reward_params['direction_penalty_factor'] * length_factor
            
            self.last_distance = current_distance
        
        self.current_reward = reward
        
        # 5. Update UI and clock only if rendering
        if self.render:
            self._update_ui()
            self.clock.tick(30)
        
        # 6. Return game over and score
        return reward, self.game_over, self.score
    
    def _update_ui(self):
        self.display.fill((0,0,0))
        
        for pt in self.snake:
            pygame.draw.rect(self.display, (0,255,0), pygame.Rect(pt.x, pt.y, self.block_size, self.block_size))
            pygame.draw.rect(self.display, (0,200,0), pygame.Rect(pt.x+4, pt.y+4, 12, 12))
        
        pygame.draw.rect(self.display, (255,0,0), pygame.Rect(self.food.x, self.food.y, self.block_size, self.block_size))
        
        # 显示分数和当前奖励
        score_text = self.font.render(f"Score: {self.score}", True, (255,255,255))
        reward_text = self.font.render(f"Reward: {self.current_reward:.2f}", True, (255,255,255))
        
        self.display.blit(score_text, [0, 0])
        self.display.blit(reward_text, [0, 40])  # 在分数下方显示奖励
        pygame.display.flip() 