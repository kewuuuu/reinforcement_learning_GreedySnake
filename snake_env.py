import numpy as np
import pygame
import random
from enum import Enum
from collections import namedtuple
from collections import deque
import os

class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4
    
Point = namedtuple('Point', 'x, y')

# RGB颜色定义
WHITE = (255, 255, 255)
RED = (200, 0, 0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0, 0, 0)

BLOCK_SIZE = 20
SPEED = 40

class SnakeGameAI:
    def __init__(self, w=640, h=480, render_ui=True):
        self.w = w
        self.h = h
        self.render_ui = render_ui
        
        # 初始化pygame
        if self.render_ui:
            pygame.init()
            self.display = pygame.display.set_mode((self.w, self.h))
            pygame.display.set_caption('Snake - Reinforcement Learning')
            self.clock = pygame.time.Clock()
            
            # 设置中文字体
            try:
                # 尝试使用系统中文字体
                if os.name == 'nt':  # Windows系统
                    self.font = pygame.font.SysFont('microsoftyaheimicrosoftyaheiui', 25)
                else:  # Linux/Mac系统
                    self.font = pygame.font.SysFont('notosanscjk', 25)
            except:
                # 如果找不到中文字体，使用默认字体
                self.font = pygame.font.SysFont(None, 25)
        
        self.reset()
    
    def reset(self):
        # 初始化游戏状态
        self.direction = Direction.RIGHT
        
        self.head = Point(self.w//2, self.h//2)
        self.snake = [
            self.head, 
            Point(self.head.x-BLOCK_SIZE, self.head.y),
            Point(self.head.x-(2*BLOCK_SIZE), self.head.y)
        ]
        
        self.score = 0
        self.food = None
        self._place_food()
        self.frame_iteration = 0
        self.steps_since_last_food = 0
        
        # 初始化位置历史记录
        self.position_history = deque(maxlen=20)  # 记录最近20步的位置
        self.last_positions = set()  # 用于检测循环
        
        return self._get_state()
    
    def _place_food(self):
        x = random.randint(0, (self.w-BLOCK_SIZE)//BLOCK_SIZE)*BLOCK_SIZE
        y = random.randint(0, (self.h-BLOCK_SIZE)//BLOCK_SIZE)*BLOCK_SIZE
        self.food = Point(x, y)
        if self.food in self.snake:
            self._place_food()
    
    def _get_state(self):
        head = self.snake[0]
        
        point_l = Point(head.x - BLOCK_SIZE, head.y)
        point_r = Point(head.x + BLOCK_SIZE, head.y)
        point_u = Point(head.x, head.y - BLOCK_SIZE)
        point_d = Point(head.x, head.y + BLOCK_SIZE)
        
        dir_l = self.direction == Direction.LEFT
        dir_r = self.direction == Direction.RIGHT
        dir_u = self.direction == Direction.UP
        dir_d = self.direction == Direction.DOWN
        
        state = [
            # 危险位置检测
            (dir_r and self._is_collision(point_r)) or 
            (dir_l and self._is_collision(point_l)) or 
            (dir_u and self._is_collision(point_u)) or 
            (dir_d and self._is_collision(point_d)),
            
            (dir_u and self._is_collision(point_r)) or 
            (dir_d and self._is_collision(point_l)) or 
            (dir_l and self._is_collision(point_u)) or 
            (dir_r and self._is_collision(point_d)),
            
            (dir_d and self._is_collision(point_r)) or 
            (dir_u and self._is_collision(point_l)) or 
            (dir_r and self._is_collision(point_u)) or 
            (dir_l and self._is_collision(point_d)),
            
            # 移动方向
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            
            # 食物相对位置
            self.food.x < self.head.x,  # 食物在左边
            self.food.x > self.head.x,  # 食物在右边
            self.food.y < self.head.y,  # 食物在上边
            self.food.y > self.head.y   # 食物在下边
        ]
        
        return np.array(state, dtype=int)
    
    def _is_collision(self, pt=None):
        if pt is None:
            pt = self.head
        # 撞墙
        if pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or pt.y < 0:
            return True
        # 撞到自己
        if pt in self.snake[1:]:
            return True
        return False
    
    def step(self, action):
        self.frame_iteration += 1
        self.steps_since_last_food += 1
        
        # 处理游戏退出
        if self.render_ui:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    quit()
        
        # 记录移动前的头部位置
        old_head = self.head
        
        # 移动
        self._move(action)
        self.snake.insert(0, self.head)
        
        # 检查游戏是否结束
        reward = 0
        game_over = False
        
        # 检测原地转圈
        current_pos = (self.head.x, self.head.y)
        self.position_history.append(current_pos)
        
        # 如果历史记录已满，检查是否出现循环
        if len(self.position_history) == self.position_history.maxlen:
            # 将当前位置添加到检测集合
            if current_pos in self.last_positions:
                # 如果当前位置在历史记录中出现过，说明在转圈
                game_over = True
                reward = -50  # 给予极大的惩罚
                return reward, game_over, self.score
            self.last_positions.add(current_pos)
        
        # 如果碰撞或者过长时间没吃到食物
        if self._is_collision() or self.frame_iteration > 100*len(self.snake):
            game_over = True
            reward = -10
            return reward, game_over, self.score
        
        # 计算到食物的距离变化
        old_distance = abs(old_head.x - self.food.x) + abs(old_head.y - self.food.y)
        new_distance = abs(self.head.x - self.food.x) + abs(self.head.y - self.food.y)
        distance_reward = 0.1 if new_distance < old_distance else -0.1
        
        # 吃到食物
        if self.head == self.food:
            self.score += 1
            reward = 10 + np.exp(-self.steps_since_last_food / 100)
            self.steps_since_last_food = 0
            self._place_food()
            # 重置位置历史记录
            self.position_history.clear()
            self.last_positions.clear()
        else:
            self.snake.pop()
            reward = distance_reward
        
        # 更新UI
        if self.render_ui:
            self._update_ui()
            self.clock.tick(SPEED)
        
        return reward, game_over, self.score
    
    def _move(self, action):
        # action: [直行, 右转, 左转]
        
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)
        
        if np.array_equal(action, [1, 0, 0]):
            # 直行
            new_dir = clock_wise[idx]
        elif np.array_equal(action, [0, 1, 0]):
            # 右转
            next_idx = (idx + 1) % 4
            new_dir = clock_wise[next_idx]
        else:  # [0, 0, 1]
            # 左转
            next_idx = (idx - 1) % 4
            new_dir = clock_wise[next_idx]
        
        self.direction = new_dir
        
        x = self.head.x
        y = self.head.y
        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE
        
        self.head = Point(x, y)
    
    def _update_ui(self):
        self.display.fill(BLACK)
        
        # 画蛇
        for pt in self.snake:
            pygame.draw.rect(self.display, BLUE1, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, BLUE2, pygame.Rect(pt.x+4, pt.y+4, 12, 12))
        
        # 画食物
        pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))
        
        # 显示分数和奖励
        score_text = self.font.render(f"Score: {self.score}", True, WHITE)
        self.display.blit(score_text, [0, 0])
        
        # 计算当前奖励（如果还没吃到食物，显示0）
        current_reward = np.exp(-self.steps_since_last_food / 100) if self.steps_since_last_food > 0 else 0
        reward_text = self.font.render(f"Current Reward: {current_reward:.3f}", True, WHITE)
        self.display.blit(reward_text, [0, 30])
        
        pygame.display.flip()
    
    def close(self):
        if self.render_ui:
            pygame.quit() 