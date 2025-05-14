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
SPEED = 1000  # 增加游戏速度

class SnakeGameAI:
    def __init__(self, w=640, h=480, render_ui=True):
        self.w = w
        self.h = h
        self.render_ui = render_ui
        
        # 默认奖励参数
        self.base_reward = 30
        self.step_bonus_multiplier = 20
        self.step_bonus_decay = 50
        self.length_bonus = 2
        self.death_penalty = -20
        self.distance_reward = 0.2
        
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
        map_width = self.w // BLOCK_SIZE
        map_height = self.h // BLOCK_SIZE
        x = random.randint(0, map_width - 1) * BLOCK_SIZE
        y = random.randint(0, map_height - 1) * BLOCK_SIZE
        self.food = Point(x, y)
        if self.food in self.snake:
            self._place_food()
    
    def get_state(self):
        """获取当前状态的公共接口"""
        return self._get_state()
    
    def _get_state(self):
        # 创建地图状态矩阵 (w//BLOCK_SIZE) x (h//BLOCK_SIZE)
        map_width = self.w // BLOCK_SIZE
        map_height = self.h // BLOCK_SIZE
        state_matrix = np.zeros((map_height, map_width), dtype=int)
        
        # 标记蛇身位置（值为2）
        for segment in self.snake:
            x = min(max(0, segment.x // BLOCK_SIZE), map_width - 1)
            y = min(max(0, segment.y // BLOCK_SIZE), map_height - 1)
            state_matrix[y, x] = 2
        
        # 标记蛇头位置（值为3）
        head_x = min(max(0, self.head.x // BLOCK_SIZE), map_width - 1)
        head_y = min(max(0, self.head.y // BLOCK_SIZE), map_height - 1)
        state_matrix[head_y, head_x] = 3
        
        # 标记食物位置（值为1）
        food_x = min(max(0, self.food.x // BLOCK_SIZE), map_width - 1)
        food_y = min(max(0, self.food.y // BLOCK_SIZE), map_height - 1)
        state_matrix[food_y, food_x] = 1
        
        # 获取上一步操作（如果没有上一步操作，则默认为直行）
        last_action = [1, 0, 0]  # 默认直行
        if hasattr(self, 'last_action'):
            last_action = self.last_action
        
        # 将状态矩阵展平并添加上一步操作
        state = np.concatenate([
            state_matrix.flatten(),  # 地图状态
            last_action  # 上一步操作
        ])
        
        return state
    
    def _is_collision(self, pt=None):
        if pt is None:
            pt = self.head
        # 撞墙
        if pt.x >= self.w or pt.x < 0 or pt.y >= self.h or pt.y < 0:
            return True
        # 撞到自己
        if pt in self.snake[1:]:
            return True
        return False
    
    def step(self, action):
        self.frame_iteration += 1
        self.steps_since_last_food += 1
        
        # 保存当前动作作为下一步的上一步操作
        self.last_action = action
        
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
            reward = self.death_penalty
            return reward, game_over, self.score
        
        # 计算到食物的距离变化
        old_distance = abs(old_head.x - self.food.x) + abs(old_head.y - self.food.y)
        new_distance = abs(self.head.x - self.food.x) + abs(self.head.y - self.food.y)
        distance_reward = self.distance_reward if new_distance < old_distance else -self.distance_reward
        
        # 吃到食物
        if self.head == self.food:
            self.score += 1
            # 计算奖励
            step_bonus = np.exp(-self.steps_since_last_food / self.step_bonus_decay) * self.step_bonus_multiplier
            length_bonus = len(self.snake) * self.length_bonus
            reward = self.base_reward + step_bonus + length_bonus
            
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
        if not self.render_ui:
            return
            
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
        self.clock.tick(SPEED)  # 控制帧率
    
    def close(self):
        if self.render_ui:
            pygame.quit()
    
    def get_state_size(self):
        """返回状态空间的大小"""
        map_width = self.w // BLOCK_SIZE
        map_height = self.h // BLOCK_SIZE
        # 状态包括：地图状态 + 上一步动作
        return map_width * map_height + 3 