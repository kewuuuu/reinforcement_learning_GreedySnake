import pygame
import sys
import json
import os
from enum import Enum
import numpy as np
from typing import List, Tuple, Dict, Optional, Set
import time
from dataclasses import dataclass
from collections import deque
import random

# Constants
WINDOW_SIZE = 800
GRID_SIZE = 20
GRID_COUNT = WINDOW_SIZE // GRID_SIZE
FPS = 10
MENU_ITEM_HEIGHT = 50
MENU_PADDING = 20

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
GRAY = (128, 128, 128)
DARK_GREEN = (0, 200, 0)
LIGHT_GREEN = (144, 238, 144)
YELLOW = (255, 255, 0)
ORANGE = (255, 165, 0)
PURPLE = (128, 0, 128)
CYAN = (0, 255, 255)

class Difficulty(Enum):
    EASY = "Easy"
    NORMAL = "Normal"
    HARD = "Hard"
    EXPERT = "Expert"

    def get_speed(self) -> int:
        speeds = {
            self.EASY: 8,
            self.NORMAL: 10,
            self.HARD: 12,
            self.EXPERT: 15
        }
        return speeds[self]

    def get_special_food_chance(self) -> float:
        chances = {
            self.EASY: 0.15,
            self.NORMAL: 0.1,
            self.HARD: 0.05,
            self.EXPERT: 0.02
        }
        return chances[self]

class Achievement(Enum):
    FIRST_FOOD = "First Food"
    SPEED_DEMON = "Speed Demon"
    SURVIVOR = "Survivor"
    COLLECTOR = "Collector"
    MASTER = "Snake Master"

    def get_description(self) -> str:
        descriptions = {
            self.FIRST_FOOD: "Eat your first food",
            self.SPEED_DEMON: "Reach speed multiplier of 2.0",
            self.SURVIVOR: "Survive for 1000 steps",
            self.COLLECTOR: "Collect 10 special foods",
            self.MASTER: "Score 100 points"
        }
        return descriptions[self]

@dataclass
class Point:
    x: int
    y: int

    def __eq__(self, other):
        if not isinstance(other, Point):
            return False
        return self.x == other.x and self.y == other.y

    def __hash__(self):
        return hash((self.x, self.y))

    def to_tuple(self) -> Tuple[int, int]:
        return (self.x, self.y)

    def distance_to(self, other: 'Point') -> float:
        return ((self.x - other.x) ** 2 + (self.y - other.y) ** 2) ** 0.5

class Direction(Enum):
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3

    @classmethod
    def get_opposite(cls, direction):
        opposites = {
            cls.UP: cls.DOWN,
            cls.RIGHT: cls.LEFT,
            cls.DOWN: cls.UP,
            cls.LEFT: cls.RIGHT
        }
        return opposites[direction]

    @classmethod
    def get_vector(cls, direction) -> Point:
        vectors = {
            cls.UP: Point(0, -1),
            cls.RIGHT: Point(1, 0),
            cls.DOWN: Point(0, 1),
            cls.LEFT: Point(-1, 0)
        }
        return vectors[direction]

class GameMode(Enum):
    PLAYER = "Player"
    AI = "AI"

class AIMode(Enum):
    TRAIN = "Train"
    TEST = "Test"

class GameState(Enum):
    MENU = "Menu"
    PLAYING = "Playing"
    PAUSED = "Paused"
    GAME_OVER = "GameOver"
    SETTINGS = "Settings"
    ACHIEVEMENTS = "Achievements"

class FoodType(Enum):
    NORMAL = "Normal"
    SPECIAL = "Special"
    GOLDEN = "Golden"

@dataclass
class Food:
    position: Point
    type: FoodType = FoodType.NORMAL
    points: int = 1
    lifetime: int = -1
    effect: str = ""

class MenuItem:
    def __init__(self, text: str, action: callable, color: Tuple[int, int, int] = WHITE):
        self.text = text
        self.action = action
        self.color = color
        self.hover = False

class Settings:
    def __init__(self):
        self.game_mode = GameMode.PLAYER
        self.game_speed = 10
        self.save_records = False
        self.record_path = "game_records"
        self.ai_speed = 10
        self.ai_algorithms = {}
        self.ai_mode = AIMode.TEST
        self.concurrent_training = 1
        self.new_training = False
        self.selected_algorithm = ""
        self.learning_rate = 0.001
        self.model_save_path = "models"
        self.model_path = ""
        self.use_player_records = False
        self.player_records_path = ""
        self.training_epochs = 100
        self.show_grid = True
        self.show_score = True
        self.snake_color = "green"
        self.food_color = "red"
        self.background_color = "black"
        self.special_food_chance = 0.1
        self.special_food_points = 5
        self.special_food_lifetime = 50
        self.wall_collision = True
        self.sound_enabled = True
        self.music_enabled = True
        self.difficulty = Difficulty.NORMAL
        self.start_length = 1
        self.max_length = GRID_COUNT * GRID_COUNT // 2
        self.particle_effects = True
        self.screen_shake = True
        self.achievements = set()
        self.high_scores = {
            Difficulty.EASY: 0,
            Difficulty.NORMAL: 0,
            Difficulty.HARD: 0,
            Difficulty.EXPERT: 0
        }

    def save(self, filename: str = "settings.json"):
        os.makedirs(os.path.dirname(filename) if os.path.dirname(filename) else '.', exist_ok=True)
        
        data = {
            "game_mode": self.game_mode.value,
            "game_speed": self.game_speed,
            "save_records": self.save_records,
            "record_path": self.record_path,
            "ai_speed": self.ai_speed,
            "ai_algorithms": self.ai_algorithms,
            "ai_mode": self.ai_mode.value,
            "concurrent_training": self.concurrent_training,
            "new_training": self.new_training,
            "selected_algorithm": self.selected_algorithm,
            "learning_rate": self.learning_rate,
            "model_save_path": self.model_save_path,
            "model_path": self.model_path,
            "use_player_records": self.use_player_records,
            "player_records_path": self.player_records_path,
            "training_epochs": self.training_epochs,
            "show_grid": self.show_grid,
            "show_score": self.show_score,
            "snake_color": self.snake_color,
            "food_color": self.food_color,
            "background_color": self.background_color,
            "special_food_chance": self.special_food_chance,
            "special_food_points": self.special_food_points,
            "special_food_lifetime": self.special_food_lifetime,
            "wall_collision": self.wall_collision,
            "sound_enabled": self.sound_enabled,
            "music_enabled": self.music_enabled,
            "difficulty": self.difficulty.value,
            "start_length": self.start_length,
            "max_length": self.max_length,
            "particle_effects": self.particle_effects,
            "screen_shake": self.screen_shake,
            "achievements": list(self.achievements),
            "high_scores": {k.value: v for k, v in self.high_scores.items()}
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=4)

    def load(self, filename: str = "settings.json"):
        try:
            if os.path.exists(filename):
                with open(filename, 'r') as f:
                    data = json.load(f)
                    self.game_mode = GameMode(data.get("game_mode", "Player"))
                    self.game_speed = data.get("game_speed", 10)
                    self.save_records = data.get("save_records", False)
                    self.record_path = data.get("record_path", "game_records")
                    self.ai_speed = data.get("ai_speed", 10)
                    self.ai_algorithms = data.get("ai_algorithms", {})
                    self.ai_mode = AIMode(data.get("ai_mode", "Test"))
                    self.concurrent_training = data.get("concurrent_training", 1)
                    self.new_training = data.get("new_training", False)
                    self.selected_algorithm = data.get("selected_algorithm", "")
                    self.learning_rate = data.get("learning_rate", 0.001)
                    self.model_save_path = data.get("model_save_path", "models")
                    self.model_path = data.get("model_path", "")
                    self.use_player_records = data.get("use_player_records", False)
                    self.player_records_path = data.get("player_records_path", "")
                    self.training_epochs = data.get("training_epochs", 100)
                    self.show_grid = data.get("show_grid", True)
                    self.show_score = data.get("show_score", True)
                    self.snake_color = data.get("snake_color", "green")
                    self.food_color = data.get("food_color", "red")
                    self.background_color = data.get("background_color", "black")
                    self.special_food_chance = data.get("special_food_chance", 0.1)
                    self.special_food_points = data.get("special_food_points", 5)
                    self.special_food_lifetime = data.get("special_food_lifetime", 50)
                    self.wall_collision = data.get("wall_collision", True)
                    self.sound_enabled = data.get("sound_enabled", True)
                    self.music_enabled = data.get("music_enabled", True)
                    self.difficulty = Difficulty(data.get("difficulty", "Normal"))
                    self.start_length = data.get("start_length", 1)
                    self.max_length = data.get("max_length", GRID_COUNT * GRID_COUNT // 2)
                    self.particle_effects = data.get("particle_effects", True)
                    self.screen_shake = data.get("screen_shake", True)
                    self.achievements = set(data.get("achievements", []))
                    self.high_scores = {
                        Difficulty(k): v for k, v in data.get("high_scores", {}).items()
                    }
            else:
                self.save(filename)
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Error loading settings: {e}")
            print("Using default settings")
            self.save(filename)

class Snake:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.reset()

    def reset(self):
        center = GRID_COUNT // 2
        self.body = deque([Point(center, center)])
        self.direction = Direction.RIGHT
        self.last_direction = Direction.RIGHT
        self.alive = True
        self.score = 0
        self.steps = 0
        self.max_steps = GRID_COUNT * GRID_COUNT
        self.grow = False
        self.length = self.settings.start_length
        self.speed_multiplier = 1.0
        self.invincible = False
        self.invincible_timer = 0
        self.power_ups = set()
        self.special_foods_collected = 0
        self.last_move_time = time.time()
        self.move_interval = 1.0 / self.settings.game_speed

    def move(self):
        if not self.alive:
            return

        current_time = time.time()
        if current_time - self.last_move_time < self.move_interval / self.speed_multiplier:
            return

        self.last_move_time = current_time
        self.steps += 1

        if self.steps > self.max_steps:
            self.alive = False
            return

        # Update invincibility
        if self.invincible:
            self.invincible_timer -= 1
            if self.invincible_timer <= 0:
                self.invincible = False

        # Calculate new head position
        head = self.body[0]
        direction_vector = Direction.get_vector(self.direction)
        new_head = Point(head.x + direction_vector.x, head.y + direction_vector.y)

        # Handle wall collision
        if self.settings.wall_collision:
            if (new_head.x < 0 or new_head.x >= GRID_COUNT or
                new_head.y < 0 or new_head.y >= GRID_COUNT):
                if not self.invincible:
                    self.alive = False
                return
        else:
            # Wrap around
            new_head.x = new_head.x % GRID_COUNT
            new_head.y = new_head.y % GRID_COUNT

        self.last_direction = self.direction
        self.body.appendleft(new_head)

        if not self.grow:
            self.body.pop()
        else:
            self.grow = False
            self.length += 1

        # Check achievements
        self.check_achievements()

    def check_achievements(self):
        if self.score >= 1 and Achievement.FIRST_FOOD not in self.settings.achievements:
            self.settings.achievements.add(Achievement.FIRST_FOOD)
        if self.speed_multiplier >= 2.0 and Achievement.SPEED_DEMON not in self.settings.achievements:
            self.settings.achievements.add(Achievement.SPEED_DEMON)
        if self.steps >= 1000 and Achievement.SURVIVOR not in self.settings.achievements:
            self.settings.achievements.add(Achievement.SURVIVOR)
        if self.special_foods_collected >= 10 and Achievement.COLLECTOR not in self.settings.achievements:
            self.settings.achievements.add(Achievement.COLLECTOR)
        if self.score >= 100 and Achievement.MASTER not in self.settings.achievements:
            self.settings.achievements.add(Achievement.MASTER)

    def change_direction(self, new_direction: Direction):
        if Direction.get_opposite(new_direction) != self.last_direction:
            self.direction = new_direction

    def check_collision(self) -> bool:
        if not self.alive or self.invincible:
            return False

        head = self.body[0]
        
        # Check self collision
        for segment in list(self.body)[1:]:
            if head == segment:
                return True

        return False

    def get_state(self) -> np.ndarray:
        state = np.zeros((GRID_COUNT, GRID_COUNT))
        # Food
        state[self.food.position.y][self.food.position.x] = 1
        # Snake body
        for segment in self.body:
            state[segment.y][segment.x] = 2
        return state

    def add_power_up(self, power_up: str, duration: int):
        self.power_ups.add(power_up)
        if power_up == "invincible":
            self.invincible = True
            self.invincible_timer = duration
        elif power_up == "speed":
            self.speed_multiplier = 1.5

    def remove_power_up(self, power_up: str):
        self.power_ups.discard(power_up)
        if power_up == "speed":
            self.speed_multiplier = 1.0

class Game:
    def __init__(self):
        pygame.init()
        pygame.mixer.init()
        self.screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
        pygame.display.set_caption("Snake Game")
        self.clock = pygame.time.Clock()
        self.settings = Settings()
        self.settings.load()
        self.snake = Snake(self.settings)
        self.food = self.generate_food()
        self.record_steps = []
        self.current_step = 0
        self.game_state = GameState.MENU
        self.pause_font = pygame.font.Font(None, 48)
        self.score_font = pygame.font.Font(None, 36)
        self.menu_font = pygame.font.Font(None, 36)
        self.high_score = self.load_high_score()
        self.load_sounds()
        self.particles = []
        self.last_food_time = time.time()
        self.food_spawn_interval = 5
        self.screen_shake = 0
        self.menu_items = self.create_menu_items()
        self.selected_menu_item = 0
        self.achievement_notification = None
        self.achievement_timer = 0

    def create_menu_items(self) -> List[MenuItem]:
        return [
            MenuItem("Start Game", lambda: self.start_game()),
            MenuItem("Settings", lambda: self.show_settings()),
            MenuItem("Achievements", lambda: self.show_achievements()),
            MenuItem("Quit", lambda: self.quit_game())
        ]

    def start_game(self):
        self.game_state = GameState.PLAYING
        self.reset_game()

    def show_settings(self):
        self.game_state = GameState.SETTINGS

    def show_achievements(self):
        self.game_state = GameState.ACHIEVEMENTS

    def quit_game(self):
        pygame.quit()
        sys.exit()

    def load_sounds(self):
        self.sounds = {}
        sound_files = {
            "eat": "sounds/eat.wav",
            "game_over": "sounds/game_over.wav",
            "power_up": "sounds/power_up.wav",
            "achievement": "sounds/achievement.wav",
            "menu_select": "sounds/menu_select.wav"
        }
        
        for name, file in sound_files.items():
            try:
                if os.path.exists(file):
                    self.sounds[name] = pygame.mixer.Sound(file)
            except:
                print(f"Could not load sound: {file}")

    def play_sound(self, name: str):
        if self.settings.sound_enabled and name in self.sounds:
            self.sounds[name].play()

    def load_high_score(self) -> int:
        try:
            if os.path.exists("high_score.txt"):
                with open("high_score.txt", "r") as f:
                    return int(f.read().strip())
        except:
            pass
        return 0

    def save_high_score(self):
        if self.snake.score > self.high_score:
            self.high_score = self.snake.score
            with open("high_score.txt", "w") as f:
                f.write(str(self.high_score))

    def generate_food(self) -> Food:
        while True:
            position = Point(
                np.random.randint(0, GRID_COUNT),
                np.random.randint(0, GRID_COUNT)
            )
            if position not in self.snake.body:
                # Randomly generate special food
                if (np.random.random() < self.settings.difficulty.get_special_food_chance() and
                    time.time() - self.last_food_time > self.food_spawn_interval):
                    self.last_food_time = time.time()
                    food_type = random.choice([FoodType.SPECIAL, FoodType.GOLDEN])
                    points = self.settings.special_food_points * (2 if food_type == FoodType.GOLDEN else 1)
                    return Food(
                        position=position,
                        type=food_type,
                        points=points,
                        lifetime=self.settings.special_food_lifetime
                    )
                return Food(position=position)

    def create_particles(self, position: Point, color: Tuple[int, int, int], count: int = 10):
        if not self.settings.particle_effects:
            return

        for _ in range(count):
            angle = np.random.random() * 2 * np.pi
            speed = np.random.random() * 5
            self.particles.append({
                'position': Point(position.x, position.y),
                'velocity': Point(
                    np.cos(angle) * speed,
                    np.sin(angle) * speed
                ),
                'color': color,
                'life': 1.0
            })

    def update_particles(self):
        for particle in self.particles[:]:
            particle['position'].x += particle['velocity'].x
            particle['position'].y += particle['velocity'].y
            particle['life'] -= 0.02
            if particle['life'] <= 0:
                self.particles.remove(particle)

    def draw_particles(self):
        for particle in self.particles:
            alpha = int(particle['life'] * 255)
            color = (*particle['color'], alpha)
            pos = particle['position']
            pygame.draw.circle(
                self.screen,
                color,
                (int(pos.x * GRID_SIZE + GRID_SIZE/2),
                 int(pos.y * GRID_SIZE + GRID_SIZE/2)),
                2
            )

    def show_achievement_notification(self, achievement: Achievement):
        self.achievement_notification = achievement
        self.achievement_timer = 180  # 3 seconds at 60 FPS
        self.play_sound("achievement")

    def draw_achievement_notification(self):
        if self.achievement_notification and self.achievement_timer > 0:
            text = f"Achievement Unlocked: {self.achievement_notification.value}"
            notification = self.score_font.render(text, True, YELLOW)
            text_rect = notification.get_rect(center=(WINDOW_SIZE // 2, 50))
            self.screen.blit(notification, text_rect)
            self.achievement_timer -= 1
            if self.achievement_timer <= 0:
                self.achievement_notification = None

    def draw_menu(self):
        self.screen.fill(BLACK)
        
        for i, item in enumerate(self.menu_items):
            color = YELLOW if i == self.selected_menu_item else item.color
            text = self.menu_font.render(item.text, True, color)
            text_rect = text.get_rect(center=(WINDOW_SIZE // 2, 
                                            WINDOW_SIZE // 2 - 
                                            (len(self.menu_items) - 1) * MENU_ITEM_HEIGHT // 2 + 
                                            i * MENU_ITEM_HEIGHT))
            self.screen.blit(text, text_rect)

        # Draw version
        version_text = self.score_font.render("v1.0.0", True, GRAY)
        self.screen.blit(version_text, (WINDOW_SIZE - 60, WINDOW_SIZE - 30))

    def draw_settings(self):
        self.screen.fill(BLACK)
        # TODO: Implement settings menu
        pass

    def draw_achievements(self):
        self.screen.fill(BLACK)
        
        title = self.pause_font.render("Achievements", True, WHITE)
        title_rect = title.get_rect(center=(WINDOW_SIZE // 2, 50))
        self.screen.blit(title, title_rect)

        y = 120
        for achievement in Achievement:
            color = YELLOW if achievement in self.settings.achievements else GRAY
            text = self.score_font.render(
                f"{achievement.value}: {achievement.get_description()}", 
                True, color)
            self.screen.blit(text, (50, y))
            y += 40

        back_text = self.score_font.render("Press ESC to return", True, WHITE)
        back_rect = back_text.get_rect(center=(WINDOW_SIZE // 2, WINDOW_SIZE - 50))
        self.screen.blit(back_text, back_rect)

    def record_step(self):
        if self.settings.save_records:
            step_data = {
                "step": self.current_step,
                "food": self.food.position.to_tuple(),
                "food_type": self.food.type.value,
                "snake_body": [p.to_tuple() for p in self.snake.body],
                "direction": self.snake.direction.value,
                "score": self.snake.score,
                "power_ups": list(self.snake.power_ups)
            }
            self.record_steps.append(step_data)
            self.current_step += 1

    def save_record(self):
        if self.settings.save_records and self.record_steps:
            os.makedirs(self.settings.record_path, exist_ok=True)
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"{self.settings.record_path}/game_{timestamp}.json"
            with open(filename, 'w') as f:
                json.dump({
                    "steps": self.record_steps,
                    "final_score": self.snake.score,
                    "total_steps": self.current_step,
                    "high_score": self.high_score,
                    "settings": self.settings.__dict__
                }, f, indent=4)

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    if self.game_state == GameState.PLAYING:
                        self.game_state = GameState.PAUSED
                    elif self.game_state == GameState.PAUSED:
                        self.game_state = GameState.PLAYING
                    elif self.game_state in [GameState.SETTINGS, GameState.ACHIEVEMENTS]:
                        self.game_state = GameState.MENU
                elif event.key == pygame.K_SPACE:
                    if self.game_state == GameState.GAME_OVER:
                        self.reset_game()
                        self.game_state = GameState.PLAYING
                elif self.game_state == GameState.MENU:
                    if event.key == pygame.K_UP:
                        self.selected_menu_item = (self.selected_menu_item - 1) % len(self.menu_items)
                        self.play_sound("menu_select")
                    elif event.key == pygame.K_DOWN:
                        self.selected_menu_item = (self.selected_menu_item + 1) % len(self.menu_items)
                        self.play_sound("menu_select")
                    elif event.key == pygame.K_RETURN:
                        self.menu_items[self.selected_menu_item].action()
                elif self.game_state == GameState.PLAYING:
                    if event.key == pygame.K_UP:
                        self.snake.change_direction(Direction.UP)
                    elif event.key == pygame.K_RIGHT:
                        self.snake.change_direction(Direction.RIGHT)
                    elif event.key == pygame.K_DOWN:
                        self.snake.change_direction(Direction.DOWN)
                    elif event.key == pygame.K_LEFT:
                        self.snake.change_direction(Direction.LEFT)
        return True

    def update(self):
        if self.game_state != GameState.PLAYING:
            return

        self.snake.move()
        self.record_step()
        self.update_particles()

        if self.snake.check_collision():
            self.snake.alive = False
            self.game_state = GameState.GAME_OVER
            self.save_high_score()
            self.play_sound("game_over")
            if self.settings.screen_shake:
                self.screen_shake = 10
            return

        # Check if food is eaten
        if self.snake.body[0] == self.food.position:
            self.snake.grow = True
            self.snake.score += self.food.points
            self.play_sound("eat")
            self.create_particles(self.food.position, YELLOW)
            
            if self.food.type == FoodType.SPECIAL:
                self.snake.add_power_up("invincible", 50)
                self.play_sound("power_up")
                self.snake.special_foods_collected += 1
            elif self.food.type == FoodType.GOLDEN:
                self.snake.add_power_up("speed", 100)
                self.play_sound("power_up")
                self.snake.special_foods_collected += 1
            
            self.food = self.generate_food()

        # Update food lifetime
        if self.food.lifetime > 0:
            self.food.lifetime -= 1
            if self.food.lifetime <= 0:
                self.food = self.generate_food()

        # Update screen shake
        if self.screen_shake > 0:
            self.screen_shake -= 1

    def draw(self):
        # Draw background
        self.screen.fill(BLACK)

        if self.game_state == GameState.MENU:
            self.draw_menu()
        elif self.game_state == GameState.SETTINGS:
            self.draw_settings()
        elif self.game_state == GameState.ACHIEVEMENTS:
            self.draw_achievements()
        else:
            # Apply screen shake
            shake_offset = (0, 0)
            if self.screen_shake > 0:
                shake_offset = (
                    random.randint(-self.screen_shake, self.screen_shake),
                    random.randint(-self.screen_shake, self.screen_shake)
                )

            if self.settings.show_grid:
                # Draw grid
                for x in range(0, WINDOW_SIZE, GRID_SIZE):
                    pygame.draw.line(self.screen, GRAY, 
                                   (x + shake_offset[0], 0), 
                                   (x + shake_offset[0], WINDOW_SIZE))
                for y in range(0, WINDOW_SIZE, GRID_SIZE):
                    pygame.draw.line(self.screen, GRAY, 
                                   (0, y + shake_offset[1]), 
                                   (WINDOW_SIZE, y + shake_offset[1]))

            # Draw snake
            for i, segment in enumerate(self.snake.body):
                color = DARK_GREEN if i == 0 else LIGHT_GREEN
                if self.snake.invincible:
                    color = YELLOW
                pygame.draw.rect(self.screen, color,
                               (segment.x * GRID_SIZE + shake_offset[0],
                                segment.y * GRID_SIZE + shake_offset[1],
                                GRID_SIZE - 1, GRID_SIZE - 1))

            # Draw food
            food_color = ORANGE if self.food.type == FoodType.SPECIAL else (
                YELLOW if self.food.type == FoodType.GOLDEN else RED
            )
            pygame.draw.rect(self.screen, food_color,
                           (self.food.position.x * GRID_SIZE + shake_offset[0],
                            self.food.position.y * GRID_SIZE + shake_offset[1],
                            GRID_SIZE - 1, GRID_SIZE - 1))

            # Draw particles
            self.draw_particles()

            if self.settings.show_score:
                # Draw score
                score_text = self.score_font.render(f"Score: {self.snake.score}", True, WHITE)
                self.screen.blit(score_text, (10 + shake_offset[0], 10 + shake_offset[1]))
                
                # Draw high score
                high_score_text = self.score_font.render(f"High Score: {self.high_score}", True, WHITE)
                self.screen.blit(high_score_text, (10 + shake_offset[0], 50 + shake_offset[1]))

                # Draw power-ups
                if self.snake.power_ups:
                    power_up_text = self.score_font.render(
                        f"Power-ups: {', '.join(self.snake.power_ups)}", True, WHITE)
                    self.screen.blit(power_up_text, (10 + shake_offset[0], 90 + shake_offset[1]))

            # Draw achievement notification
            self.draw_achievement_notification()

            if self.game_state == GameState.PAUSED:
                pause_text = self.pause_font.render("PAUSED", True, WHITE)
                text_rect = pause_text.get_rect(center=(WINDOW_SIZE // 2, WINDOW_SIZE // 2))
                self.screen.blit(pause_text, text_rect)
            elif self.game_state == GameState.GAME_OVER:
                game_over_text = self.pause_font.render("GAME OVER", True, WHITE)
                text_rect = game_over_text.get_rect(center=(WINDOW_SIZE // 2, WINDOW_SIZE // 2))
                self.screen.blit(game_over_text, text_rect)
                
                restart_text = self.score_font.render("Press SPACE to restart", True, WHITE)
                restart_rect = restart_text.get_rect(center=(WINDOW_SIZE // 2, WINDOW_SIZE // 2 + 50))
                self.screen.blit(restart_text, restart_rect)

        pygame.display.flip()

    def reset_game(self):
        self.snake.reset()
        self.food = self.generate_food()
        self.record_steps = []
        self.current_step = 0
        self.particles = []
        self.screen_shake = 0

    def run(self):
        running = True
        self.game_state = GameState.MENU
        
        while running:
            running = self.handle_events()
            self.update()
            self.draw()
            self.clock.tick(self.settings.game_speed * self.snake.speed_multiplier)

        if self.settings.save_records:
            self.save_record()
        pygame.quit()
        sys.exit()

if __name__ == "__main__":
    game = Game()
    game.run() 