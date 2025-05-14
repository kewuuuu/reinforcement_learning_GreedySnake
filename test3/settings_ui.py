import pygame
import sys
import os
from snake_game import Settings, GameMode, AIMode

# Constants
WINDOW_SIZE = (800, 600)
BUTTON_SIZE = (200, 40)
INPUT_SIZE = (300, 40)
MARGIN = 20

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GRAY = (128, 128, 128)
LIGHT_GRAY = (200, 200, 200)
BLUE = (0, 0, 255)
RED = (255, 0, 0)

class Button:
    def __init__(self, x, y, width, height, text, color=GRAY, hover_color=LIGHT_GRAY):
        self.rect = pygame.Rect(x, y, width, height)
        self.text = text
        self.color = color
        self.hover_color = hover_color
        self.is_hovered = False

    def draw(self, screen):
        color = self.hover_color if self.is_hovered else self.color
        pygame.draw.rect(screen, color, self.rect)
        pygame.draw.rect(screen, BLACK, self.rect, 2)
        
        font = pygame.font.Font(None, 24)
        text_surface = font.render(self.text, True, BLACK)
        text_rect = text_surface.get_rect(center=self.rect.center)
        screen.blit(text_surface, text_rect)

    def handle_event(self, event):
        if event.type == pygame.MOUSEMOTION:
            self.is_hovered = self.rect.collidepoint(event.pos)
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if self.is_hovered:
                return True
        return False

class InputBox:
    def __init__(self, x, y, width, height, text=''):
        self.rect = pygame.Rect(x, y, width, height)
        self.text = text
        self.active = False
        self.font = pygame.font.Font(None, 24)

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            self.active = self.rect.collidepoint(event.pos)
        if event.type == pygame.KEYDOWN:
            if self.active:
                if event.key == pygame.K_RETURN:
                    return True
                elif event.key == pygame.K_BACKSPACE:
                    self.text = self.text[:-1]
                else:
                    self.text += event.unicode
        return False

    def draw(self, screen):
        pygame.draw.rect(screen, WHITE, self.rect)
        pygame.draw.rect(screen, BLACK, self.rect, 2)
        text_surface = self.font.render(self.text, True, BLACK)
        screen.blit(text_surface, (self.rect.x + 5, self.rect.y + 5))

class SettingsUI:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode(WINDOW_SIZE)
        pygame.display.set_caption("Snake Game Settings")
        self.settings = Settings()
        self.settings.load()
        self.create_ui_elements()

    def create_ui_elements(self):
        # Game speed
        self.speed_label = pygame.font.Font(None, 24).render("Game Speed:", True, BLACK)
        self.speed_input = InputBox(MARGIN, 50, INPUT_SIZE[0], INPUT_SIZE[1], str(self.settings.game_speed))

        # Control mode
        self.control_label = pygame.font.Font(None, 24).render("Control Mode:", True, BLACK)
        self.player_control_btn = Button(MARGIN, 100, BUTTON_SIZE[0], BUTTON_SIZE[1], "Player Control")
        self.ai_control_btn = Button(MARGIN + BUTTON_SIZE[0] + MARGIN, 100, BUTTON_SIZE[0], BUTTON_SIZE[1], "AI Control")

        # Player settings
        self.save_records_label = pygame.font.Font(None, 24).render("Save Game Records:", True, BLACK)
        self.save_records_btn = Button(MARGIN, 150, BUTTON_SIZE[0], BUTTON_SIZE[1], 
                                     "Yes" if self.settings.save_records else "No")
        
        self.record_path_label = pygame.font.Font(None, 24).render("Record Save Path:", True, BLACK)
        self.record_path_input = InputBox(MARGIN, 200, INPUT_SIZE[0], INPUT_SIZE[1], self.settings.record_path)

        # AI settings
        self.algorithm_list_label = pygame.font.Font(None, 24).render("Algorithm Files:", True, BLACK)
        self.algorithm_list_input = InputBox(MARGIN, 250, INPUT_SIZE[0], INPUT_SIZE[1], "")
        
        self.model_path_label = pygame.font.Font(None, 24).render("Model Save Path:", True, BLACK)
        self.model_path_input = InputBox(MARGIN, 300, INPUT_SIZE[0], INPUT_SIZE[1], self.settings.model_save_path)

        # AI mode
        self.ai_mode_label = pygame.font.Font(None, 24).render("AI Mode:", True, BLACK)
        self.train_mode_btn = Button(MARGIN, 350, BUTTON_SIZE[0], BUTTON_SIZE[1], "Train")
        self.test_mode_btn = Button(MARGIN + BUTTON_SIZE[0] + MARGIN, 350, BUTTON_SIZE[0], BUTTON_SIZE[1], "Test")

        # Training settings
        self.concurrent_label = pygame.font.Font(None, 24).render("Concurrent Training:", True, BLACK)
        self.concurrent_input = InputBox(MARGIN, 400, INPUT_SIZE[0], INPUT_SIZE[1], str(self.settings.concurrent_training))
        
        self.new_training_label = pygame.font.Font(None, 24).render("New Training:", True, BLACK)
        self.new_training_btn = Button(MARGIN, 450, BUTTON_SIZE[0], BUTTON_SIZE[1], 
                                     "Yes" if self.settings.new_training else "No")

        # New training settings
        self.algorithm_label = pygame.font.Font(None, 24).render("Algorithm:", True, BLACK)
        self.algorithm_input = InputBox(MARGIN, 500, INPUT_SIZE[0], INPUT_SIZE[1], self.settings.selected_algorithm)
        
        self.learning_rate_label = pygame.font.Font(None, 24).render("Learning Rate:", True, BLACK)
        self.learning_rate_input = InputBox(MARGIN, 550, INPUT_SIZE[0], INPUT_SIZE[1], str(self.settings.learning_rate))

        # Existing training settings
        self.model_file_label = pygame.font.Font(None, 24).render("Model File Path:", True, BLACK)
        self.model_file_input = InputBox(MARGIN, 500, INPUT_SIZE[0], INPUT_SIZE[1], self.settings.model_path)
        
        self.use_player_records_label = pygame.font.Font(None, 24).render("Use Player Records:", True, BLACK)
        self.use_player_records_btn = Button(MARGIN, 550, BUTTON_SIZE[0], BUTTON_SIZE[1], 
                                           "Yes" if self.settings.use_player_records else "No")

        # Bottom buttons
        self.start_game_btn = Button(WINDOW_SIZE[0] - 3 * BUTTON_SIZE[0] - 3 * MARGIN, 
                                   WINDOW_SIZE[1] - BUTTON_SIZE[1] - MARGIN,
                                   BUTTON_SIZE[0], BUTTON_SIZE[1], "Start Game")
        self.apply_btn = Button(WINDOW_SIZE[0] - 2 * BUTTON_SIZE[0] - 2 * MARGIN, 
                              WINDOW_SIZE[1] - BUTTON_SIZE[1] - MARGIN,
                              BUTTON_SIZE[0], BUTTON_SIZE[1], "Apply")
        self.cancel_btn = Button(WINDOW_SIZE[0] - BUTTON_SIZE[0] - MARGIN, 
                               WINDOW_SIZE[1] - BUTTON_SIZE[1] - MARGIN,
                               BUTTON_SIZE[0], BUTTON_SIZE[1], "Cancel")

    def update_settings(self):
        # Update game speed
        self.settings.game_speed = int(self.speed_input.text) if self.speed_input.text.isdigit() else 10

        # Update control mode
        if self.player_control_btn.is_hovered:
            self.settings.game_mode = GameMode.PLAYER
        elif self.ai_control_btn.is_hovered:
            self.settings.game_mode = GameMode.AI

        # Update player settings
        self.settings.save_records = self.save_records_btn.text == "Yes"
        self.settings.record_path = self.record_path_input.text

        # Update AI settings
        self.settings.model_save_path = self.model_path_input.text
        if self.train_mode_btn.is_hovered:
            self.settings.ai_mode = AIMode.TRAIN
        elif self.test_mode_btn.is_hovered:
            self.settings.ai_mode = AIMode.TEST

        # Update training settings
        self.settings.concurrent_training = int(self.concurrent_input.text) if self.concurrent_input.text.isdigit() else 1
        self.settings.new_training = self.new_training_btn.text == "Yes"

        if self.settings.new_training:
            self.settings.selected_algorithm = self.algorithm_input.text
            self.settings.learning_rate = float(self.learning_rate_input.text) if self.learning_rate_input.text.replace('.', '').isdigit() else 0.001
        else:
            self.settings.model_path = self.model_file_input.text
            self.settings.use_player_records = self.use_player_records_btn.text == "Yes"

    def run(self):
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return False

                # Handle button events
                if self.player_control_btn.handle_event(event):
                    self.settings.game_mode = GameMode.PLAYER
                elif self.ai_control_btn.handle_event(event):
                    self.settings.game_mode = GameMode.AI
                elif self.save_records_btn.handle_event(event):
                    self.save_records_btn.text = "No" if self.save_records_btn.text == "Yes" else "Yes"
                elif self.train_mode_btn.handle_event(event):
                    self.settings.ai_mode = AIMode.TRAIN
                elif self.test_mode_btn.handle_event(event):
                    self.settings.ai_mode = AIMode.TEST
                elif self.new_training_btn.handle_event(event):
                    self.new_training_btn.text = "No" if self.new_training_btn.text == "Yes" else "Yes"
                elif self.use_player_records_btn.handle_event(event):
                    self.use_player_records_btn.text = "No" if self.use_player_records_btn.text == "Yes" else "Yes"
                elif self.apply_btn.handle_event(event):
                    self.update_settings()
                    self.settings.save()
                elif self.start_game_btn.handle_event(event):
                    self.update_settings()
                    self.settings.save()
                    return True
                elif self.cancel_btn.handle_event(event):
                    return False

                # Handle input box events
                self.speed_input.handle_event(event)
                self.record_path_input.handle_event(event)
                self.algorithm_list_input.handle_event(event)
                self.model_path_input.handle_event(event)
                self.concurrent_input.handle_event(event)
                self.algorithm_input.handle_event(event)
                self.learning_rate_input.handle_event(event)
                self.model_file_input.handle_event(event)

            # Draw everything
            self.screen.fill(WHITE)
            
            # Draw game speed
            self.screen.blit(self.speed_label, (MARGIN, 30))
            self.speed_input.draw(self.screen)

            # Draw control mode
            self.screen.blit(self.control_label, (MARGIN, 80))
            self.player_control_btn.draw(self.screen)
            self.ai_control_btn.draw(self.screen)

            if self.settings.game_mode == GameMode.PLAYER:
                # Draw player settings
                self.screen.blit(self.save_records_label, (MARGIN, 130))
                self.save_records_btn.draw(self.screen)
                
                if self.settings.save_records:
                    self.screen.blit(self.record_path_label, (MARGIN, 180))
                    self.record_path_input.draw(self.screen)
            else:  # AI mode
                # Draw AI settings
                self.screen.blit(self.algorithm_list_label, (MARGIN, 230))
                self.algorithm_list_input.draw(self.screen)
                
                self.screen.blit(self.model_path_label, (MARGIN, 280))
                self.model_path_input.draw(self.screen)

                # Draw AI mode
                self.screen.blit(self.ai_mode_label, (MARGIN, 330))
                self.train_mode_btn.draw(self.screen)
                self.test_mode_btn.draw(self.screen)

                if self.settings.ai_mode == AIMode.TRAIN:
                    self.screen.blit(self.concurrent_label, (MARGIN, 380))
                    self.concurrent_input.draw(self.screen)
                    
                    self.screen.blit(self.new_training_label, (MARGIN, 430))
                    self.new_training_btn.draw(self.screen)

                    if self.settings.new_training:
                        self.screen.blit(self.algorithm_label, (MARGIN, 480))
                        self.algorithm_input.draw(self.screen)
                        
                        self.screen.blit(self.learning_rate_label, (MARGIN, 530))
                        self.learning_rate_input.draw(self.screen)
                    else:
                        self.screen.blit(self.model_file_label, (MARGIN, 480))
                        self.model_file_input.draw(self.screen)
                        
                        self.screen.blit(self.use_player_records_label, (MARGIN, 530))
                        self.use_player_records_btn.draw(self.screen)

            # Draw bottom buttons
            self.start_game_btn.draw(self.screen)
            self.apply_btn.draw(self.screen)
            self.cancel_btn.draw(self.screen)

            pygame.display.flip()

        pygame.quit()
        return False

if __name__ == "__main__":
    settings_ui = SettingsUI()
    settings_ui.run() 