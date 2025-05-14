import pygame
import sys
from snake_game import Game, Settings, GameMode
from settings_ui import SettingsUI

def main():
    # Initialize pygame
    pygame.init()
    
    # Create settings instance
    settings = Settings()
    settings.load()
    
    while True:
        # Show settings UI
        settings_ui = SettingsUI()
        if not settings_ui.run():
            break
            
        # Start game
        game = Game()
        game.run()
        
        # If in player mode, ask to play again
        if settings.game_mode == GameMode.PLAYER:
            # TODO: Add play again dialog
            break

if __name__ == "__main__":
    main() 