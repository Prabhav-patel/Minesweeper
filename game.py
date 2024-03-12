from random import randint

import numpy as np
import pygame

from grid import Grid


class QLearningAgent:
    def __init__(self, grid_size):
        self.grid_size = grid_size
        self.q_table = np.zeros((grid_size, grid_size, 2))  # 2 actions: reveal (0) or flag (1)
        self.alpha = 0.1  # learning rate
        self.gamma = 0.9  # discount factor
        self.epsilon = 0.9  # exploration-exploitation trade-off

    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.choice([0, 1])  # explore
        else:
            return np.argmax(self.q_table[state[0], state[1]])

    def update_q_table(self, state, action, reward, next_state):
        predict = self.q_table[state[0], state[1], action]
        target = reward + self.gamma * np.max(self.q_table[next_state[0], next_state[1]])
        self.q_table[state[0], state[1], action] += self.alpha * (target - predict)

class Game:
    def __init__(self, size, bombs_num):
        self.grid = Grid(size, bombs_num)
        self.breadth = 50
        self.tiles_to_reveal = size * size - bombs_num
        self.agent = QLearningAgent(size)
    
    def draw(self, screen):
        y = 0
        for row in self.grid.grid:
            x = 0
            for tile in row:
                if tile.flagged:
                    pygame.draw.rect(screen, (0, 0, 255), pygame.Rect(x, y, self.breadth, self.breadth))
                elif tile.revealed:
                    if tile.bomb:
                        pygame.draw.rect(screen, (255, 0, 0), pygame.Rect(x, y, self.breadth, self.breadth))
                    else:
                        pygame.draw.rect(screen, (200, 200, 200), pygame.Rect(x, y, self.breadth, self.breadth))
                    if not tile.bomb and tile.adjacent_bombs > 0:
                        font = pygame.font.Font(None, 36)
                        text = font.render(str(tile.adjacent_bombs), True, (0, 0, 0))
                        text_rect = text.get_rect(center=(x + self.breadth // 2, y + self.breadth // 2))
                        screen.blit(text, text_rect)
                else:
                    pygame.draw.rect(screen, (150, 150, 150), pygame.Rect(x, y, self.breadth, self.breadth))
                pygame.draw.rect(screen, (0, 0, 0), pygame.Rect(x, y, self.breadth, self.breadth), 1)
                x += self.breadth
            y += self.breadth
        for i in range(self.grid.size + 1):
            pygame.draw.line(screen, (0, 0, 0), (0, i * self.breadth), (self.breadth * self.grid.size, i * self.breadth), 1)
            pygame.draw.line(screen, (0, 0, 0), (i * self.breadth, 0), (i * self.breadth, self.breadth * self.grid.size), 1)

    def reveal_neighbors(self, row, col):
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                x = row + dx
                y = col + dy
                if self.grid.isTileInGrid(x, y):
                    if not self.grid.isRevealed(x, y) and not self.grid.grid[x][y].bomb:
                        self.grid.setRevealed(x, y, True)
                        self.tiles_to_reveal -= 1
                        if self.grid.isSafeTile(x, y):
                            self.reveal_neighbors(x, y)

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            clicked_col = event.pos[0] // self.breadth
            clicked_row = event.pos[1] // self.breadth

            if event.button == 1: #leftclick
                if not self.grid.isRevealed(clicked_row, clicked_col) and not self.grid.isFlagged(clicked_row, clicked_col):
                    self.grid.setRevealed(clicked_row, clicked_col, True)
                    if self.grid.grid[clicked_row][clicked_col].bomb:
                        print("Game Over!")
                        return False
                    
                    self.tiles_to_reveal -= 1
                    if self.grid.isSafeTile(clicked_row, clicked_col):
                        self.reveal_neighbors(clicked_row, clicked_col)
                    if self.tiles_to_reveal == 0:
                        print("You won!")
                        return False
                    
            elif event.button == 3: #rightclick
                if self.grid.isRevealed(clicked_row, clicked_col):
                    return True
                elif self.grid.isFlagged(clicked_row, clicked_col):
                    self.grid.setFlagged(clicked_row, clicked_col ,False)
                else:
                    self.grid.setFlagged(clicked_row, clicked_col ,True)
            
        return True
    
    def agent_play(self, delay_ms=1000):
        if self.tiles_to_reveal == 0:
            print("Agent won!")
            return False
        
        if self.tiles_to_reveal == self.grid.size*self.grid.size:
            row, col = state
            self.grid.setRevealed(row, col, True)
            if self.grid.grid[row][col].bomb:
                print("Game Over!")
                return False

            self.tiles_to_reveal -= 1
            if self.grid.isSafeTile(row, col):
                self.reveal_neighbors(row, col)

        unrevealed_tiles = [(i, j) for i in range(self.grid.size) for j in range(self.grid.size)
                            if not self.grid.isRevealed(i, j)]
        
        initial_tiles = self.tiles_to_reveal

        if unrevealed_tiles:
            state = (unrevealed_tiles[0][0], unrevealed_tiles[0][1])
            action = self.agent.choose_action(state)

            if action == 0:  # reveal
                row, col = state
                self.grid.setRevealed(row, col, True)
                if self.grid.grid[row][col].bomb:
                    print("Game Over!")
                    return False

                self.tiles_to_reveal -= 1
                if self.grid.isSafeTile(row, col):
                    self.reveal_neighbors(row, col)

            elif action == 1:  # flag
                row, col = state
                if not self.grid.isFlagged(row, col):
                    self.grid.setFlagged(row, col, True)

            next_state = (unrevealed_tiles[1][0], unrevealed_tiles[1][1])
            self.agent.update_q_table(state, action, initial_tiles - self.tiles_to_reveal, next_state)

            pygame.time.delay(delay_ms + 200)

        return True