import pygame
from pygame.locals import *
from sys import exit
import time
import random
from pygame.sprite import Sprite
from pygame.font import Font
from pygame import Surface, Rect
from math import sin, cos, tan, pi, sqrt, asin, acos, atan, sqrt
from matplotlib import pyplot as plt

from simulator import Simulator

class Board(Sprite):
    def __init__(self, pos, size, screensize):
        Sprite.__init__(self)
        self.pos = list(pos)
        self.size = size
        self.img = Surface(size)
        self.img.fill((255, 255, 255))
        self.rect = Rect(self.pos, self.size)
        self.speed = 0
        self.screensize = screensize

    def changeLoc(self, y):
        self.pos = [self.pos[0], y * self.screensize[1]]
    def update(self):
        self.rect = Rect(self.pos, self.size)
        pass

class Ball(Sprite):
    def __init__(self, pos, size, screensize):
        speedsize = 10
        Sprite.__init__(self)
        self.pos = list(pos)
        self.size = size
        self.img = Surface(size)
        #self.img.fill((255, 255, 255))
        self.rect = Rect(self.pos, self.size)
        self.screensize = screensize

    def changeLoc(self, x, y):
        self.pos = [x * self.screensize[0], y * self.screensize[1]]
    def update(self):
        self.rect = Rect(self.pos, self.size)
        pygame.draw.circle(self.img, (127, 127, 127), [self.size[0]//2, self.size[1]//2], self.size[0]//2)
        pass


simulator = Simulator(alpha_value=0.4, gamma_value=0.95, epsilon_value=0.04, decreasing=False)
pygame.init()
SCREENSIZE = (800, 600)
BOARDSIZE = (10, 0.2 * 600)
screen = pygame.display.set_mode(SCREENSIZE, 0, 32)
ball_x, ball_y, velocity_x, velocity_y, paddle_y = simulator.state

#leftBoard = Board((0, 0), BOARDSIZE, SCREENSIZE)
rightBoard = Board((SCREENSIZE[0]-BOARDSIZE[0], paddle_y * 600), BOARDSIZE, SCREENSIZE)
ball = Ball((ball_x * SCREENSIZE[0], ball_y * SCREENSIZE[1]), [40, 40], SCREENSIZE)

clock = pygame.time.Clock()
SPEED = 40
while True:
    t = simulator.totalGame
    while True:
        simulator.one_step()
        if simulator.totalGame >= t + 10000:
            break
    t = simulator.totalGame
    while True:
        if pygame.mouse.get_pressed()[0]:
            if SPEED == 40:
                SPEED = 1
            else:
                SPEED = 40
        clock.tick(60)
        event = pygame.event.poll()
        if event.type == QUIT:
            exit()
        screen.fill((0, 0, 0))
        
        simulator.one_step()
        ball_x, ball_y, velocity_x, velocity_y, paddle_y = simulator.state
        ball.changeLoc(ball_x, ball_y)
        rightBoard.changeLoc(paddle_y)



        #leftBoard.update()
        rightBoard.update()
        screen.blit(ball.img, ball.rect)
        #screen.blit(leftBoard.img, leftBoard.rect)
        screen.blit(rightBoard.img, rightBoard.rect)
        
        ball.update()
        pygame.display.update()

        if simulator.totalGame >= t + 5:
            break

    if simulator.totalGame > 100000:
        #pygame.quit()
        #break
        pass

res = simulator.debug[-1000:]
print(sum(res)/1000)
plt.plot(res)
plt.show()

