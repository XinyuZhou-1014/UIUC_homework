from random import random, uniform, choice
from matplotlib import pyplot as plt
from numpy import arange

class Simulator:
    
    def __init__(self, alpha_value=0, gamma_value=0, epsilon_value=0, 
        decreasing=False, randomInitiate=False):
        '''
        Setup the Simulator with the provided values.
        :param num_games - number of games to be trained on.
        :param alpha_value - 1/alpha_value is the decay constant.
        :param gamma_value - Discount Factor.
        :param epsilon_value - Probability value for the epsilon-greedy approach.
        '''
        #self.num_games = num_games       
        self.epsilon_value = epsilon_value       
        self.alpha_value = alpha_value       
        self.gamma_value = gamma_value
        self.decreasing = decreasing
        self.randInit = randomInitiate
        
        # Your Code Goes Here!
        ball_x = 0.5
        ball_y = 0.5
        velocity_x = 0.03 
        velocity_y = 0.01
        paddle_y = 0.4
        self.paddle_height = 0.2

        self.totalGame = 1
        self.action = 0
        self.totalPoint = 0
        self.Q = [[random() for i in range(3)] for j in range(144 * 2 * 3 * 12 + 1)]
        self.Q[-1] = [-1, -1, -1]
        self.R = [0 for j in range(144 * 2 * 3 * 12)]

        self.debug = []

        self.init_state = (ball_x, ball_y, velocity_x, velocity_y, paddle_y)
        self.state = self.init_state
        self.oldDiscreteState = self.state_calculate()
        self.newDiscreteState = self.oldDiscreteState
        
        
    def randomInitiate(self):
        ball_x = uniform(0.3, 0.7)
        ball_y = uniform(0.3, 7)
        velocity_x = uniform(-0.06, 0.06)
        if abs(velocity_x) < 0.03:
            velocity_x = (abs(velocity_x) // velocity_x) * 0.03 
        velocity_y = uniform(-0.02, 0.02)
        paddle_y = choice(arange(0, 0.84, 0.04))
        self.init_state = (ball_x, ball_y, velocity_x, velocity_y, paddle_y)
        
    
    def f_function(self):

        '''
        Choose action based on an epsilon greedy approach
        :return action selected
        '''

        action_selected = 0
        
        # Your Code Goes Here!
        if random() < self.epsilon_value:
            action_selected = choice([-1, 0, 1])
        else:
            l = self.Q[self.oldDiscreteState]
            if l[0] == l[1] == l[2]:
                action_selected = choice([-1, 0, 1])
            index = l.index(max(l))
            action_selected = [-1, 0, 1][index]
        return action_selected


    def ball_and_paddle_move(self):
        flag = None
        ball_x, ball_y, velocity_x, velocity_y, paddle_y = self.state
        ball_x += velocity_x
        ball_y += velocity_y
        paddle_y += self.action * 0.04

        if ball_y < 0:
            ball_y *= -1
            velocity_y *= -1
        elif ball_y > 1:
            ball_y = 2 - ball_y
            velocity_y *= -1
        if ball_x < 0:
            ball_x *= -1
            velocity_x *= -1
        elif ball_x > 1:
            if ball_y > paddle_y and ball_y < paddle_y + self.paddle_height:
                flag = 'Point'
                U = uniform(-0.015, 0.015)
                V = uniform(-0.03, 0.03)
                ball_x = 2 - ball_x
                velocity_x = - velocity_x + U
                velocity_y += V
                if abs(velocity_x) < 0.03:
                    velocity_x = (abs(velocity_x) // velocity_x) * 0.03
                
            else:
                flag = 'Fail'
                #ball_x, ball_y, velocity_x, velocity_y, paddle_y = self.state


        self.state = (ball_x, ball_y, velocity_x, velocity_y, paddle_y)
        return flag

    def train_Q(self):
        def _func(Q, a, s, ss, R):
            return Q[s][a] + self.alpha_value * (R[s] + self.gamma_value * max(Q[ss]) - Q[s][a])
        s = self.oldDiscreteState
        ss = self.newDiscreteState
        a = self.action + 1
        self.Q[s][a] = _func(self.Q, a, s, ss, self.R)


    def state_calculate(self):
        def _grid_calulate(x, y):
            x, y = int(x * 12), int(y * 12)
            x, y = min(x, 11), min(y, 11)
            x, y = max(0, x), max(0, y)
            return x * 12 + y
        def _others(v_x, v_y, paddle_y, paddle_height):
            v_x_discrete = int (v_x / abs(v_x))
            if abs(v_y) < 0.015:
                v_y_discrete = 0
            else:
                v_y_discrete = int (v_y / abs(v_y))
            paddle_height_discrete = int(12 * paddle_y / (1 - paddle_height))
            paddle_height_discrete = min(paddle_height_discrete, 11)
            paddle_height_discrete = max(paddle_height_discrete, 0)
            return (v_x_discrete, v_y_discrete, paddle_height_discrete)

        ball_x, ball_y, velocity_x, velocity_y, paddle_y = self.state
        grid = _grid_calulate(ball_x, ball_y)
        v_x_discrete, v_y_discrete, paddle_height_discrete = \
        _others(velocity_x, velocity_y, paddle_y, self.paddle_height)

        discreteState = grid * (2 * 3 * 12) + int((v_x_discrete + 1) / 2) * (3 * 12) \
        + (v_y_discrete + 1) * 12 + paddle_height_discrete
        
        return discreteState

    def one_step(self):
        self.action = self.f_function()
        flag = self.ball_and_paddle_move()
        self.newDiscreteState = self.state_calculate()
        if flag == 'Point': 
            self.game_point()
        elif flag == 'Fail':
            self.game_fail()
        
        self.train_Q()
        self.oldDiscreteState = self.newDiscreteState

    def game_point(self):
        #print('point', ...)
        self.R[self.oldDiscreteState] = 1
        self.totalPoint += 1 
        pass

    def game_fail(self):
        #print('fail', ...)
        self.R[self.oldDiscreteState] = -1
        self.Q[self.oldDiscreteState][self.action + 1] += -1
        if self.randInit == True:
            self.randomInitiate()
        self.state = self.init_state
        self.totalGame += 1
        if self.totalGame % 1000 == 0:
            print(self.totalGame, sum(self.debug[-1000:]) / 1000)
        self.debug.append(self.totalPoint)
        self.totalPoint = 0

        if self.decreasing:
            self.epsilon_value = self.epsilon_value * self.totalGame / (1 + self.totalGame) 

        #self.oldDiscreteState = self.state_calculate()
        #self.ball_and_paddle_move()
        #self.newDiscreteState = self.state_calculate()
        pass


if 0:
#if __name__ == '__main__':
    for alpha_value in [0.1, 0.4][1:]:
        for gamma_value in [0.95, 0.75]:
            for epsilon_value in [0.01, 0.04, 0.1]:
                simulator = Simulator(alpha_value, gamma_value, epsilon_value)
                while True:
                    simulator.one_step()
                    if simulator.totalGame > 100000:
                        break
                speed = []
                for time in [5000, 10000, 20000, 40000, 60000]:
                    speed.append(sum(simulator.debug[time-1000: time])/1000)
                res = []
                for length in [5, 10, 100, 1000]:
                    res.append(sum(simulator.debug[-length:])/length)
                t = 'parameters: %s, speed: %s, res:%s \n' \
                % ((alpha_value, gamma_value, epsilon_value), speed, res)
                with open('record.txt', 'a') as f:
                    f.write(t)
 
if 1:
    alpha_value = 0.2
    gamma_value = 0.95
    epsilon_value = 0.04
    simulator = Simulator(alpha_value=0.2, gamma_value=0.95, epsilon_value=0.04, randomInitiate=False)
    while True:
        simulator.one_step()
        if simulator.totalGame > 100000:
            break
    speed = []
    for time in [5000, 10000, 20000, 40000, 60000]:
        speed.append(sum(simulator.debug[time-1000: time])/1000)
        res = []
    for length in [5, 10, 100, 1000]:
        res.append(sum(simulator.debug[-length:])/length)
    t = 'parameters: %s, speed: %s, res:%s \n' \
    % ((alpha_value, gamma_value, epsilon_value), speed, res)
    with open('record.txt', 'a') as f:
        f.write(t)