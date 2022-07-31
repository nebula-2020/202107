# -*- coding: utf-8 -*-
"""模型训练和预测。
"""
import random
import sys
from collections import OrderedDict

import numpy as np
import pygame
import torch
from torch import nn, optim

from game import Game
from util import plot, print_bar


class Model(nn.Module):
    """Dueling DQN结构。
    """

    def __init__(self):
        super(Model, self).__init__()
        self.layers = nn.ModuleDict({
            'c': nn.Sequential(
                nn.Linear(3, 12, device=CUDA),
                nn.Sigmoid(),
            ),
            'a': nn.Linear(12, 2, device=CUDA),
            'v': nn.Linear(12, 1, device=CUDA),
            'o': nn.ReLU(),
        })

    def forward(self, arg: torch.Tensor) -> torch.Tensor:
        """模型前向传播。

        Parameters
        ----------
        x : torch.Tensor
            样本输入模型

        Returns
        -------
        torch.Tensor
            预测值。
        """
        output = arg
        output = self.layers['c'](output)
        adv = self.layers['a'](output)
        val = self.layers['v'](output)
        output = self.layers['o'](adv+val)
        return output

    def load_params(self, model: 'Model', rate: float = 1):
        """模型参数软更新。

        Parameters
        ----------
        model : Model
            将这个模型的参数复制到当前模型
        rate : float, optional
            `1`表示将模型参数完全复制到当前模型, by default 1
        """
        for key, value in self.layers.items():
            if rate >= 1.:
                forign = model.layers[key].state_dict()
                value.load_state_dict(forign)
            else:
                local = value.state_dict()
                forign = model.layers[key].state_dict()
                mix = OrderedDict()
                for key in local.keys():
                    mix[key] = local.get(key)*(1-rate) + forign.get(key)*rate
                value.load_state_dict(mix)


def simulate(
    model: Model,
    batch_size: int,
    epslion: float = .1,
    eval_step: int = None,
    env_args: dict = None
) -> 'tuple[list,float,int]':
    """模拟游戏过程并收集数据。

    Parameters
    ----------
    model : Model
        决策用
    batch_size : int
        收集数据总条数
    epslion : float, optional
        尝试比例, by default .1
    eval_step : int, optional
        模型将控制游戏的最大步数，参与模型评估, by default `batch_size`
    env_args : dict, optional
        环境初始化参数, by default None

    Returns
    -------
    tuple[list,float,int]
        采集的数据, 平均存活时长, 无探索情况下生存时间
    """
    cache = []
    env = Game(**env_args, without_screen=True)
    livetimes = []
    livetime = 0
    for _ in range(batch_size):
        state = env.shot()
        if random.random() <= epslion:
            action_index = random.randint(0, len(ACTIONS)-1)
        else:
            values = model(torch.tensor(state, device=CUDA))
            action_index = values.argmax(-1)
        jump = ACTIONS[action_index]
        env.step(jump)
        next_state = env.shot()
        reward = float(env.playing)
        cache.append((state, action_index, next_state, reward))
        if not env.playing:
            env = Game(**env_args, without_screen=True)
            livetimes.append(livetime)
        else:
            livetime += 1
    env = Game(**env_args, without_screen=True)
    max_step = eval_step or batch_size
    livetime = 0
    for _ in range(max_step):
        state = env.shot()
        values = model(torch.tensor(state, device=CUDA))
        action_index = values.argmax(-1)
        jump = ACTIONS[action_index]
        env.step(jump)
        if not env.playing:
            break
        livetime += 1
    return cache, sum(livetimes)/max(1, len(livetimes))/batch_size, livetime


def train(
    policy_net: Model,
    opt: optim.Optimizer,
    loss_func: 'nn._Loss',
    epochs: int,
    batch_size: int,
    cache_size: int,
    epslion: float = .1,
    gamma: float = .5,
    update_ratio: float = .5,
    eval_step: int = None,
    target_accuracy=.99,
    env_args: dict = None,
) -> 'tuple[Model,list[float],list[float],list[int]]':
    """训练模型。

    Parameters
    ----------
    policy_net : Model
        决策网络对象
    opt : optim.Optimizer
        优化器
    loss_func : nn._Loss
        损失函数
    epochs : int
        迭代轮数
    batch_size : int
        批量
    epslion : float, optional
        探索动作比例, by default .1
    gamma : float, optional
        未来奖励权重，`0`表示仅考虑当前奖励, by default .5
    update_ratio : float, optional
        软更新比例, by default .5
    target_accuracy : float, optional
        模型决策目标得分, by default .99
    env_args : dict, optional
        环境初始化参数, by default None

    Returns
    -------
    tuple[Model,list[float],list[float],list[int]]
        目标网络, 损失, 得分
    """
    target_net = Model()
    target_net.load_params(policy_net)
    policy_net.train(mode=True)
    target_net.train(mode=False)
    loss_vals, accuracies, livetimes, cache = [], [], [], []
    for epoch in range(epochs):
        target_net.load_params(policy_net, update_ratio)
        batch, accuracy, livetime = simulate(
            model=target_net,
            batch_size=batch_size,
            epslion=epslion,
            eval_step=eval_step,
            env_args=env_args
        )
        accuracies.append(accuracy)
        livetimes.append(livetime)
        if livetime/(eval_step or batch_size) >= target_accuracy:
            break
        cache.extend(batch)
        cache = cache[-cache_size:]
        states, actions, nexts, rewards = [], [], [], []
        for state, action, next_state, reward in random.sample(cache, batch_size):
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            nexts.append(next_state)
        states = torch.tensor(states, device=CUDA)
        actions = torch.tensor(actions, device=CUDA).unsqueeze(-1)
        rewards = torch.tensor(rewards, device=CUDA)
        nexts = torch.tensor(nexts, device=CUDA)
        v_target = target_net.forward(nexts).detach()
        y_target = v_target.max(dim=-1).values * gamma
        y_target += rewards * (1-gamma)
        v_eval = policy_net.forward(states)
        y_eval = v_eval.gather(index=actions, dim=-1)
        loss = loss_func(y_eval, y_target)
        opt.zero_grad()
        loss.backward()
        opt.step()
        loss = loss.item()
        loss_vals.append(loss)
        print_bar(
            epoch+1,
            epochs,
            ("%.10f" % loss, '%.10f' % accuracy, livetime),
        )
    return target_net, loss_vals, accuracies, livetimes


np.set_printoptions(suppress=True)
CUDA = torch.device("cuda")
MODEL = Model()
OPT = optim.Adam(MODEL.parameters(), lr=.01)
LOSS_FUNCTION = nn.MSELoss()
ACTIONS = (True, False)
SCREEN_SIZE = (800, 600)
FPS = 20
GAME_CONFIG = {
    'screen_size': SCREEN_SIZE,
    'door_size': (80, 180),
    'speed': 10,
    'jump_force': 3,
    'g': 2,
    'door_distance': 60,
}
if __name__ == "__main__":
    pygame.init()  # 初始化
    model, loss_vals, accuracies, livetimes = train(
        policy_net=MODEL,
        opt=OPT,
        loss_func=LOSS_FUNCTION,
        epochs=20000,
        batch_size=192,
        cache_size=2000,
        epslion=.3,
        gamma=.9,
        update_ratio=.1,
        target_accuracy=.95,
        env_args=GAME_CONFIG,
        eval_step=1200,
    )
    plot(
        data={
            'loss': loss_vals,
            'accuracy': accuracies,
            'livetimes': livetimes,
        },
        colors=['#f80', '#08f', '#f08']
    )

    print('\n\n')
    model = model.to('cpu')
    model.train(mode=False)
    SCREEN = pygame.display.set_mode(SCREEN_SIZE)  # 定义屏幕要在开头
    fcclock = pygame.time.Clock()  # 创建一个时间对象
    game = Game(**GAME_CONFIG)
    while True:  # 循环，直到接收到窗口关闭事件
        for event in pygame.event.get():  # 处理事件
            if event.type == pygame.QUIT:  # 接收到窗口关闭事件
                pygame.quit()  # 退出
                sys.exit()
        keys = pygame.key.get_pressed()
        if keys[pygame.K_ESCAPE]:
            pygame.quit()
            sys.exit()
        else:
            state = torch.tensor(game.shot())
            values = model.forward(state)
            action_index = values.argmax(-1)
            jump = ACTIONS[action_index]
            print(f"\033[A{values.squeeze().tolist()}")
        game.step(jump)
        pygame.display.set_caption(f'SCORE: {game.score}')  # 设置窗口标题
        game.draw(SCREEN)
        fcclock.tick(FPS)  # 卡时间
        pygame.display.update()
        if not game.playing:
            game = Game(**GAME_CONFIG)
