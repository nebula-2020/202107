#!/usr/bin/env python
# -*- coding: utf-8 -*-
import random
import sys

import numpy as np
import pygame
import tensorflow as tf
import tensorflow.python.keras as keras
import tensorflow.python.keras.activations as afs
import tensorflow.python.keras.layers as layers
import tensorflow.python.keras.losses as losses
import tensorflow.python.keras.optimizers as opts

from core.game import Box, Game, step

np.set_printoptions(suppress=True)

LAYERS = [
    layers.Dense(units=2, activation=afs.sigmoid,),
    layers.Dense(units=3, activation=afs.sigmoid,),
    layers.Dense(units=2, activation=afs.sigmoid,),
]
MODEL = keras.Sequential(layers=LAYERS, name="pilot")
opt = opts.nadam_v2.Nadam(learning_rate=.0001)
MODEL.compile(
    optimizer=opt,
    loss=losses.binary_crossentropy
)


def calc_value(
    model: keras.Model,
    player: Box, door: Box,
    screen_size: 'tuple[int,int]',
    jump_force: int, g: int,
    actions: 'list|tuple',
    r: int = -1e3,
):
    a_star = None
    v_star = r-1
    for a in actions:  # 这个得打乱顺序
        value = 0
        door_ = door
        player_ = player
        while door_.right > player_.left:
            player_, door_, living = step(
                player_, door_, a, screen_size[1], jump_force, g
            )
            if living:
                value += 1
            else:
                value += r
                break
            res = model.call(
                tf.convert_to_tensor([[
                    (door_.x-player_.x)/screen_size[0],
                    (player_.y-door_.y)/screen_size[-1],
                ]]), training=False
            )
            res = np.array(res).T.tolist()
            a = res[0][0]
        if v_star < value:
            a_star = a
            v_star = value
    return a_star


def episode(
    model: keras.Model,
    player_size: 'tuple[int, int]',
    door_size: 'tuple[int, int]',
    screen_size: 'tuple[int, int]',
    game_speed: 'int|float',
    jump_force: 'int|float',
    g: 'int|float',
    status_count=50,
    epochs=200
):
    # 随机初始化状态
    status = []
    x = []
    y = []
    for _ in range(status_count):
        player = Box(
            cx=0,
            cy=random.randint(
                player_size[1]/2,
                screen_size[1]-player_size[1]/2
            ),
            w=player_size[0],
            h=player_size[1],
            sx=0,
            sy=random.randint(0, screen_size[1]/4),
            ax=0,
            ay=g,
        )
        door = Box(
            cx=random.randint((player_size[0]+door_size[0])/2, screen_size[0]),
            cy=random.randint(
                door_size[1]/2,
                screen_size[1]-door_size[1]/2
            ),
            w=door_size[0],
            h=door_size[1],
            sx=game_speed,
        )
        status.append({
            'player': player,
            'door': door,
            'screen': screen_size,
            'jump': jump_force,
            'g': g,
        })
        x.append(
            np.array([
                    (door.x-player.x)/screen_size[0],
                    (player.y-door.y)/screen_size[-1],
            ]).T
        )
        a = calc_value(
            model, player, door, screen_size,
            jump_force, g, actions=[1, 0]
        )
        y.append([a, 1])
    model.fit(
        tf.convert_to_tensor(x),
        tf.convert_to_tensor(y),
        epochs=epochs,
        verbose=1
    )


ACTIONS = (True, False)
SCREEN_SIZE = (800, 600)
WEIGHT_SHAPE = (5, 1)
FPS = 20
FIT_DELAY = 2
GAME_CONFIG = {
    'screen_size': SCREEN_SIZE,
    'speed': 10,
    'jump_force': 12,
    'g': 2,
    'door_distance': 60,
}


def main():
    pygame.init()  # 初始化
    screen = pygame.display.set_mode(SCREEN_SIZE)
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
        elif keys[pygame.K_SPACE]:
            a = 1
        else:
            a = 0
        game.step(a)
        pygame.display.set_caption(f'SCORE: {game.score}')  # 设置窗口标题
        game.draw(screen)
        fcclock.tick(FPS)  # 卡时间
        pygame.display.update()
        if not game.playing:
            game = Game(**GAME_CONFIG)


if __name__ == "__main__":
    main()
