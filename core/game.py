#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pygame
import random


class Box:
    __position = None
    __size = None
    __speed = None
    __acceleration = None

    def __init__(
        self,
        cx: int, cy: int, w: int, h: int,
        sx: int = 0, sy: int = 0, ax: int = 0, ay: int = 0,
        *args, **kwargs
    ):
        self.__position = (cx, cy)
        self.__size = (w, h)
        self.__speed = (sx, sy)
        self.__acceleration = (ax, ay)

    @property
    def width(self):
        return self.__size[0]

    @property
    def height(self):
        return self.__size[-1]

    @property
    def size(self):
        return self.__size

    @property
    def x(self):
        return self.__position[0]

    @property
    def y(self):
        return self.__position[-1]

    @property
    def position(self):
        return self.__position

    @property
    def speed_x(self):
        return self.__speed[0]

    @property
    def speed_y(self):
        return self.__speed[-1]

    @property
    def speed(self):
        return self.__speed

    @property
    def acceleration_x(self):
        return self.__acceleration[0]

    @property
    def acceleration_y(self):
        return self.__acceleration[-1]

    @property
    def acceleration(self):
        return self.__acceleration

    @property
    def left(self):
        return self.x-self.width/2

    @property
    def right(self):
        return self.x+self.width/2

    @property
    def top(self):
        return self.y-self.height/2

    @property
    def bottom(self):
        return self.y+self.height/2

    def move_to(self, box: 'Box'):
        self.__position = box.position
        self.__speed = box.speed
        self.__acceleration = box.acceleration
        self.__size = box.size


def move(
    box: Box,
    speed_x=None, speed_y=None,
    acceleration_x=None, acceleration_y=None
) -> Box:
    cx = box.x+box.speed_x
    cy = box.y+box.speed_y
    ax = box.acceleration_x if acceleration_x is None else acceleration_x
    ay = box.acceleration_y if acceleration_y is None else acceleration_y
    sx = (box.speed_x if speed_x is None else speed_x)+ax
    sy = (box.speed_y if speed_y is None else speed_y)+ay
    return Box(
        cx=cx, cy=cy, w=box.width, h=box.height,
        sx=sx, sy=sy, ax=ax, ay=ay
    )


def is_intersect(player: Box, door: Box) -> bool:
    return (door.top > player.top or player.bottom > door.bottom) \
        and not (player.left >= door.right or door.left >= player.right)


class GameObject(Box):
    __imgs = None
    __img_cd = 1
    img_index = -1
    living = True

    def __init__(self, imgs: list, img_cd: int = 1, *args, **kwargs):
        super(GameObject, self).__init__(*args, **kwargs)
        self.__imgs = [item for item in imgs]
        self.__img_cd = img_cd

    def img_grow(self):
        self.img_index = (self.img_index+1) % self.__img_cd

    @property
    def img(self):
        return self.__imgs[self.img_index]


def step(
    player: Box, door: Box, action: 'bool|float',
    screen_height: int, jump_force: int, g: int
) -> 'tuple[Box, Box, bool]':
    moved_player = move(
        box=player,
        speed_y=0 if action else None,
        acceleration_y=-jump_force if action >= .5 else g
    )
    moved_door = move(door)
    living = 0 < moved_player.y < screen_height \
        and not is_intersect(moved_player, moved_door)
    return moved_player, moved_door, living


class Game:
    door_size = None
    player = None
    jump_force = 0
    g = 1
    door_distance = 0
    doors = None
    time = 1
    score = 0

    def __init__(
        self,
        screen_size=(800, 600),
        player_size=(160, 80),
        door_size=(80, 160),
        speed=5,
        jump_force=1.25,
        g=0.2,
        door_distance=100
    ):
        self.player = GameObject(
            cx=screen_size[0]/4,
            cy=screen_size[1]/2,
            w=player_size[0],
            h=player_size[1],
            sx=0, sy=0,
            ax=0, ay=g,
            imgs=[
                pygame.image.load(
                    './assets/textures/player_age0.gif'
                ).convert_alpha(),
                pygame.image.load(
                    './assets/textures/player_age1.gif'
                ).convert_alpha(),
            ],
            img_cd=2
        )
        self.screen_size = screen_size
        self.door_size = door_size
        self.speed = speed
        self.jump_force = jump_force
        self.g = g
        self.door_distance = door_distance
        self.doors = [self.create_door()]

    @property
    def playing(self) -> bool:
        return self.player.living

    @property
    def door(self) -> 'GameObject|None':
        for door in self.doors:
            if door.living:
                return door
        return None

    def create_door(self):
        door = GameObject(
            cx=self.screen_size[0],
            cy=random.randint(
                self.door_size[1]/2,
                self.screen_size[1]-self.door_size[1]/2
            ),
            w=self.door_size[0],
            h=self.door_size[1],
            sx=-self.speed,
            imgs=[
                pygame.image.load(
                    './assets/textures/door.gif'
                ).convert_alpha(),
            ],
            img_cd=2
        )
        return door

    def draw(self, screen: pygame.Surface):
        if not self.player.living:
            return
        screen.fill([86, 92, 66])
        self.player.img_grow()
        screen.blit(
            pygame.transform.scale(
                self.player.img,
                (self.player.width, self.player.height)
            ),
            (self.player.left, self.player.top)
        )
        for door in self.doors:
            screen.blit(
                pygame.transform.scale(door.img, (door.width, door.top)),
                (door.left, 0)
            )
            screen.blit(
                pygame.transform.scale(
                    door.img,
                    (door.width, self.screen_size[1]-door.bottom)
                ),
                (door.left, door.bottom)
            )

    def step(self, jump=False):
        # 玩家必须存活才能继续游戏
        if not self.player.living:
            return

        if self.time % self.door_distance == 0 or not len(self.doors):
            # 时间间隔生成门，时间重置
            self.doors.append(self.create_door())
            self.time = 1
        else:
            # 时间正常递增直到时间间隔
            self.time += 1

        # 清除已经移除屏幕的门
        while self.doors[0].right < 0:
            del self.doors[0]

        # 移动玩家和所有门
        for door in self.doors:
            door.move_to(move(door))
        door = self.door
        player_box, _, living = step(
            self.player,
            self.door,
            screen_height=self.screen_size[1],
            jump_force=self.jump_force,
            g=self.g,
            action=jump,
        )
        self.player.move_to(player_box)

        # 判断玩家和门存活
        if door.living and self.player.left >= door.right:
            door.living = False
            self.score += 1
        self.player.living = living
