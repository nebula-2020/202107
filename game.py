# -*- coding: utf-8 -*-
"""游戏环境相关。
"""
import random
import sys
import pygame


class Box:
    """包含基础位置、尺寸、速度、加速度的盒子类。
    """
    __position = None
    __size = None
    __speed = None
    __acceleration = None

    def __init__(
        self,
        cx: int, cy: int, w: int, h: int,
        sx: int = 0, sy: int = 0, ax: int = 0, ay: int = 0
    ):
        self.__position = [cx, cy]
        self.__size = [w, h]
        self.__speed = [sx or 0, sy or 0]
        self.__acceleration = [ax or 0, ay or 0]

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

    @speed_x.setter
    def speed_x(self, v):
        self.__speed[0] = v

    @property
    def speed_y(self):
        return self.__speed[-1]

    @speed_y.setter
    def speed_y(self, v):
        self.__speed[-1] = v

    @property
    def speed(self):
        return self.__speed

    @speed.setter
    def speed(self, v: 'tuple[int,int]'):
        self.__speed[0] = v[0]
        self.__speed[-1] = v[-1]

    @property
    def acceleration_x(self):
        return self.__acceleration[0]

    @acceleration_x.setter
    def acceleration_x(self, v: int):
        self.__acceleration[0] = v

    @property
    def acceleration_y(self):
        return self.__acceleration[-1]

    @acceleration_y.setter
    def acceleration_y(self, v: int):
        self.__acceleration[-1] = v

    @property
    def acceleration(self):
        return self.__acceleration

    @acceleration.setter
    def acceleration(self, v: 'tuple[int,int]'):
        self.__acceleration[0] = v[0]
        self.__acceleration[-1] = v[-1]

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

    def move(self, force_x: int = None, force_y: int = None):
        """为盒子施力使其移动。

        Parameters
        ----------
        force_x : int, optional
            水平分量, by default None
        force_y : int, optional
            垂直分量, by default None
        """
        self.acceleration_x = force_x or 0
        self.acceleration_y = force_y or 0
        self.speed_x += self.acceleration_x
        self.speed_y += self.acceleration_y
        self.__position[0] += self.speed_x
        self.__position[-1] += self.speed_y


def is_intersect(player: Box, door: Box) -> bool:
    return (door.top > player.top or player.bottom > door.bottom) \
        and not (player.left >= door.right or door.left >= player.right)


class GameObject(Box):
    """游戏基础对象。
    """

    def __init__(self, imgs: list, img_cd: int = 1, *args, **kwargs):
        super(GameObject, self).__init__(*args, **kwargs)
        self.__imgs = [item for item in imgs]
        self.__img_cd = img_cd or -1
        self.living = True
        self.img_index = -1

    def img_grow(self):
        self.img_index = (self.img_index+1) % self.__img_cd

    @property
    def img(self):
        return self.__imgs[self.img_index]


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
        jump_force=1.3,
        g=0.4,
        door_distance=100,
        max_falling_speed: int = 100,
        without_screen=False,
        **_
    ):
        self.player = GameObject(
            cx=screen_size[0]/4,
            cy=screen_size[1]/2,
            w=player_size[0],
            h=player_size[1],
            sx=0, sy=0,
            ax=0, ay=g,
            imgs=[None, ] if without_screen else[
                pygame.image.load(
                    './assets/textures/player_age0.gif'
                ).convert_alpha(),
                pygame.image.load(
                    './assets/textures/player_age1.gif'
                ).convert_alpha(),
            ],
            img_cd=2
        )
        self.without_screen = without_screen
        self.screen_size = screen_size
        self.door_size = door_size
        self.speed = speed
        self.jump_force = jump_force
        self.g = g
        self.door_distance = door_distance
        self.max_falling_speed = max_falling_speed
        self.doors = [self.create_door()]

    @property
    def playing(self) -> bool:
        """描述玩家是否存活。
        """
        return self.player.living

    @property
    def door(self) -> 'GameObject|None':
        """距离玩家最近的且玩家未穿过的门。
        """
        for door in self.doors:
            if door.right >= self.player.left:
                return door
        return None

    def create_door(self) -> GameObject:
        """随机初始化门。

        Returns
        -------
        GameObject
            屏幕右侧随机位置的门。
        """
        door = GameObject(
            cx=self.screen_size[0]+self.door_size[0]/2,
            cy=random.randint(
                self.door_size[1]/2,
                self.screen_size[1]-self.door_size[1]/2
            ),
            w=self.door_size[0],
            h=self.door_size[1],
            sx=-self.speed,
            imgs=[None, ] if self.without_screen else [
                pygame.image.load(
                    './assets/textures/door.gif'
                ).convert_alpha(),
            ],
            img_cd=2
        )
        return door

    def draw(self, surface: 'pygame.Surface'):
        """绘制游戏帧。

        Parameters
        ----------
        surface : pygame.Surface
            pygame屏幕
        """
        if not self.player.living:
            return
        surface.fill([86, 92, 66])
        self.player.img_grow()
        surface.blit(
            pygame.transform.scale(
                self.player.img,
                (self.player.width, self.player.height)
            ),
            (self.player.left, self.player.top)
        )
        for door in self.doors:
            surface.blit(
                pygame.transform.scale(door.img, (door.width, door.top)),
                (door.left, 0)
            )
            surface.blit(
                pygame.transform.scale(
                    door.img,
                    (door.width, self.screen_size[1]-door.bottom)
                ),
                (door.left, door.bottom)
            )

    @staticmethod
    def __shot(door: Box, player: Box, screen_size: 'tuple[int,int]', speed_scale: int) -> 'list[float]':
        return [
            (door.right-player.left)/screen_size[0],
            (player.y-door.y)/screen_size[-1],
            # (player.y-screen_size[-1]/2)/screen_size[-1]/2,
            player.speed_y/speed_scale,
        ]

    def shot(self) -> 'list[float]':
        """组装并返回当前游戏环境状态。

        Returns
        -------
        list[float]
            模型所需的多元组。
        """
        return Game.__shot(
            self.door,
            self.player,
            [self.door_distance*self.speed, self.screen_size[-1]],
            self.max_falling_speed,
        )

    def step(self, jump: 'bool|int|float' = False):
        """游戏步进。

        Parameters
        ----------
        jump : bool, optional
            玩家是否跳跃, by default False
        """
        # 玩家必须存活才能继续游戏
        if not self.player.living:
            return

        if self.time % self.door_distance == 0 \
                or not (self.doors and len(self.doors)):
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
            door.move()
        door = self.door
        living = 0 < self.player.y < self.screen_size[1] \
            and not is_intersect(self.player, door)
        self.player.move(None, -self.jump_force if jump else self.g)
        if jump:
            self.player.speed_y = min(0, self.player.speed_y)

        self.player.living = living
        # 判断玩家和门存活
        if door.living and self.player.left >= door.right:
            door.living = False
            self.score += 1


if __name__ == "__main__":
    SCREEN_SIZE = (800, 600)
    FPS = 20
    GAME_CONFIG = {
        'screen_size': SCREEN_SIZE,
        'speed': 10,
        'jump_force': 3,
        'g': 2,
        'door_distance': 60,
    }
    pygame.init()  # 初始化
    screen = pygame.display.set_mode(SCREEN_SIZE)
    fcclock = pygame.time.Clock()  # 创建一个时间对象
    game = Game(**GAME_CONFIG)
    data = []
    while True:  # 循环，直到接收到窗口关闭事件
        for event in pygame.event.get():  # 处理事件
            if event.type == pygame.QUIT:  # 接收到窗口关闭事件
                pygame.quit()  # 退出
                sys.exit()
        keys = pygame.key.get_pressed()
        if keys[pygame.K_ESCAPE]:
            pygame.quit()
            break
        elif keys[pygame.K_SPACE] or keys[pygame.K_UP]:
            action = True
        else:
            action = False
        game.step(action)
        reward = game.playing
        pygame.display.set_caption(f'SCORE: {game.score}')  # 设置窗口标题
        game.draw(screen)
        fcclock.tick(FPS)  # 卡时间
        pygame.display.update()
        print(game.score)
        if not game.playing:
            game = Game(**GAME_CONFIG)
