# -*- coding: utf-8 -*-
"""输出打印工具模块。
"""
import random
from typing import Iterable

import matplotlib.pyplot as plt
from mpl_toolkits.axisartist.parasite_axes import HostAxes, ParasiteAxes


def print_bar(epoch, epochs, etc=None, bar_size=50):
    """打印进度条。

    Parameters
    ----------
    epoch : int
        当前进度
    epochs : int
        总进度
    etc : Any, optional
        打印后缀, by default None
    bar_size : int, optional
        进度条长度, by default 50
    """
    process = bar_size*epoch/epochs
    process = int(process+(int(process) < process))
    strs = [
        f"Epoch {epoch}/{epochs}",
        f" |\033[1;30;47m{' ' * process}\033[0m{' ' * (bar_size-process)}| ",
    ]
    if etc is not None:
        strs.append(str(etc))
    if epoch:
        strs.insert(0, "\033[A")
    print("".join(strs)+"    ")


def hsv_to_rgb(h, s, v):
    c = v*s
    x = c*(1 - abs(int(h/60) % 2-1))
    m = v-c
    if 0 <= h < 60:
        r, g, b = c, x, 0
    elif 60 <= h < 120:
        r, g, b = x, c, 0
    elif 120 <= h < 180:
        r, g, b = 0, c, x
    elif 180 <= h < 240:
        r, g, b = 0, x, c
    elif 240 <= h < 300:
        r, g, b = x, 0, c
    elif 300 <= h < 360:
        r, g, b = c, 0, x
    r, g, b = r+m, g+m, b+m
    return r, g, b


def plot(data: 'dict[str,Iterable]', axis_distance: int = 40, x_label: str = None, colors: Iterable = None, padding: float = .1):
    """多曲线展示

    Parameters
    ----------
    data : dict[str,Iterable]
        待展示数据
    axis_distance : int, optional
        坐标轴间距, by default 40
    x_label : str, optional
        横轴标签, by default None
    colors : Iterable, optional
        候选色彩，若色彩数量不足以显示曲线则分配不到色彩的曲线用随机颜色, by default None
    """
    fig = plt.figure(1)
    ax_cof = HostAxes(fig, [padding, padding, 1 -
                      padding * 2, 1 - padding * 2])
    ax_cof.axis['right'].set_visible(False)
    ax_cof.axis['top'].set_visible(False)
    color_list = [
        hsv_to_rgb(
            h, random.random()*.2+.6, random.random()*.2+.7
        ) for h in range(0, 360, 30)
    ]
    random.shuffle(color_list)
    color_list = [ele for ele in colors or []]+color_list
    random_color_count = len(data.keys())-len(color_list)
    if random_color_count > 0:
        color_list += [
            hsv_to_rgb(
                random.randint(0, 360-1),
                random.random()*.2+.6,
                random.random()*.2+.7
            ) for _ in range(random_color_count)
        ]
    color_list.reverse()
    index = 0
    for label, value in data.items():
        color = color_list.pop()
        if index:
            axe = ParasiteAxes(ax_cof, sharex=ax_cof)
            axe.set_ylabel(label)
            ax_cof.parasites.append(axe)
            axisline = axe.get_grid_helper().new_fixed_axis
            key_name = '?'+label
            axe.axis[key_name] = axisline(
                loc='right',
                axes=axe,
                offset=((index-1)*axis_distance, 0)
            )
            axe.plot(value, label=label, color=color)
            axe.axis[key_name].label.set_color(color)
            axe.axis[key_name].major_ticks.set_color(color)
            axe.axis[key_name].major_ticklabels.set_color(color)
            axe.axis[key_name].line.set_color(color)
        else:
            if x_label:
                ax_cof.set_xlabel(x_label)
            ax_cof.set_ylabel(label)
            ax_cof.plot(value, label=label, color=color)
        index += 1
    fig.add_axes(ax_cof)
    ax_cof.legend()
    plt.show()


if __name__ == "__main__":
    plot(
        data={
            'temperature': [i*30 for i in [0.007282, 0.04509, 0.0616, 0.06987, 0.07256, 0.07008, 0.10462, 0.13526, 0.14477, 0.14762, 0.18293, 0.21886, 0.2164, 0.22928, 0.2664, 0.28604, 0.2816, 0.37823, 0.37576, 0.4199, 0.4219, 0.4897, 0.5545, 0.5919, 0.6012, 0.6377, 0.6545, 0.6836, 0.6871, 0.7081, 0.7369, 0.768, 0.782, 0.79, 0.8052, 0.8225, 0.8385, 0.8674, 0.897, 0.8927]],
        },
        # colors=['#1E90FF','#8470FF','#87CEEB']
    )
