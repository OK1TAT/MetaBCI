# -*- coding: utf-8 -*-
"""
静息态采集范式

Brainstim 功能使用:
    2. VisualStim    - 视觉刺激基类，继承实现 RestingState
    3. paradigm      - 范式流程控制函数
    4. NeuroScanPort - 并口/串口标签通信

范式流程:
    开始界面 → 闭眼提示(2s) → 静息采集(倒计时) → 结束提示(2s)
"""

import numpy as np
from psychopy import core, event, visual

# Brainstim 功能2: VisualStim
from metabci.brainstim.paradigm import VisualStim

# Brainstim 功能4: NeuroScanPort
from metabci.brainstim.utils import NeuroScanPort

from metabci_3class.config import REST_DURATION


class RestingState(VisualStim):
    """
    静息态 EEG 刺激类

    继承自 VisualStim，实现静息态采集的视觉界面。
    """

    def __init__(self, win, colorSpace="rgb", allowGUI=True):
        super().__init__(win=win, colorSpace=colorSpace, allowGUI=allowGUI)
        self.rest_duration = REST_DURATION
        self.colorSpace = colorSpace

    def config_prompt(self, text_color=[1, 1, 1], height=40):
        """配置提示文字"""
        self.prompt_text = visual.TextStim(
            win=self.win,
            text="",
            color=text_color,
            height=height,
            units="pix",
            pos=(0, 50),
            colorSpace=self.colorSpace,
            bold=True,
            autoLog=False
        )

        self.countdown_text = visual.TextStim(
            win=self.win,
            text="",
            color=[0.5, 0.5, 0.5],
            height=height * 0.8,
            units="pix",
            pos=(0, -30),
            colorSpace=self.colorSpace,
            autoLog=False
        )

    def config_duration(self, rest_duration=REST_DURATION):
        """配置采集时长"""
        self.rest_duration = rest_duration

    def config_response(self):
        """配置反馈界面（兼容官方接口）"""
        pass


def resting_paradigm(VSObject, win, bg_color, fps,
                     port_addr=None, nrep=1,
                     lsl_source_id=None, online=False, **kwargs):
    """
    静息态 EEG 采集范式流程

    Brainstim 功能3: 仿照 paradigm() 函数风格

    Args:
        VSObject: RestingState 实例
        win: PsychoPy 窗口
        bg_color: 背景色
        fps: 刷新率
        port_addr: 并口地址
        nrep: block 重复次数
        lsl_source_id: LSL 数据流 ID
        online: 是否在线模式
    """
    # Brainstim 功能4: 标签通信
    if port_addr is not None:
        port = NeuroScanPort(port_addr, use_serial=False)
    else:
        port = None

    duration = VSObject.rest_duration

    for rep in range(nrep):
        print(f"\n===== Block {rep + 1}/{nrep} =====")

        # 阶段1: 闭眼提示 (2s)
        VSObject.prompt_text.text = "请闭眼，保持放松"
        VSObject.prompt_text.draw()
        VSObject.countdown_text.text = f"Block {rep + 1}/{nrep}"
        VSObject.countdown_text.draw()
        win.flip()

        if port:
            win.callOnFlip(port.setData, 1)  # 标签1: 开始

        core.wait(2.0)

        # 阶段2: 静息采集
        start_time = core.getTime()
        while core.getTime() - start_time < duration:
            remaining = int(duration - (core.getTime() - start_time))
            VSObject.prompt_text.text = ""
            VSObject.countdown_text.text = f"{remaining}"
            VSObject.countdown_text.draw()
            win.flip()

            if event.getKeys(keyList=["escape", "q"]):
                if port:
                    port.setData(0)
                return
            core.wait(0.5)

        if port:
            win.callOnFlip(port.setData, 2)  # 标签2: 结束

        # 阶段3: 结束提示
        VSObject.prompt_text.text = "采集完成，请睁眼"
        VSObject.countdown_text.text = ""
        VSObject.prompt_text.draw()
        win.flip()
        core.wait(2.0)

        if port:
            port.setData(0)
