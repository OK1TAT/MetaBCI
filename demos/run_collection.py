# -*- coding: utf-8 -*-
"""
静息态采集范式 - 独立运行入口

使用方式:
    python -m metabci_3class.brainstim.paradigm.resting_paradigm_main

或使用脚本:
    python run_collection.py
"""

import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from metabci_3class.config import REST_DURATION, PORT_ADDR
from metabci_3class.brainstim.framework import RestingExperiment
from metabci_3class.brainstim.paradigm import RestingState, resting_paradigm


def main():
    """静息态采集主入口"""
    print("="*60)
    print("静息态 EEG 采集范式")
    print("基于 MetaBCI Brainstim + Brainda")
    print("="*60)

    # 创建实验框架
    ex = RestingExperiment()
    win = ex.get_window()

    # 获取刷新率
    measured_fps = win.getActualFrameRate()
    fps = int(measured_fps) if measured_fps else 60

    # 创建刺激对象
    stim = RestingState(win=win)
    stim.config_prompt(text_color=[1, 1, 1], height=40)
    stim.config_duration(rest_duration=REST_DURATION)
    stim.config_response()

    # 注册范式
    ex.register_paradigm(
        "静息态采集",

        resting_paradigm,
        VSObject=stim,
        bg_color=[-1, -1, -1],
        fps=fps,
        port_addr=PORT_ADDR,
        nrep=1,
        lsl_source_id=None,
        online=False
    )

    print("\n按 ↑↓ 选择范式，Enter 开始，q 退出\n")

    # 运行
    ex.run()


if __name__ == "__main__":
    main()
