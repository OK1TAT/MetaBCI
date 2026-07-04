# -*- coding: utf-8 -*-
"""
静息态采集实验框架

Brainstim 功能使用:
    1. Experiment - 实验框架、开始界面、窗口管理

Brainda 功能使用:
    5. BaseDataset       - 采集后数据接口
    8. set_random_seeds  - 确保可复现
"""

import os
import numpy as np
from psychopy import monitors

# Brainstim 功能1: Experiment
from metabci.brainstim.framework import Experiment

# Brainda 功能8: set_random_seeds
from metabci.brainda.algorithms.utils.model_selection import set_random_seeds

from metabci_3class.config import SCREEN_SIZE, RANDOM_SEED


class RestingExperiment:
    """
    静息态采集实验框架

    封装 MetaBCI Experiment，提供简化的创建流程。
    """

    def __init__(self, screen_size=SCREEN_SIZE, is_fullscr=True):
        """
        Args:
            screen_size: 屏幕分辨率 [width, height]
            is_fullscr: 是否全屏
        """
        self.screen_size = screen_size
        self.is_fullscr = is_fullscr

        # Brainda: 设置随机种子
        set_random_seeds(RANDOM_SEED)

        # 创建 Experiment
        self.ex = self._create_experiment()

    def _create_experiment(self):
        """创建 MetaBCI Experiment 实例"""
        # Monitor 配置
        mon = monitors.Monitor(
            name="primary_monitor",
            width=59.6,
            distance=60,
            verbose=False
        )
        mon.setSizePix(self.screen_size)
        mon.save()

        # Brainstim 功能1: Experiment
        ex = Experiment(
            monitor=mon,
            bg_color_warm=np.array([-1, -1, -1]),
            screen_id=0,
            win_size=np.array(self.screen_size),
            is_fullscr=self.is_fullscr,
            record_frames=False,
            disable_gc=False,
            process_priority="normal",
            use_fbo=False
        )

        return ex

    def get_window(self):
        """获取 PsychoPy 窗口"""
        return self.ex.get_window()

    def register_paradigm(self, name, func, **kwargs):
        """注册范式"""
        self.ex.register_paradigm(name, func, **kwargs)

    def run(self):
        """运行实验 - 跳过帧率严格检查"""
        # 调用原始 run 逻辑，但 warmup 用 strict=False 避免帧率检查失败
        self.ex.initEvent()
        self.ex.warmup(strict=False)  # ← 关键：strict=False 跳过帧率检查
        win = self.ex.get_window()

        # 简化版运行循环
        from psychopy import core, event, logging
        trialClock = core.Clock()
        pindex = 0

        try:
            while True:
                t = trialClock.getTime()
                keys = event.getKeys(keyList=["q", "up", "down", "return"])

                if "q" in keys:
                    break

                names = list(self.ex.paradigms.keys())
                if names:
                    if "up" in keys:
                        pindex = (pindex - 1) % len(names)
                    elif "down" in keys:
                        pindex = (pindex + 1) % len(names)
                    self.ex.current_paradigm = names[pindex]

                if "return" in keys:
                    old_color = win.color
                    logging.warning("Start paradigm {}".format(self.ex.current_paradigm))
                    self.ex.paradigms[self.ex.current_paradigm](win=win)
                    logging.warning("Finish paradigm {}".format(self.ex.current_paradigm))
                    win.color = old_color

                self.ex.update_startup()
                win.flip()

        except Exception as e:
            print("Error Info:", e)
            raise e
        finally:
            win.close()
            self.ex.closeEvent()
