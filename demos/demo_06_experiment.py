# -*- coding: utf-8 -*-
"""
表1-7: Experiment 实验框架基类（官方 metabci）
运行: python demo_06_experiment.py
"""
try:
    from metabci.brainstim.framework import Experiment
    HAS = True
except ImportError:
    HAS = False
    print("[!] metabci 未安装")

def demo():
    print("Experiment 关键参数:")
    print("  win_size, is_fullscr, screen_id, bg_color_warm, fps")
    if HAS:
        try:
            exp = Experiment(win_size=[800,600], is_fullscr=False, screen_id=0, bg_color_warm=[-1,-1,-1])
            print("Experiment 创建成功")
        except Exception as e:
            print("需要显示器环境:", e)
    print("\n✓ Demo完成: Experiment 实验框架")

if __name__ == "__main__":
    demo()
