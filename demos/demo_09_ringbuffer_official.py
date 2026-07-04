# -*- coding: utf-8 -*-
"""
表1-11: RingBuffer 环形缓冲区基类（官方 metabci）
运行: python demo_09_ringbuffer_official.py
"""
import numpy as np
from collections import deque
try:
    from metabci.brainflow.amplifiers import RingBuffer
    HAS = True
except ImportError:
    HAS = False
    print("[!] metabci 未安装, 使用deque模拟")

def demo():
    n_ch, buf_size = 6, 1000
    print("环形缓冲区: %d通道, 容量%d" % (n_ch, buf_size))
    if HAS:
        buf = RingBuffer(buf_size)
        for i in range(50):
            buf.put(np.random.randn(n_ch))
        print("写入50帧, 读取成功")
    else:
        buf = deque(maxlen=buf_size)
        for i in range(50):
            buf.append(np.random.randn(n_ch))
        print("写入50帧, 读取%d帧" % len(buf))
    print("\n✓ Demo完成: RingBuffer")

if __name__ == "__main__":
    demo()
