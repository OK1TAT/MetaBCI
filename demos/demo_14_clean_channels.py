# -*- coding: utf-8 -*-
"""
表2-1: EDF通道名智能清洗
运行: python demo_14_clean_channels.py
"""
def clean_channel_names(ch_names):
    new = []
    for name in ch_names:
        n = name.strip()
        if n.upper().startswith("EEG "):
            n = n[4:]
        if n.upper().endswith("-REF"):
            n = n[:-4]
        if n.upper().startswith("POL "):
            n = n[4:]
        new.append(n.strip())
    return new

def demo():
    tests = ["EEG FP1-REF", "EEG FP2-REF", "EEG F3-REF", "EEG F4-REF",
             "EEG F7-REF", "EEG F8-REF", "EEG Fp1-Ref", "EEG FP2-Ref",
             "POL E", "POL $A1", "POL X1", "EEG CZ-REF"]
    print("通道名清洗演示:")
    for name in tests:
        cleaned = clean_channel_names([name])
        print("  %-16s -> %s" % (name, cleaned[0]))
    print("\n✓ Demo完成: clean_channel_names")

if __name__ == "__main__":
    demo()
