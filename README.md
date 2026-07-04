# 三分类 EEG 认知障碍检测系统

基于 MetaBCI 平台的抑郁症认知障碍三分类系统（HC / D-CI / D-NCI）。

## 目录结构

```
metabci_3class/
├── metabci_3class/                    # Python包
│   ├── __init__.py
│   ├── config.py                      # 全局配置
│   │
│   ├── brainda/                       # 数据处理模块
│   │   ├── datasets/
│   │   │   └── frontal_dataset.py     # FrontalEEGDataset
│   │   ├── paradigms/
│   │   │   └── resting_state.py       # RestingStateParadigm
│   │   └── algorithms/
│   │       ├── feature_extraction/
│   │       │   └── frontal_features.py    # PLI(75)+RP(30)+FE(30)=135维
│   │       ├── training/
│   │       │   └── three_class_trainer.py # TabPFN三分类训练器
│   │       └── utils/
│   │
│   ├── brainflow/                     # 实时数据流模块
│   │   ├── lsl_adapter.py             # LSL实时采集适配器
│   │   ├── online_feature_pipeline.py # 在线滑动窗口特征管道
│   │   ├── online_preprocessing.py    # 在线预处理
│   │   ├── ring_buffer.py             # 环形缓冲区
│   │   ├── classifier_manager.py      # 分类器管理+模型热更新
│   │   └── devices/                   # WiFi EEG设备采集
│   │       ├── config.py              # DeviceConfig
│   │       ├── device_adapter.py      # WiFi/TCP/UDP三协议适配器
│   │       ├── packet_decoder.py      # OpenBCI 33字节包解码
│   │       ├── acquisition_controller.py
│   │       ├── data_saver.py
│   │       ├── ring_buffer.py
│   │       ├── simulator.py
│   │       └── utils.py
│   │
│   ├── brainstim/                     # 刺激呈现模块
│   │   ├── framework/
│   │   │   └── resting_framework.py   # RestingExperiment
│   │   ├── paradigm/
│   │   │   └── resting_paradigm.py    # RestingState + resting_paradigm
│   │   └── utils/
│   │
│   └── utils/
│       └── data_loader.py             # EDF扫描+名单匹配
│
├── demos/                             # 运行脚本
│   ├── run_3class.py                  # 离线训练主程序
│   ├── run_tabpfn.py                  # TabPFN训练脚本
│   ├── run_collection.py              # 静息态采集主程序
│   ├── visualization.py               # 特征可视化
│   └── check.py                       # 环境检查
│
├── eeg_gui/                           # Java GUI
│   ├── src/                           # Java源码
│   │   ├── EEGMonitor.java            # 主窗口
│   │   ├── ControlTabs.java           # 控制面板
│   │   ├── EEGCore.java               # RingBuffer+特征+分类
│   │   ├── WaveformPanel.java         # 波形面板
│   │   ├── SpectrumPanel.java         # 频谱面板
│   │   ├── EnergyPanel.java           # 能量面板
│   │   ├── LogPanel.java              # 日志面板
│   │   ├── StatsBar.java              # 状态栏
│   │   ├── EDFReader.java             # EDF读取
│   │   └── EDFWriter.java             # EDF写入
│   ├── lib/                           # Java库 (放lsl-java.jar)
│   ├── eeg_wifi_acq/                  # WiFi采集模块(eeg_bridge.py依赖)
│   ├── eeg_bridge.py                  # Python-Java数据桥接
│   ├── wifi_relay.py                  # WiFi中继
│   ├── decode.py                      # OpenBCI包解码
│   └── model_export.py                # 模型蒸馏导出
│
├── data/                              # 数据
│   └── features.csv                   # 135维特征CSV
│
├── results_3class/                    # 训练输出(运行时生成)
│   ├── models/                        # 模型文件
│   └── figures/                       # 可视化图表
│
├── feature_plots/                     # 特征分析图
├── setup.py
├── requirements.txt
└── README.md
```

## MetaBCI 依赖

本项目的 `brainda`、`brainstim` 模块继承自 MetaBCI 官方仓库，使用以下官方组件：

| 模块 | 官方组件 | 使用位置 |
|------|---------|---------|
| Brainda | BaseDataset, upper_ch_names | frontal_dataset.py |
| Brainda | generate_filterbank | frontal_features.py |
| Brainda | set_random_seeds, EnhancedStratifiedKFold | three_class_trainer.py |
| Brainstim | Experiment | resting_framework.py |
| Brainstim | VisualStim, paradigm, NeuroScanPort | resting_paradigm.py |

安装官方依赖：`pip install metabci`

## 运行方式

### 离线训练

```bash
# 从项目根目录运行
python demos/run_3class.py

# 或单独训练TabPFN
python demos/run_tabpfn.py
```

### 静息态采集

```bash
python demos/run_collection.py
```

### Java GUI

```bash
cd eeg_gui/src
javac -encoding UTF-8 *.java
java -cp ".;../lib/lsl-java.jar" EEGMonitor
```

## 配置

修改 `metabci_3class/config.py` 中的数据路径：

```python
HC_DIR = r"D:\metabci_3class\用户上传\ALL_Noraml_Data"
DEP_DIR = r"D:\metabci_3class\用户上传"
LABEL_EXCEL = r"D:\metabci_3class\用户上传\名单.xlsx"
```

## 技术规格

- 导联：前额6导联 (FP1, FP2, F3, F4, F7, F8)
- 采样率：250 Hz (离线) / 500 Hz (在线)
- 特征：135维 (PLI 75 + RP 30 + FE 30)
- 分类：TabPFN三分类 (HC / D-CI / D-NCI)
- 频率带：θ(4-8) α_low(8-10) α_high(10-13) β_low(13-20) β_high(20-30) Hz
