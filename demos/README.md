# 功能点Demo脚本汇总

## 表1: 基础功能Demo（13个）
| Demo | 序号 | 功能点 |
|------|------|--------|
| demo_01_basedataset.py | 1 | BaseDataset基类 |
| demo_02_channel_names.py | 2 | upper_ch_names |
| demo_03_filterbank.py | 3 | generate_filterbank |
| demo_04_hilbert.py | 4 | TimeFrequencyAnalysis.fun_hilbert |
| demo_05_model_selection.py | 5,6 | set_random_seeds + EnhancedStratifiedKFold |
| demo_06_experiment.py | 7 | Experiment |
| demo_07_visualstim.py | 8,9 | VisualStim + paradigm |
| demo_08_neuroscan.py | 10 | NeuroScanPort |
| demo_09_ringbuffer_official.py | 11 | RingBuffer |
| demo_10_frontal_dataset.py | 12 | FrontalEEGDataset |
| demo_11_resting_paradigm.py | 13 | RestingStateParadigm |
| demo_12_resting_experiment.py | 14 | RestingExperiment |
| demo_13_resting_stimulus.py | 15 | RestingState |

## 表2: 创新功能Demo（19个）
| Demo | 序号 | 功能点 |
|------|------|--------|
| demo_14_clean_channels.py | 1 | EDF通道名清洗 |
| demo_15_feature_extraction.py | 2-5 | 135维特征提取 |
| demo_16_tabpfn_trainer.py | 6,7 | TabPFN训练 + 类别平衡 |
| demo_17_data_loader.py | 8 | 被试信息构建 |
| demo_18_lsl_adapter.py | 9-11 | LSL适配器 |
| demo_19_online_pipeline.py | 12-15 | 在线特征管道 |
| demo_20_ring_buffer.py | 16,17 | 环形缓冲区 |
| demo_21_preprocessing.py | 18-22 | 在线预处理 |
| demo_22_classifier_manager.py | 23-25 | 分类器管理 |
| demo_23_device_adapter.py | 26,27 | 设备适配器 |
| demo_24_packet_decoder.py | 28 | 包解码器 |
| demo_25_acquisition.py | 29 | 采集控制 |
| demo_26_device_config.py | 30 | 设备配置 |
| demo_27_data_saver.py | 31 | 数据保存 |
| demo_28_simulator.py | 32 | 设备模拟器 |
| demo_29_bridge.py | 43 | Python-Java桥接 |
| demo_30_wifi_relay.py | 44 | WiFi中继 |
| demo_31_decode.py | 45 | 数据包解码 |
| demo_32_model_export.py | 46 | 模型导出 |

## Java GUI（表2第33-42项）
```bash
cd eeg_gui/src && javac -encoding UTF-8 *.java && java EEGMonitor
```

## 已有Demos（表2第47-51项）
直揥运行demos目录下: run_3class.py / run_tabpfn.py / run_collection.py / visualization.py / check.py
