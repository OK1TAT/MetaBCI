/*
 * EEG 实时采集与分析系统 v3.0 (Java) - 主窗口框架
 * 模块化设计：EEGCore / ControlTabs / WaveformPanel / SpectrumPanel / EnergyPanel / StatsBar / LogPanel
 * 编译运行: javac -encoding UTF-8 *.java && java EEGMonitor
 */
import javax.swing.*;
import javax.swing.filechooser.FileNameExtensionFilter;
import java.awt.*;
import java.io.*;

public class EEGMonitor extends JFrame {
    private static final int NUM_CH = 16;
    private static final int BUF_SIZE = 4000;

    // 核心数据
    private final EEGCore.RingBuffer buffer = new EEGCore.RingBuffer(NUM_CH, BUF_SIZE);
    private boolean connected = false;
    private boolean paused = false;
    private boolean isSimulating = false;
    private long totalSamples = 0;
    private int frameSeq = 0;
    private long startTimeMs = 0;
    private double sampleRate = 250;
    private boolean classifying = false;

    // 模块
    private ControlTabs controlTabs;
    private WaveformPanel waveformPanel;
    private SpectrumPanel spectrumPanel;
    private EnergyPanel energyPanel;
    private StatsBar statsBar;
    private LogPanel logPanel;

    // 波形控制
    private JSlider zoomSlider;
    private JComboBox<String> filterCombo;
    private JComboBox<String> specChCombo;

    // WiFi设备连接 (Python全权负责接收+解码, Java只读取结果)
    private Process wifiProcess = null;
    private Thread wifiThread = null;
    private volatile boolean wifiRunning = false;

    // 定时器
    private javax.swing.Timer refreshTimer, simTimer, classTimer, clockTimer;

    public static void main(String[] args) {
        // 全局UI设置：不用系统LAF，避免颜色被覆盖
        Font f = new Font("Microsoft YaHei", Font.PLAIN, 13);
        Color darkText = new Color(33, 33, 33);
        Color panelBg = new Color(245, 248, 252);

        String[] keys = {
            "Label.foreground", "TextField.foreground", "TextArea.foreground",
            "TextPane.foreground", "ComboBox.foreground", "CheckBox.foreground",
            "RadioButton.foreground", "Panel.foreground", "TitledBorder.titleColor"
        };
        for (String k : keys) UIManager.put(k, darkText);
        UIManager.put("Panel.background", panelBg);
        UIManager.put("TextField.background", Color.WHITE);
        UIManager.put("ComboBox.background", Color.WHITE);

        SwingUtilities.invokeLater(() -> {
            EEGMonitor app = new EEGMonitor();
            app.setVisible(true);
        });
    }

    public EEGMonitor() {
        super("EEG 实时采集与分析系统 v3.0");
        setDefaultCloseOperation(EXIT_ON_CLOSE);
        setSize(1400, 900);
        setLocationRelativeTo(null);
        setLayout(new BorderLayout(2, 2));
        getContentPane().setBackground(new Color(240, 244, 248));

        buildUI();
        setupCallbacks();
        startTimers();
        logPanel.log("INFO", "系统启动完成 - 模块化架构");
    }

    private void buildUI() {
        // 顶部工具栏
        JPanel toolbar = new JPanel(new FlowLayout(FlowLayout.LEFT, 12, 6));
        toolbar.setBackground(new Color(13, 71, 161));
        JLabel title = new JLabel("EEG 实时采集与分析系统 v3.0 (Java)");
        title.setFont(new Font("Microsoft YaHei", Font.BOLD, 16));
        title.setForeground(Color.WHITE);
        toolbar.add(title);
        toolbar.add(Box.createHorizontalStrut(30));
        // LED
        JPanel led = new JPanel() {
            @Override
            protected void paintComponent(Graphics g) {
                super.paintComponent(g);
                Graphics2D g2 = (Graphics2D) g.create();
                g2.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
                g2.setColor(connected ? new Color(76, 175, 80) : new Color(158, 158, 158));
                g2.fillOval(2, 2, 16, 16);
                g2.setColor(new Color(255, 255, 255, 80));
                g2.fillOval(4, 3, 6, 6);
                g2.dispose();
            }
        };
        led.setOpaque(false);
        led.setPreferredSize(new Dimension(22, 22));
        toolbar.add(led);
        add(toolbar, BorderLayout.NORTH);

        // 左侧：Tab控制面板
        controlTabs = new ControlTabs();
        // 通道复选框联动：变更时刷新所有绘图面板
        controlTabs.addChannelListener(e -> refreshEnabledChannels());
        JScrollPane leftScroll = new JScrollPane(controlTabs);
        leftScroll.setBorder(null);
        leftScroll.setPreferredSize(new Dimension(380, 0));
        leftScroll.getVerticalScrollBar().setUnitIncrement(16);

        // 右侧：波形 + 频谱 + 能量 + 控制
        JPanel rightArea = new JPanel(new BorderLayout(0, 3));
        rightArea.setBackground(new Color(240, 244, 248));

        // 波形面板 + 控制条
        JPanel waveWrapper = new JPanel(new BorderLayout());
        waveWrapper.setBorder(BorderFactory.createCompoundBorder(
            BorderFactory.createTitledBorder(BorderFactory.createLineBorder(new Color(25, 118, 210)),
                "实时波形",
                javax.swing.border.TitledBorder.LEFT, javax.swing.border.TitledBorder.TOP,
                new Font("Microsoft YaHei", Font.BOLD, 13), new Color(25, 118, 210)),
            BorderFactory.createEmptyBorder(2, 2, 2, 2)
        ));
        waveformPanel = new WaveformPanel(buffer, controlTabs.getEnabledChannels());
        waveWrapper.add(waveformPanel, BorderLayout.CENTER);

        // 波形控制条
        JPanel waveCtrl = new JPanel(new FlowLayout(FlowLayout.LEFT, 10, 3));
        waveCtrl.setBackground(new Color(245, 248, 252));
        JLabel zl = new JLabel("缩放:"); zl.setForeground(new Color(33,33,33));
        waveCtrl.add(zl);
        zoomSlider = new JSlider(100, 2000, 500);
        zoomSlider.setPreferredSize(new Dimension(150, 20));
        zoomSlider.addChangeListener(e -> waveformPanel.setSamplesPerScreen(zoomSlider.getValue()));
        waveCtrl.add(zoomSlider);
        JButton pauseBtn = new JButton("暂停显示");
        pauseBtn.setFont(new Font("Microsoft YaHei", Font.PLAIN, 12));
        pauseBtn.addActionListener(e -> {
            paused = !paused;
            waveformPanel.setPaused(paused);
            pauseBtn.setText(paused ? "恢复显示" : "暂停显示");
        });
        waveCtrl.add(pauseBtn);
        JLabel fl = new JLabel("滤波:"); fl.setForeground(new Color(33,33,33));
        waveCtrl.add(fl);
        filterCombo = new JComboBox<>(new String[]{"基础滤波", "+低通平滑", "+50Hz陷波增强"});
        filterCombo.addActionListener(e -> waveformPanel.setFilterMode(filterCombo.getSelectedIndex()));
        waveCtrl.add(filterCombo);
        waveWrapper.add(waveCtrl, BorderLayout.SOUTH);

        // 频谱 + 能量（左右排列）
        JPanel analysisPanel = new JPanel(new GridLayout(1, 2, 3, 0));
        analysisPanel.setOpaque(false);

        // 频谱
        JPanel specWrapper = new JPanel(new BorderLayout());
        specWrapper.setBorder(BorderFactory.createCompoundBorder(
            BorderFactory.createTitledBorder(BorderFactory.createLineBorder(new Color(25, 118, 210)),
                "频谱分析 (PSD)",
                javax.swing.border.TitledBorder.LEFT, javax.swing.border.TitledBorder.TOP,
                new Font("Microsoft YaHei", Font.BOLD, 13), new Color(25, 118, 210)),
            BorderFactory.createEmptyBorder(2, 2, 2, 2)
        ));
        spectrumPanel = new SpectrumPanel(buffer, controlTabs.getEnabledChannels());
        specWrapper.add(spectrumPanel, BorderLayout.CENTER);
        JPanel specCtrl = new JPanel(new FlowLayout(FlowLayout.LEFT, 8, 2));
        specCtrl.setBackground(new Color(245, 248, 252));
        JLabel scl = new JLabel("通道:"); scl.setForeground(new Color(33,33,33));
        specCtrl.add(scl);
        String[] chOpts = new String[17];
        chOpts[0] = "全部叠加";
        String[] chLabels = {"FP1","FP2","F3","F4","C3","C4","P3","P4","O1","O2","F7","F8","T7","T8","P7","P8"};
        for (int i = 0; i < 16; i++) chOpts[i + 1] = chLabels[i];
        specChCombo = new JComboBox<>(chOpts);
        specChCombo.addActionListener(e -> {
            int idx = specChCombo.getSelectedIndex();
            spectrumPanel.setSelectedChannel(idx == 0 ? -1 : idx - 1);
        });
        specCtrl.add(specChCombo);
        specWrapper.add(specCtrl, BorderLayout.SOUTH);
        analysisPanel.add(specWrapper);

        // 能量
        JPanel energyWrapper = new JPanel(new BorderLayout());
        energyWrapper.setBorder(BorderFactory.createCompoundBorder(
            BorderFactory.createTitledBorder(BorderFactory.createLineBorder(new Color(25, 118, 210)),
                "通道能量分布 (RMS)",
                javax.swing.border.TitledBorder.LEFT, javax.swing.border.TitledBorder.TOP,
                new Font("Microsoft YaHei", Font.BOLD, 13), new Color(25, 118, 210)),
            BorderFactory.createEmptyBorder(2, 2, 2, 2)
        ));
        energyPanel = new EnergyPanel();
        energyWrapper.add(energyPanel, BorderLayout.CENTER);
        analysisPanel.add(energyWrapper);

        // 组装右侧：上下布局（上=波形，下=频谱+能量左右并排）
        JSplitPane rightSplit = new JSplitPane(JSplitPane.VERTICAL_SPLIT, waveWrapper, analysisPanel);
        rightSplit.setResizeWeight(0.55);
        rightSplit.setDividerSize(4);
        rightArea.add(rightSplit, BorderLayout.CENTER);

        // 主分割
        JSplitPane mainSplit = new JSplitPane(JSplitPane.HORIZONTAL_SPLIT, leftScroll, rightArea);
        mainSplit.setResizeWeight(0.25);
        mainSplit.setDividerLocation(380);
        mainSplit.setDividerSize(4);
        add(mainSplit, BorderLayout.CENTER);

        // 底部
        JPanel bottomPanel = new JPanel(new BorderLayout(3, 0));
        statsBar = new StatsBar();
        bottomPanel.add(statsBar, BorderLayout.NORTH);
        logPanel = new LogPanel();
        bottomPanel.add(logPanel, BorderLayout.CENTER);
        add(bottomPanel, BorderLayout.SOUTH);
    }

    private void setupCallbacks() {
        controlTabs.setCallbacks(
            // onConnect
            () -> {
                String proto = (String) controlTabs.protocolCombo.getSelectedItem();
                connected = true;
                totalSamples = 0;
                frameSeq = 0;
                startTimeMs = System.currentTimeMillis();
                controlTabs.connectBtn.setText("断开");
                controlTabs.connectBtn.setBackground(new Color(211, 47, 47));
                logPanel.log("INFO", "已连接: " + proto);
                if ("模拟器".equals(proto)) {
                    isSimulating = true;
                    startSimulator();
                } else if ("WiFi Shield".equals(proto)) {
                    // 启动 wifi_relay.py 真实设备接收
                    String host = controlTabs.addressField.getText().trim();
                    int port = Integer.parseInt(controlTabs.portField.getText().trim());
                    startWiFiConnection(host, port);
                }
                // TODO: TCP/UDP 直连接口
            },
            // onDisconnect
            () -> {
                connected = false;
                isSimulating = false;
                if (simTimer != null) simTimer.stop();
                stopWiFiConnection();
                controlTabs.connectBtn.setText("连  接");
                controlTabs.connectBtn.setBackground(new Color(25, 118, 210));
                logPanel.log("WARN", "已断开连接");
            },
            // onLoadData
            () -> {
                JFileChooser fc = new JFileChooser();
                fc.setFileFilter(new FileNameExtensionFilter("数据文件 (*.csv, *.json, *.edf)", "csv", "json", "edf"));
                if (fc.showOpenDialog(this) != JFileChooser.APPROVE_OPTION) return;
                try {
                    File f = fc.getSelectedFile();
                    String name = f.getName().toLowerCase();
                    if (name.endsWith(".json")) {
                        controlTabs.loadJSONData(f);
                    } else if (name.endsWith(".edf")) {
                        controlTabs.loadEDFData(f);
                    } else {
                        controlTabs.loadCSVData(f);
                    }
                    // 填充缓冲区
                    if (controlTabs.loadedData != null && controlTabs.loadedData[0].length > 0) {
                        buffer.clear();
                        int nSamples = controlTabs.loadedData[0].length;
                        boolean[] enabled = controlTabs.getEnabledChannels();
                        for (int s = 0; s < nSamples; s++) {
                            double[] sample = new double[NUM_CH];
                            for (int c = 0; c < NUM_CH; c++) {
                                if (c < controlTabs.loadedData.length && enabled[c]) {
                                    sample[c] = (s < controlTabs.loadedData[c].length) ? controlTabs.loadedData[c][s] : 0;
                                }
                            }
                            buffer.push(sample);
                        }
                        totalSamples = nSamples;
                        controlTabs.dataFileInfo.setText("已加载: " + f.getName() + " (" + nSamples + "样本)");
                        logPanel.log("INFO", "数据加载完成: " + f.getName());
                    }
                } catch (Exception ex) {
                    logPanel.log("ERROR", "加载数据失败: " + ex.getMessage());
                }
            },
            // onStart
            () -> {
                paused = false;
                waveformPanel.setPaused(false);
                logPanel.log("INFO", "采集已开始");
            },
            // onPause
            () -> {
                paused = true;
                waveformPanel.setPaused(true);
                logPanel.log("INFO", "采集已暂停");
            },
            // onStop
            () -> {
                paused = false;
                isSimulating = false;
                if (simTimer != null) simTimer.stop();
                logPanel.log("INFO", "采集已停止");
            },
            // onSaveCSV
            () -> {
                JFileChooser fc = new JFileChooser();
                String subject = controlTabs.subjectField.getText().trim();
                if (subject.isEmpty()) subject = "S001";
                fc.setSelectedFile(new File(subject + "_eeg_data.csv"));
                fc.setFileFilter(new FileNameExtensionFilter("CSV文件 (*.csv)", "csv"));
                if (fc.showSaveDialog(this) != JFileChooser.APPROVE_OPTION) return;
                File file = fc.getSelectedFile();
                if (!file.getName().toLowerCase().endsWith(".csv"))
                    file = new File(file.getAbsolutePath() + ".csv");

                try {
                    int n = Math.min(buffer.validSamples, BUF_SIZE);
                    try (PrintWriter pw = new PrintWriter(new OutputStreamWriter(
                            new FileOutputStream(file), java.nio.charset.StandardCharsets.UTF_8))) {
                        pw.println("# EEG Data - Subject: " + subject);
                        pw.println("# Sample Rate: " + (int) sampleRate + " Hz");
                        pw.println("# Channels: FP1,FP2,F3,F4,C3,C4,P3,P4,O1,O2,F7,F8,T7,T8,P7,P8");
                        pw.println("# Samples: " + n);
                        for (int s = 0; s < n; s++) {
                            StringBuilder sb = new StringBuilder();
                            for (int c = 0; c < NUM_CH; c++) {
                                if (c > 0) sb.append(",");
                                int idx = (buffer.writePos - n + s + BUF_SIZE) % BUF_SIZE;
                                sb.append(String.format("%.4f", buffer.data[c][idx]));
                            }
                            pw.println(sb.toString());
                        }
                        logPanel.log("INFO", "数据已保存为CSV: " + file.getName());
                    }
                } catch (Exception ex) {
                    logPanel.log("ERROR", "保存失败: " + ex.getMessage());
                }
            },
            // onSaveEDF
            () -> {
                JFileChooser fc = new JFileChooser();
                String subject = controlTabs.subjectField.getText().trim();
                if (subject.isEmpty()) subject = "S001";
                fc.setSelectedFile(new File(subject + "_eeg_data.edf"));
                fc.setFileFilter(new FileNameExtensionFilter("EDF文件 (*.edf)", "edf"));
                if (fc.showSaveDialog(this) != JFileChooser.APPROVE_OPTION) return;
                File file = fc.getSelectedFile();
                if (!file.getName().toLowerCase().endsWith(".edf"))
                    file = new File(file.getAbsolutePath() + ".edf");

                try {
                    int n = Math.min(buffer.validSamples, BUF_SIZE);
                    double[][] saveData = new double[NUM_CH][n];
                    String[] chLabels = {"FP1","FP2","F3","F4","C3","C4","P3","P4","O1","O2","F7","F8","T7","T8","P7","P8"};
                    for (int s = 0; s < n; s++) {
                        for (int c = 0; c < NUM_CH; c++) {
                            int idx = (buffer.writePos - n + s + BUF_SIZE) % BUF_SIZE;
                            saveData[c][s] = buffer.data[c][idx];
                        }
                    }
                    EDFWriter.write(file, saveData, sampleRate, chLabels);
                    logPanel.log("INFO", "数据已保存为EDF: " + file.getName());
                } catch (Exception ex) {
                    logPanel.log("ERROR", "保存失败: " + ex.getMessage());
                }
            },
            // onLoadModel
            () -> {
                JFileChooser fc = new JFileChooser();
                fc.setFileFilter(new FileNameExtensionFilter("模型文件 (*.json, *.joblib, *.pkl)", "json", "joblib", "pkl"));
                if (fc.showOpenDialog(this) != JFileChooser.APPROVE_OPTION) return;
                try {
                    controlTabs.loadModelJSON(fc.getSelectedFile());
                    if (controlTabs.modelWeights != null && controlTabs.modelLabels != null) {
                        controlTabs.setupProbBars();
                        controlTabs.modelInfoLabel.setText(String.format("维度: %d | 类别: %d | %s",
                            controlTabs.modelWeights[0].length, controlTabs.modelLabels.length,
                            String.join(", ", controlTabs.modelLabels)));
                        controlTabs.startClassBtn.setEnabled(true);
                        logPanel.log("INFO", "模型加载完成");
                    }
                } catch (Exception ex) {
                    logPanel.log("ERROR", "模型加载失败: " + ex.getMessage());
                }
            },
            // onStartClass
            () -> {
                classifying = true;
                controlTabs.startClassBtn.setEnabled(false);
                controlTabs.stopClassBtn.setEnabled(true);
                logPanel.log("INFO", "实时分类已启动 (每2秒)");
            },
            // onStopClass
            () -> {
                classifying = false;
                controlTabs.startClassBtn.setEnabled(true);
                controlTabs.stopClassBtn.setEnabled(false);
                logPanel.log("INFO", "实时分类已停止");
            }
        );
    }

    // 通道启用/禁用联动刷新
    private void refreshEnabledChannels() {
        boolean[] enabled = controlTabs.getEnabledChannels();
        waveformPanel.setEnabledChannels(enabled);
        spectrumPanel.setEnabledChannels(enabled);
        energyPanel.setEnabledChannels(enabled);
        waveformPanel.repaint();
        spectrumPanel.repaint();
        energyPanel.repaint();
    }

    // ==================== WiFi 设备连接 ====================
    // Python (eeg_bridge.py) 全权负责: 接收 + 解码
    // Java 只负责: 读行 → 画图, 零信号处理

    // 从 Python stdout 读取 CSV 行 (16通道 uV), 推入缓冲区
    private void readWifiLines(BufferedReader reader) throws IOException {
        String line;
        while (wifiRunning && (line = reader.readLine()) != null) {
            line = line.trim();
            if (line.isEmpty()) continue;
            String[] parts = line.split(",");
            if (parts.length < 16) continue;
            double[] sample = new double[NUM_CH];
            for (int i = 0; i < NUM_CH && i < parts.length; i++) {
                try { sample[i] = Double.parseDouble(parts[i]); }
                catch (NumberFormatException e) { sample[i] = 0; }
            }
            buffer.push(sample);
            totalSamples++;
        }
    }

    private void startWiFiConnection(String host, int localPort) {
        stopWiFiConnection();
        wifiRunning = true;
        wifiThread = new Thread(() -> {
            try {
                // 找 eeg_bridge.py (与 class 同目录)
                String dir = EEGMonitor.class.getProtectionDomain()
                    .getCodeSource().getLocation().getPath();
                java.io.File script = new java.io.File(new java.io.File(dir), "eeg_bridge.py");
                if (!script.exists()) script = new java.io.File("eeg_bridge.py");

                String python = "D:\\python312\\python.exe";
                if (!new java.io.File(python).exists()) python = "python";

                ProcessBuilder pb = new ProcessBuilder(
                    python, script.getAbsolutePath(), host, String.valueOf(localPort));
                pb.redirectErrorStream(false);
                wifiProcess = pb.start();

                // stderr → 日志面板
                new Thread(() -> {
                    try (BufferedReader e = new BufferedReader(
                            new InputStreamReader(wifiProcess.getErrorStream()))) {
                        String s;
                        while ((s = e.readLine()) != null) {
                            final String m = s;
                            SwingUtilities.invokeLater(() -> logPanel.log("INFO", "[Py] " + m));
                        }
                    } catch (IOException ignored) {}
                }, "WiFi-err").start();

                logPanel.log("INFO", "Python 接收解码中... (" + host + ":" + localPort + ")");

                // stdout → 纯CSV数据 → 直接推入缓冲区
                try (BufferedReader out = new BufferedReader(
                        new InputStreamReader(wifiProcess.getInputStream()))) {
                    readWifiLines(out);
                }
            } catch (Exception ex) {
                SwingUtilities.invokeLater(() -> {
                    logPanel.log("ERROR", "启动失败: " + ex.getMessage());
                    controlTabs.connectBtn.doClick();
                });
            } finally {
                wifiRunning = false;
            }
        }, "WiFi");
        wifiThread.setDaemon(true);
        wifiThread.start();
    }

    private void stopWiFiConnection() {
        wifiRunning = false;
        if (wifiThread != null) { wifiThread.interrupt(); wifiThread = null; }
        if (wifiProcess != null) { wifiProcess.destroyForcibly(); wifiProcess = null; }
    }

    private void startTimers() {
        // 模拟器
        sampleRate = 250;
        int samplesPerTick = Math.max(1, (int) (sampleRate / 30.0));

        // 波形刷新 33ms (~30fps)
        refreshTimer = new javax.swing.Timer(33, e -> {
            if (!paused) {
                waveformPanel.repaint();
                // 更新频谱和能量
                if (buffer.validSamples >= 64) {
                    double[] rms = EEGCore.channelRMS(buffer.data, NUM_CH, buffer.writePos, buffer.capacity, buffer.validSamples);
                    energyPanel.setEnergy(rms);
                    energyPanel.repaint();
                    spectrumPanel.repaint();
                }
            }
            // 统计
            statsBar.totalSamplesLabel.setText(String.valueOf(totalSamples));
            double usage = (double) buffer.validSamples / BUF_SIZE * 100;
            statsBar.bufferUsageLabel.setText(String.format("%.0f%%", usage));
            statsBar.frameSeqLabel.setText(String.valueOf(frameSeq));
            long elapsed = System.currentTimeMillis() - startTimeMs;
            if (elapsed > 0) {
                double rate = totalSamples * 1000.0 / elapsed;
                statsBar.dataRateLabel.setText(String.format("%.0f s/s", rate));
            }
        });
        refreshTimer.start();

        // 时钟
        clockTimer = new javax.swing.Timer(1000, e -> {
            if (startTimeMs > 0) {
                long sec = (System.currentTimeMillis() - startTimeMs) / 1000;
                controlTabs.timerLabel.setText(String.format("%02d:%02d:%02d", sec / 3600, (sec % 3600) / 60, sec % 60));
            }
        });
        clockTimer.start();

        // 分类器 2000ms
        classTimer = new javax.swing.Timer(2000, e -> {
            if (!classifying || controlTabs.modelWeights == null) return;
            double fs = Double.parseDouble((String) controlTabs.fsCombo.getSelectedItem());
            double[] features = EEGCore.extractFeatures(buffer.data, NUM_CH, buffer.writePos, buffer.capacity, buffer.validSamples, fs);
            double[] probs = EEGCore.linearClassify(features, controlTabs.modelWeights, controlTabs.modelBiases);
            int best = 0;
            for (int i = 1; i < probs.length; i++) if (probs[i] > probs[best]) best = i;
            if (controlTabs.probBars != null) {
                for (int i = 0; i < probs.length && i < controlTabs.probBars.length; i++) {
                    controlTabs.probBars[i].setValue((int) (probs[i] * 100));
                    controlTabs.probLabels[i].setText(String.format("%s: %.1f%%", controlTabs.modelLabels[i], probs[i] * 100));
                }
            }
            controlTabs.predictLabel.setText("预测结果: " + controlTabs.modelLabels[best] +
                String.format(" (%.1f%%)", probs[best] * 100));
            frameSeq++;
        });
        classTimer.start();
    }

    private void startSimulator() {
        if (simTimer != null) simTimer.stop();
        final double fs = sampleRate;
        final int spt = Math.max(1, (int) (fs / 30.0));
        simTimer = new javax.swing.Timer(33, e -> {
            if (!isSimulating || paused) return;
            double t0 = totalSamples / fs;
            boolean[] enabled = controlTabs.getEnabledChannels();
            for (int i = 0; i < spt; i++) {
                double t = t0 + (double) i / fs;
                double[] sample = EEGCore.simulateSample(t, NUM_CH);
                // 禁用通道填0
                for (int c = 0; c < NUM_CH; c++) {
                    if (!enabled[c]) sample[c] = 0;
                }
                buffer.push(sample);
                totalSamples++;
            }
        });
        simTimer.start();
    }
}
