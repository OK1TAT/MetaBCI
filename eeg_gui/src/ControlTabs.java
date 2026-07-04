/*
 * ControlTabs - 左侧控制面板(JTabbedPane，3个Tab)
 */
import javax.swing.*;
import javax.swing.border.TitledBorder;
import javax.swing.filechooser.FileNameExtensionFilter;
import java.awt.*;
import java.io.*;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;

public class ControlTabs extends JTabbedPane {
    private static final int NUM_CH = 16;
    private static final String[] CH_LABELS = {"FP1","FP2","F3","F4","C3","C4","P3","P4","O1","O2","F7","F8","T7","T8","P7","P8"};
    private static final Color PANEL_BG = new Color(245, 248, 252);
    private static final Color BORDER_CLR = new Color(25, 118, 210);
    private static final Color TEXT_CLR = new Color(33, 33, 33);
    private static final Color INPUT_BG = Color.WHITE;

    // ---- Tab1: 设备与数据 ----
    public JComboBox<String> protocolCombo;
    public JTextField addressField, portField;
    public JButton connectBtn;
    public JButton loadDataBtn;
    public JLabel dataFileInfo;

    // ---- Tab2: 采集配置 ----
    public JComboBox<String> gainCombo, fsCombo;
    public JCheckBox[] chChecks = new JCheckBox[NUM_CH];
    public JButton startBtn, pauseBtn, stopBtn;
    public JTextField markerField;
    public JButton markerBtn;
    public JLabel timerLabel;
    public JTextField subjectField;
    public JButton saveCSVBtn, saveEDFBtn;

    // ---- Tab3: 分类模型 ----
    public JButton loadModelBtn, startClassBtn, stopClassBtn;
    public JLabel modelInfoLabel;
    public JPanel probBarsPanel;
    public JLabel predictLabel;
    public JProgressBar[] probBars;
    public JLabel[] probLabels;

    // 回调
    private Runnable onConnect, onDisconnect, onLoadData, onStart, onPause, onStop, onSaveCSV, onSaveEDF, onLoadModel, onStartClass, onStopClass;
    private java.awt.event.ActionListener channelListener;

    // 模型数据
    public double[][] modelWeights;
    public double[] modelBiases;
    public String[] modelLabels;

    // 加载的数据
    public double[][] loadedData; // [nChannels][nSamples]
    public double loadedSampleRate = 250;

    public ControlTabs() {
        setFont(new Font("Microsoft YaHei", Font.BOLD, 13));
        setBackground(PANEL_BG);

        // Tab 1
        JPanel tab1 = makeTab();
        tab1.add(buildConnectionPanel());
        tab1.add(Box.createVerticalStrut(8));
        tab1.add(buildDataLoadPanel());
        tab1.add(Box.createVerticalGlue());
        addTab("设备与数据", tab1);

        // Tab 2
        JPanel tab2 = makeTab();
        tab2.add(buildChannelConfigPanel());
        tab2.add(Box.createVerticalStrut(8));
        tab2.add(buildAcquisitionPanel());
        tab2.add(Box.createVerticalStrut(8));
        tab2.add(buildDataSavePanel());
        tab2.add(Box.createVerticalGlue());
        addTab("采集配置", tab2);

        // Tab 3
        JPanel tab3 = makeTab();
        tab3.add(buildClassificationPanel());
        tab3.add(Box.createVerticalGlue());
        addTab("分类模型", tab3);
    }

    private JPanel makeTab() {
        JPanel p = new JPanel();
        p.setLayout(new BoxLayout(p, BoxLayout.Y_AXIS));
        p.setBackground(PANEL_BG);
        return p;
    }

    public void setCallbacks(Runnable onConnect, Runnable onDisconnect, Runnable onLoadData,
                             Runnable onStart, Runnable onPause, Runnable onStop,
                             Runnable onSaveCSV, Runnable onSaveEDF,
                             Runnable onLoadModel, Runnable onStartClass, Runnable onStopClass) {
        this.onConnect = onConnect;
        this.onDisconnect = onDisconnect;
        this.onLoadData = onLoadData;
        this.onStart = onStart;
        this.onPause = onPause;
        this.onStop = onStop;
        this.onSaveCSV = onSaveCSV;
        this.onSaveEDF = onSaveEDF;
        this.onLoadModel = onLoadModel;
        this.onStartClass = onStartClass;
        this.onStopClass = onStopClass;
    }

    // ==================== 设备连接 ====================
    private JPanel buildConnectionPanel() {
        JPanel p = new JPanel(new GridBagLayout());
        p.setBorder(titledBorder("设备连接"));
        p.setBackground(PANEL_BG);
        p.setMaximumSize(new Dimension(Integer.MAX_VALUE, 160));
        GridBagConstraints gbc = new GridBagConstraints();
        gbc.insets = new Insets(4, 6, 4, 6);
        gbc.fill = GridBagConstraints.HORIZONTAL;

        addLabel(p, gbc, "协议:", 0, 0);
        protocolCombo = new JComboBox<>(new String[]{"WiFi Shield", "TCP", "UDP", "模拟器"});
        protocolCombo.setSelectedItem("模拟器");
        styleCombo(protocolCombo);
        gbc.gridx = 1; gbc.gridy = 0; gbc.gridwidth = 2;
        p.add(protocolCombo, gbc);
        gbc.gridwidth = 1;

        addLabel(p, gbc, "地址:", 0, 1);
        addressField = styledField("192.168.4.1");
        gbc.gridx = 1; gbc.gridy = 1; gbc.gridwidth = 2;
        p.add(addressField, gbc);
        gbc.gridwidth = 1;

        addLabel(p, gbc, "端口:", 0, 2);
        portField = styledField("9000");
        gbc.gridx = 1; gbc.gridy = 2; gbc.gridwidth = 2;
        p.add(portField, gbc);
        gbc.gridwidth = 1;

        gbc.gridx = 0; gbc.gridy = 3; gbc.gridwidth = 3;
        connectBtn = blueBtn("连  接");
        connectBtn.addActionListener(e -> {
            if (connectBtn.getText().contains("连")) {
                if (onConnect != null) onConnect.run();
            } else {
                if (onDisconnect != null) onDisconnect.run();
            }
        });
        p.add(connectBtn, gbc);
        return p;
    }

    // ==================== 本地数据加载 ====================
    private JPanel buildDataLoadPanel() {
        JPanel p = new JPanel(new BorderLayout(5, 5));
        p.setBorder(titledBorder("本地数据"));
        p.setBackground(PANEL_BG);
        p.setMaximumSize(new Dimension(Integer.MAX_VALUE, 100));

        JPanel top = new JPanel(new FlowLayout(FlowLayout.LEFT, 5, 3));
        top.setOpaque(false);
        loadDataBtn = blueBtn("加载数据");
        loadDataBtn.addActionListener(e -> { if (onLoadData != null) onLoadData.run(); });
        top.add(loadDataBtn);
        p.add(top, BorderLayout.NORTH);

        dataFileInfo = new JLabel("未加载数据");
        dataFileInfo.setFont(new Font("Microsoft YaHei", Font.PLAIN, 12));
        dataFileInfo.setForeground(TEXT_CLR);
        dataFileInfo.setBorder(BorderFactory.createEmptyBorder(2, 8, 4, 8));
        p.add(dataFileInfo, BorderLayout.CENTER);
        return p;
    }

    // ==================== 通道配置 ====================
    private JPanel buildChannelConfigPanel() {
        JPanel p = new JPanel(new BorderLayout(5, 5));
        p.setBorder(titledBorder("通道配置"));
        p.setBackground(PANEL_BG);
        p.setMaximumSize(new Dimension(Integer.MAX_VALUE, 180));

        JPanel top = new JPanel(new FlowLayout(FlowLayout.LEFT, 8, 3));
        top.setOpaque(false);
        JLabel gl = new JLabel("增益:"); gl.setForeground(TEXT_CLR);
        top.add(gl);
        gainCombo = new JComboBox<>(new String[]{"×1","×2","×4","×6","×8","×12","×24"});
        gainCombo.setSelectedItem("×6");
        styleCombo(gainCombo);
        top.add(gainCombo);

        JLabel fl = new JLabel("采样率:"); fl.setForeground(TEXT_CLR);
        top.add(fl);
        fsCombo = new JComboBox<>(new String[]{"125","250","500","1000"});
        fsCombo.setSelectedItem("250");
        styleCombo(fsCombo);
        top.add(fsCombo);
        p.add(top, BorderLayout.NORTH);

        JPanel chGrid = new JPanel(new GridLayout(4, 4, 4, 2));
        chGrid.setOpaque(false);
        chGrid.setBorder(BorderFactory.createEmptyBorder(2, 6, 4, 6));
        for (int i = 0; i < NUM_CH; i++) {
            chChecks[i] = new JCheckBox(CH_LABELS[i], true);
            chChecks[i].setFont(new Font("Microsoft YaHei", Font.PLAIN, 11));
            chChecks[i].setForeground(TEXT_CLR);
            chChecks[i].addActionListener(e -> {
                if (channelListener != null) channelListener.actionPerformed(e);
            });
            chGrid.add(chChecks[i]);
        }
        p.add(chGrid, BorderLayout.CENTER);
        return p;
    }

    // ==================== 采集控制 ====================
    private JPanel buildAcquisitionPanel() {
        JPanel p = new JPanel(new BorderLayout(5, 5));
        p.setBorder(titledBorder("采集控制"));
        p.setBackground(PANEL_BG);
        p.setMaximumSize(new Dimension(Integer.MAX_VALUE, 140));

        JPanel btnRow = new JPanel(new FlowLayout(FlowLayout.LEFT, 5, 3));
        btnRow.setOpaque(false);
        startBtn = greenBtn("开始");
        startBtn.addActionListener(e -> { if (onStart != null) onStart.run(); });
        btnRow.add(startBtn);
        pauseBtn = blueBtn("暂停");
        pauseBtn.addActionListener(e -> { if (onPause != null) onPause.run(); });
        btnRow.add(pauseBtn);
        stopBtn = redBtn("停止");
        stopBtn.addActionListener(e -> { if (onStop != null) onStop.run(); });
        btnRow.add(stopBtn);
        p.add(btnRow, BorderLayout.NORTH);

        JPanel bottom = new JPanel(new FlowLayout(FlowLayout.LEFT, 8, 3));
        bottom.setOpaque(false);
        JLabel ml = new JLabel("标记:"); ml.setForeground(TEXT_CLR);
        bottom.add(ml);
        markerField = styledField("");
        markerField.setPreferredSize(new Dimension(100, 26));
        bottom.add(markerField);
        markerBtn = blueBtn("标记");
        markerBtn.setPreferredSize(new Dimension(60, 26));
        bottom.add(markerBtn);
        timerLabel = new JLabel("00:00:00");
        timerLabel.setFont(new Font("Consolas", Font.BOLD, 16));
        timerLabel.setForeground(new Color(13, 71, 161));
        bottom.add(timerLabel);
        p.add(bottom, BorderLayout.CENTER);
        return p;
    }

    // ==================== 数据保存 ====================
    private JPanel buildDataSavePanel() {
        JPanel p = new JPanel(new FlowLayout(FlowLayout.LEFT, 8, 3));
        p.setBorder(titledBorder("数据保存"));
        p.setBackground(PANEL_BG);
        p.setMaximumSize(new Dimension(Integer.MAX_VALUE, 70));

        JLabel sl = new JLabel("被试ID:"); sl.setForeground(TEXT_CLR);
        p.add(sl);
        subjectField = styledField("S001");
        subjectField.setPreferredSize(new Dimension(80, 26));
        p.add(subjectField);
        
        saveCSVBtn = blueBtn("保存CSV");
        saveCSVBtn.addActionListener(e -> { if (onSaveCSV != null) onSaveCSV.run(); });
        p.add(saveCSVBtn);
        
        saveEDFBtn = blueBtn("保存EDF");
        saveEDFBtn.addActionListener(e -> { if (onSaveEDF != null) onSaveEDF.run(); });
        p.add(saveEDFBtn);
        
        return p;
    }

    // ==================== 分类模型 ====================
    private JPanel buildClassificationPanel() {
        JPanel p = new JPanel(new BorderLayout(5, 5));
        p.setBorder(titledBorder("分类模型"));
        p.setBackground(PANEL_BG);

        JPanel top = new JPanel(new FlowLayout(FlowLayout.LEFT, 5, 3));
        top.setOpaque(false);
        loadModelBtn = blueBtn("载入模型");
        loadModelBtn.addActionListener(e -> { if (onLoadModel != null) onLoadModel.run(); });
        top.add(loadModelBtn);
        startClassBtn = greenBtn("开始分类");
        startClassBtn.setEnabled(false);
        startClassBtn.addActionListener(e -> { if (onStartClass != null) onStartClass.run(); });
        top.add(startClassBtn);
        stopClassBtn = redBtn("停止分类");
        stopClassBtn.setEnabled(false);
        stopClassBtn.addActionListener(e -> { if (onStopClass != null) onStopClass.run(); });
        top.add(stopClassBtn);
        p.add(top, BorderLayout.NORTH);

        modelInfoLabel = new JLabel("未加载模型");
        modelInfoLabel.setFont(new Font("Microsoft YaHei", Font.PLAIN, 12));
        modelInfoLabel.setForeground(TEXT_CLR);
        modelInfoLabel.setBorder(BorderFactory.createEmptyBorder(2, 8, 2, 8));
        p.add(modelInfoLabel, BorderLayout.CENTER);

        probBarsPanel = new JPanel();
        probBarsPanel.setLayout(new BoxLayout(probBarsPanel, BoxLayout.Y_AXIS));
        probBarsPanel.setOpaque(false);
        p.add(probBarsPanel, BorderLayout.SOUTH);

        predictLabel = new JLabel("预测结果: --");
        predictLabel.setFont(new Font("Microsoft YaHei", Font.BOLD, 14));
        predictLabel.setForeground(new Color(13, 71, 161));
        predictLabel.setBorder(BorderFactory.createEmptyBorder(4, 8, 6, 8));
        p.add(predictLabel, BorderLayout.AFTER_LAST_LINE);

        return p;
    }

    // ==================== 公共数据加载方法 ====================
    public void loadCSVData(File file) throws Exception {
        java.util.List<double[]> rows = new java.util.ArrayList<>();
        try (BufferedReader br = new BufferedReader(new InputStreamReader(new FileInputStream(file), StandardCharsets.UTF_8))) {
            String line;
            while ((line = br.readLine()) != null) {
                line = line.trim();
                if (line.isEmpty() || line.startsWith("#")) continue;
                String[] parts = line.split(",");
                double[] row = new double[parts.length];
                for (int i = 0; i < parts.length; i++) row[i] = Double.parseDouble(parts[i].trim());
                rows.add(row);
            }
        }
        int nCh = rows.isEmpty() ? 0 : rows.get(0).length;
        int nSamples = rows.size();
        loadedData = new double[nCh][nSamples];
        for (int s = 0; s < nSamples; s++) {
            double[] row = rows.get(s);
            for (int c = 0; c < nCh && c < NUM_CH; c++) {
                loadedData[c][s] = row[c];
            }
        }
    }

    public void loadJSONData(File file) throws Exception {
        String json = new String(Files.readAllBytes(file.toPath()), StandardCharsets.UTF_8);
        // 简单JSON解析
        loadedSampleRate = parseDouble(json, "sample_rate", 250);
        loadedData = parse2DArray(json, "channels");
    }

    public void loadEDFData(File file) throws Exception {
        EDFReader.EDFData edf = EDFReader.read(file);
        loadedSampleRate = edf.sampleRate;
        // 将EDF数据复制到loadedData (最多16通道)
        int nCh = Math.min(edf.numChannels, NUM_CH);
        loadedData = new double[nCh][edf.numSamples];
        for (int c = 0; c < nCh; c++) {
            System.arraycopy(edf.data[c], 0, loadedData[c], 0, edf.numSamples);
        }
    }

    public void loadModelJSON(File file) throws Exception {
        String name = file.getName().toLowerCase();
        if (name.endsWith(".joblib") || name.endsWith(".pkl")) {
            throw new Exception("不支持直接加载.joblib/.pkl文件。\n" +
                "请先在Python中转换为JSON格式：\n" +
                "  manager.export_model('json')\n" +
                "或使用独立脚本：\n" +
                "  python model_export.py your_model.joblib");
        }
        String json = new String(Files.readAllBytes(file.toPath()), StandardCharsets.UTF_8);
        modelWeights = parse2DArray(json, "weights");
        modelBiases = parse1DArray(json, "biases");
        modelLabels = parseStringArray(json, "labels");
    }

    public void setupProbBars() {
        if (modelLabels == null) return;
        probBarsPanel.removeAll();
        probBars = new JProgressBar[modelLabels.length];
        probLabels = new JLabel[modelLabels.length];
        for (int i = 0; i < modelLabels.length; i++) {
            JPanel row = new JPanel(new BorderLayout(5, 0));
            row.setOpaque(false);
            row.setBorder(BorderFactory.createEmptyBorder(2, 8, 2, 8));
            probLabels[i] = new JLabel(modelLabels[i] + ": 0.0%");
            probLabels[i].setFont(new Font("Microsoft YaHei", Font.PLAIN, 12));
            probLabels[i].setForeground(TEXT_CLR);
            probLabels[i].setPreferredSize(new Dimension(100, 20));
            row.add(probLabels[i], BorderLayout.WEST);
            probBars[i] = new JProgressBar(0, 100);
            probBars[i].setValue(0);
            probBars[i].setStringPainted(true);
            probBars[i].setForeground(new Color(25, 118, 210));
            row.add(probBars[i], BorderLayout.CENTER);
            probBarsPanel.add(row);
        }
        probBarsPanel.revalidate();
        probBarsPanel.repaint();
    }

    // ==================== 通道状态 ====================
    public void addChannelListener(java.awt.event.ActionListener listener) {
        this.channelListener = listener;
    }

    public boolean[] getEnabledChannels() {
        boolean[] enabled = new boolean[NUM_CH];
        for (int i = 0; i < NUM_CH; i++) {
            enabled[i] = (chChecks[i] == null || chChecks[i].isSelected());
        }
        return enabled;
    }

    // ==================== 简易JSON解析 ====================
    private double parseDouble(String json, String key, double def) {
        int idx = json.indexOf("\"" + key + "\"");
        if (idx < 0) return def;
        int colon = json.indexOf(':', idx);
        int end = colon + 1;
        while (end < json.length() && (Character.isDigit(json.charAt(end)) || json.charAt(end) == '.' || json.charAt(end) == '-')) end++;
        try { return Double.parseDouble(json.substring(colon + 1, end).trim()); } catch (Exception e) { return def; }
    }

    private double[][] parse2DArray(String json, String key) {
        int idx = json.indexOf("\"" + key + "\"");
        if (idx < 0) return null;
        int start = json.indexOf('[', idx);
        // 找到匹配的]
        int depth = 0;
        int end = start;
        for (int i = start; i < json.length(); i++) {
            if (json.charAt(i) == '[') depth++;
            else if (json.charAt(i) == ']') { depth--; if (depth == 0) { end = i; break; } }
        }
        String arr = json.substring(start + 1, end);
        java.util.List<double[]> rows = new java.util.ArrayList<>();
        int rowStart;
        while ((rowStart = arr.indexOf('[')) >= 0) {
            int rowEnd = arr.indexOf(']', rowStart);
            if (rowEnd < 0) break;
            String rowStr = arr.substring(rowStart + 1, rowEnd);
            String[] parts = rowStr.split(",");
            double[] row = new double[parts.length];
            for (int i = 0; i < parts.length; i++) {
                try { row[i] = Double.parseDouble(parts[i].trim()); } catch (Exception e) { row[i] = 0; }
            }
            rows.add(row);
            arr = arr.substring(rowEnd + 1);
        }
        return rows.isEmpty() ? null : rows.toArray(new double[0][]);
    }

    private double[] parse1DArray(String json, String key) {
        double[][] arr2d = parse2DArray(json, key);
        if (arr2d != null && arr2d.length == 1) return arr2d[0];
        // 如果parse2DArray没找到（因为不是2D数组），尝试直接解析
        int idx = json.indexOf("\"" + key + "\"");
        if (idx < 0) return null;
        int start = json.indexOf('[', idx);
        int end = json.indexOf(']', start);
        if (end < 0) return null;
        String arr = json.substring(start + 1, end);
        String[] parts = arr.split(",");
        double[] result = new double[parts.length];
        for (int i = 0; i < parts.length; i++) {
            try { result[i] = Double.parseDouble(parts[i].trim()); } catch (Exception e) { result[i] = 0; }
        }
        return result;
    }

    private String[] parseStringArray(String json, String key) {
        int idx = json.indexOf("\"" + key + "\"");
        if (idx < 0) return null;
        int start = json.indexOf('[', idx);
        int end = json.indexOf(']', start);
        if (end < 0) return null;
        String arr = json.substring(start + 1, end);
        java.util.List<String> list = new java.util.ArrayList<>();
        int p = 0;
        while (p < arr.length()) {
            int q1 = arr.indexOf('"', p);
            if (q1 < 0) break;
            int q2 = arr.indexOf('"', q1 + 1);
            if (q2 < 0) break;
            list.add(arr.substring(q1 + 1, q2));
            p = q2 + 1;
        }
        return list.isEmpty() ? null : list.toArray(new String[0]);
    }

    // ==================== UI辅助 ====================
    private TitledBorder titledBorder(String title) {
        return BorderFactory.createTitledBorder(
            BorderFactory.createLineBorder(BORDER_CLR),
            title, TitledBorder.LEFT, TitledBorder.TOP,
            new Font("Microsoft YaHei", Font.BOLD, 13), BORDER_CLR
        );
    }

    private void addLabel(JPanel p, GridBagConstraints gbc, String text, int x, int y) {
        JLabel l = new JLabel(text);
        l.setForeground(TEXT_CLR);
        l.setFont(new Font("Microsoft YaHei", Font.PLAIN, 13));
        gbc.gridx = x; gbc.gridy = y; gbc.gridwidth = 1;
        p.add(l, gbc);
    }

    private JTextField styledField(String text) {
        JTextField f = new JTextField(text);
        f.setFont(new Font("Microsoft YaHei", Font.PLAIN, 13));
        f.setForeground(TEXT_CLR);
        f.setBackground(INPUT_BG);
        f.setCaretColor(TEXT_CLR);
        return f;
    }

    private void styleCombo(JComboBox<String> c) {
        c.setFont(new Font("Microsoft YaHei", Font.PLAIN, 12));
        c.setForeground(TEXT_CLR);
        c.setBackground(INPUT_BG);
    }

    private JButton blueBtn(String text) {
        JButton b = new JButton(text);
        b.setFont(new Font("Microsoft YaHei", Font.PLAIN, 13));
        b.setBackground(new Color(25, 118, 210));
        b.setForeground(Color.WHITE);
        b.setOpaque(true);
        b.setContentAreaFilled(true);
        b.setFocusPainted(false);
        b.setPreferredSize(new Dimension(100, 30));
        b.setCursor(new Cursor(Cursor.HAND_CURSOR));
        return b;
    }

    private JButton greenBtn(String text) {
        JButton b = blueBtn(text);
        b.setBackground(new Color(56, 142, 60));
        return b;
    }

    private JButton redBtn(String text) {
        JButton b = blueBtn(text);
        b.setBackground(new Color(211, 47, 47));
        return b;
    }
}
