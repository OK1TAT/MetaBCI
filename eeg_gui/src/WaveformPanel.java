/*
 * WaveformPanel - 16通道实时波形绘制
 */
import javax.swing.*;
import java.awt.*;

public class WaveformPanel extends JPanel {
    private static final int NUM_CH = 16;
    private static final String[] CH_LABELS = {"FP1","FP2","F3","F4","C3","C4","P3","P4","O1","O2","F7","F8","T7","T8","P7","P8"};
    private static final Color[] CH_COLORS = {
        new Color(13,71,161), new Color(21,101,192), new Color(25,118,210),
        new Color(30,136,229), new Color(33,150,243), new Color(66,165,245),
        new Color(100,181,246), new Color(3,169,176), new Color(0,150,136),
        new Color(0,137,123), new Color(56,142,60), new Color(67,160,71),
        new Color(94,139,29), new Color(158,157,36), new Color(249,171,0),
        new Color(245,124,0)
    };

    private EEGCore.RingBuffer buffer;
    private boolean[] chEnabled;
    private int samplesPerScreen = 500;
    private boolean paused = false;
    private int filterMode = 0;  // 0=无, 1=低通平滑, 2=50Hz陷波
    private static final int FILTER_WINDOW = 5;
    private double filterFs = 250.0;

    public WaveformPanel(EEGCore.RingBuffer buffer, boolean[] chEnabled) {
        this.buffer = buffer;
        this.chEnabled = chEnabled;
        setBackground(Color.WHITE);
        setPreferredSize(new Dimension(600, 500));
    }

    public void setEnabledChannels(boolean[] enabled) { this.chEnabled = enabled; }
    public void setSamplesPerScreen(int n) { this.samplesPerScreen = n; }
    public int getSamplesPerScreen() { return samplesPerScreen; }
    public void setPaused(boolean p) { this.paused = p; }
    public void setFilterMode(int m) { this.filterMode = m; }

    @Override
    protected void paintComponent(Graphics g) {
        super.paintComponent(g);
        Graphics2D g2 = (Graphics2D) g.create();
        g2.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);

        int w = getWidth();
        int h = getHeight();
        int margin = 60;
        int plotW = w - margin - 10;
        int plotH = h - 20;

        // 收集启用通道索引
        int[] enabledIdx = new int[NUM_CH];
        int numEnabled = 0;
        for (int c = 0; c < NUM_CH; c++) {
            if (chEnabled == null || chEnabled[c]) {
                enabledIdx[numEnabled++] = c;
            }
        }
        // 没有启用通道时提示
        if (numEnabled == 0) {
            g2.setColor(new Color(158, 158, 158));
            g2.setFont(new Font("Microsoft YaHei", Font.PLAIN, 16));
            g2.drawString("未选择任何通道，请在左侧面板勾选通道", w / 2 - 160, h / 2);
            g2.dispose();
            return;
        }

        int chH = plotH / numEnabled;

        // 网格
        g2.setColor(new Color(230, 230, 230));
        for (int i = 0; i <= numEnabled; i++) {
            int y = 10 + i * chH;
            g2.drawLine(margin, y, margin + plotW, y);
        }
        for (int i = 0; i <= 10; i++) {
            int x = margin + i * plotW / 10;
            g2.drawLine(x, 10, x, 10 + plotH);
        }

        // 波形
        int n = Math.min(samplesPerScreen, buffer.validSamples);
        if (n < 2) { g2.dispose(); return; }

        for (int row = 0; row < numEnabled; row++) {
            int c = enabledIdx[row];
            double[] seg = buffer.getLatest(n, c);
            // ====== Java 侧滤波 (Python 已做基础滤波) ======
            if (filterMode == 1) {
                seg = EEGCore.lowPass(seg, FILTER_WINDOW);
            } else if (filterMode == 2) {
                seg = EEGCore.notch50(seg, filterFs);
            }
            // 自动缩放
            double min = Double.MAX_VALUE, max = -Double.MAX_VALUE;
            for (double v : seg) { if (v < min) min = v; if (v > max) max = v; }
            double range = max - min;
            if (range < 1e-9) range = 1;

            int yBase = 10 + row * chH + chH / 2;
            int halfH = chH / 2 - 2;

            g2.setColor(CH_COLORS[c]);
            g2.setStroke(new BasicStroke(1.2f));

            int prevX = -1, prevY = -1;
            for (int i = 0; i < n; i++) {
                int x = margin + (int) ((double) i / (n - 1) * plotW);
                int y = yBase - (int) ((seg[i] - min) / range * 2 * halfH - halfH);
                if (prevX >= 0) {
                    g2.drawLine(prevX, prevY, x, y);
                }
                prevX = x;
                prevY = y;
            }

            // 通道标签
            g2.setColor(new Color(33, 33, 33));
            g2.setFont(new Font("Consolas", Font.PLAIN, 11));
            g2.drawString(CH_LABELS[c], 2, yBase + 4);
        }

        g2.dispose();
    }
}
