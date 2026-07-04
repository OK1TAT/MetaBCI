/*
 * SpectrumPanel - 频谱分析(PSD)绘制
 */
import javax.swing.*;
import java.awt.*;

public class SpectrumPanel extends JPanel {
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
    private double sampleRate = 250;
    private int selectedChannel = -1; // -1 = 全部通道叠加
    private int nfft = 256;
    private double maxFreq = 50;

    public SpectrumPanel(EEGCore.RingBuffer buffer, boolean[] chEnabled) {
        this.buffer = buffer;
        this.chEnabled = chEnabled;
        setBackground(Color.WHITE);
        setPreferredSize(new Dimension(400, 250));
    }

    public void setEnabledChannels(boolean[] enabled) { this.chEnabled = enabled; }
    public void setSampleRate(double fs) { this.sampleRate = fs; }
    public void setSelectedChannel(int ch) { this.selectedChannel = ch; }

    @Override
    protected void paintComponent(Graphics g) {
        super.paintComponent(g);
        Graphics2D g2 = (Graphics2D) g.create();
        g2.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);

        int w = getWidth(), h = getHeight();
        int margin = 50, right = 20, top = 10, bottom = 30;
        int plotW = w - margin - right;
        int plotH = h - top - bottom;

        if (plotW <= 0 || plotH <= 0) { g2.dispose(); return; }

        // 频带背景色
        Color[] bandColors = {
            new Color(255,0,0,20), new Color(255,165,0,20),
            new Color(0,128,0,20), new Color(0,0,255,20), new Color(128,0,128,20)
        };
        for (int b = 0; b < 5; b++) {
            int x1 = margin + (int) (EEGCore.BAND_LOW[b] / maxFreq * plotW);
            int x2 = margin + (int) (EEGCore.BAND_HIGH[b] / maxFreq * plotW);
            g2.setColor(bandColors[b]);
            g2.fillRect(x1, top, x2 - x1, plotH);
        }

        // 网格
        g2.setColor(new Color(200, 200, 200));
        g2.setStroke(new BasicStroke(0.5f));
        for (int f = 0; f <= maxFreq; f += 10) {
            int x = margin + (int) (f / maxFreq * plotW);
            g2.drawLine(x, top, x, top + plotH);
        }
        for (int i = 0; i <= 5; i++) {
            int y = top + i * plotH / 5;
            g2.drawLine(margin, y, margin + plotW, y);
        }

        // 计算PSD
        int nUse = Math.min(nfft, buffer.validSamples);
        if (nUse < 16) { g2.dispose(); return; }

        double[] psd;
        if (selectedChannel >= 0 && selectedChannel < 16) {
            // 单通道模式：检查是否启用
            boolean enabled = (chEnabled == null || chEnabled[selectedChannel]);
            if (enabled) {
                double[] seg = buffer.getLatest(nUse, selectedChannel);
                psd = EEGCore.dftMagnitude(seg, nfft);
                drawPSD(g2, psd, CH_COLORS[selectedChannel], margin, top, plotW, plotH);
            } else {
                // 禁用通道：不画任何内容
            }
        } else {
            // 叠加模式：只画启用通道
            for (int c = 0; c < 16; c++) {
                if (chEnabled != null && !chEnabled[c]) continue;
                double[] seg = buffer.getLatest(nUse, c);
                psd = EEGCore.dftMagnitude(seg, nfft);
                Color col = new Color(CH_COLORS[c].getRed(), CH_COLORS[c].getGreen(), CH_COLORS[c].getBlue(), 120);
                drawPSD(g2, psd, col, margin, top, plotW, plotH);
            }
        }

        // 坐标轴
        g2.setColor(new Color(33, 33, 33));
        g2.setFont(new Font("SansSerif", Font.PLAIN, 10));
        for (int f = 0; f <= maxFreq; f += 10) {
            int x = margin + (int) (f / maxFreq * plotW);
            g2.drawString(f + "Hz", x - 10, top + plotH + 15);
        }
        g2.drawString("PSD", 5, top + plotH / 2);

        // 频带标签
        g2.setFont(new Font("SansSerif", Font.BOLD, 10));
        for (int b = 0; b < 5; b++) {
            int x1 = margin + (int) (EEGCore.BAND_LOW[b] / maxFreq * plotW);
            int x2 = margin + (int) (EEGCore.BAND_HIGH[b] / maxFreq * plotW);
            int cx = (x1 + x2) / 2;
            g2.setColor(new Color(100, 100, 100, 180));
            g2.drawString(EEGCore.BAND_NAMES[b], cx - 4, top + 12);
        }

        g2.dispose();
    }

    private void drawPSD(Graphics2D g2, double[] psd, Color color, int mx, int my, int pw, int ph) {
        int nBins = psd.length;
        double df = sampleRate / (2 * (nBins - 1));
        int maxBin = (int) Math.min(nBins - 1, maxFreq / df);
        if (maxBin < 2) return;

        // 找最大值做归一化
        double maxVal = 0;
        for (int i = 1; i <= maxBin; i++) if (psd[i] > maxVal) maxVal = psd[i];
        if (maxVal < 1e-12) return;

        g2.setColor(color);
        g2.setStroke(new BasicStroke(1.2f));
        int prevX = -1, prevY = -1;
        for (int i = 1; i <= maxBin; i++) {
            double freq = i * df;
            int x = mx + (int) (freq / maxFreq * pw);
            int y = my + ph - (int) (psd[i] / maxVal * ph * 0.9);
            if (prevX >= 0) g2.drawLine(prevX, prevY, x, y);
            prevX = x;
            prevY = y;
        }
    }
}
