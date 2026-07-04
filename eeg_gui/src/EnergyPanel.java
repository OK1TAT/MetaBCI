/*
 * EnergyPanel - 通道能量分布(RMS)柱状图
 */
import javax.swing.*;
import java.awt.*;

public class EnergyPanel extends JPanel {
    private static final int NUM_CH = 16;
    private static final String[] CH_LABELS = {"FP1","FP2","F3","F4","C3","C4","P3","P4","O1","O2","F7","F8","T7","T8","P7","P8"};

    private double[] energy = new double[NUM_CH];
    private boolean[] chEnabled;

    public EnergyPanel() {
        setBackground(Color.WHITE);
        setPreferredSize(new Dimension(400, 200));
    }

    public void setEnabledChannels(boolean[] enabled) { this.chEnabled = enabled; }

    public void setEnergy(double[] e) {
        if (e != null && e.length == NUM_CH) {
            System.arraycopy(e, 0, this.energy, 0, NUM_CH);
        }
    }

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

        // 找最大值
        double maxE = 0;
        for (double v : energy) if (v > maxE) maxE = v;
        if (maxE < 1e-9) maxE = 1;

        // 网格
        g2.setColor(new Color(220, 220, 220));
        for (int i = 0; i <= 5; i++) {
            int y = top + i * plotH / 5;
            g2.drawLine(margin, y, margin + plotW, y);
        }

        // 统计启用通道
        int numEnabled = 0;
        for (int c = 0; c < NUM_CH; c++) {
            if (chEnabled == null || chEnabled[c]) numEnabled++;
        }
        if (numEnabled == 0) {
            g2.dispose();
            return;
        }

        // 柱子（启用通道均匀分布）
        int barW = plotW / numEnabled - 4;
        int enabledIdx = 0;
        for (int c = 0; c < NUM_CH; c++) {
            boolean enabled = (chEnabled == null || chEnabled[c]);
            if (!enabled) continue;

            int x = margin + enabledIdx * (plotW / numEnabled) + 2;
            int barH = (int) (energy[c] / maxE * plotH * 0.9);
            int y = top + plotH - barH;

            // 渐变色：蓝→青→绿
            float ratio = (float) (energy[c] / maxE);
            Color col = new Color(
                (int) (30 + 40 * (1 - ratio)),
                (int) (120 + 100 * ratio),
                (int) (210 - 50 * ratio)
            );
            g2.setColor(col);
            g2.fillRect(x, y, barW, barH);

            // 边框
            g2.setColor(col.darker());
            g2.drawRect(x, y, barW, barH);

            // 通道标签
            g2.setColor(new Color(33, 33, 33));
            g2.setFont(new Font("Consolas", Font.PLAIN, 9));
            g2.drawString(CH_LABELS[c], x, top + plotH + 12);

            enabledIdx++;
        }

        // Y轴标签
        g2.setColor(new Color(100, 100, 100));
        g2.setFont(new Font("SansSerif", Font.PLAIN, 9));
        g2.drawString("RMS", 5, top + plotH / 2);

        g2.dispose();
    }
}
