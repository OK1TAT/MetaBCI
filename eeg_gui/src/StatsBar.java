/*
 * StatsBar - 底部统计仪表盘
 */
import javax.swing.*;
import java.awt.*;

public class StatsBar extends JPanel {
    public final JLabel totalSamplesLabel;
    public final JLabel lossRateLabel;
    public final JLabel bufferUsageLabel;
    public final JLabel dataRateLabel;
    public final JLabel frameSeqLabel;

    public StatsBar() {
        setLayout(new GridLayout(1, 5, 10, 0));
        setBackground(new Color(13, 71, 161));
        setBorder(BorderFactory.createEmptyBorder(6, 15, 6, 15));

        totalSamplesLabel = createCell("0");
        lossRateLabel = createCell("0.0%");
        bufferUsageLabel = createCell("0%");
        dataRateLabel = createCell("0 s/s");
        frameSeqLabel = createCell("0");

        add(wrap("总采样点", totalSamplesLabel));
        add(wrap("丢包率", lossRateLabel));
        add(wrap("缓冲区", bufferUsageLabel));
        add(wrap("数据速率", dataRateLabel));
        add(wrap("帧序号", frameSeqLabel));
    }

    private JLabel createCell(String value) {
        JLabel l = new JLabel(value, SwingConstants.CENTER);
        l.setFont(new Font("Consolas", Font.BOLD, 22));
        l.setForeground(Color.WHITE);
        return l;
    }

    private JPanel wrap(String title, JLabel valueLabel) {
        JPanel p = new JPanel(new BorderLayout(0, 2));
        p.setOpaque(false);
        JLabel tl = new JLabel(title, SwingConstants.CENTER);
        tl.setFont(new Font("Microsoft YaHei", Font.PLAIN, 11));
        tl.setForeground(new Color(187, 222, 251));
        p.add(tl, BorderLayout.NORTH);
        p.add(valueLabel, BorderLayout.CENTER);
        return p;
    }
}
