/*
 * LogPanel - 运行日志面板
 */
import javax.swing.*;
import javax.swing.text.*;
import java.awt.*;
import java.text.SimpleDateFormat;
import java.util.Date;

public class LogPanel extends JPanel {
    private final JTextPane logArea;
    private static final SimpleDateFormat SDF = new SimpleDateFormat("HH:mm:ss");

    public LogPanel() {
        setLayout(new BorderLayout());
        setPreferredSize(new Dimension(0, 140));
        setBorder(BorderFactory.createCompoundBorder(
            BorderFactory.createTitledBorder(BorderFactory.createLineBorder(new Color(25, 118, 210)),
                "运行日志", javax.swing.border.TitledBorder.LEFT, javax.swing.border.TitledBorder.TOP,
                new Font("Microsoft YaHei", Font.BOLD, 13), new Color(25, 118, 210)),
            BorderFactory.createEmptyBorder(2, 2, 2, 2)
        ));

        logArea = new JTextPane();
        logArea.setEditable(false);
        logArea.setFont(new Font("Microsoft YaHei", Font.PLAIN, 12));
        logArea.setBackground(new Color(252, 252, 252));

        JScrollPane sp = new JScrollPane(logArea);
        sp.setVerticalScrollBarPolicy(ScrollPaneConstants.VERTICAL_SCROLLBAR_ALWAYS);
        add(sp, BorderLayout.CENTER);
    }

    public void log(String level, String msg) {
        SwingUtilities.invokeLater(() -> {
            try {
                StyledDocument doc = logArea.getStyledDocument();
                SimpleAttributeSet attrs = new SimpleAttributeSet();
                Color color;
                switch (level) {
                    case "WARN": color = new Color(230, 150, 0); break;
                    case "ERROR": color = new Color(200, 30, 30); break;
                    default: color = new Color(33, 33, 33);
                }
                StyleConstants.setForeground(attrs, color);
                StyleConstants.setFontFamily(attrs, "Microsoft YaHei");
                StyleConstants.setFontSize(attrs, 12);
                String line = String.format("[%s] [%s] %s\n", SDF.format(new Date()), level, msg);
                doc.insertString(doc.getLength(), line, attrs);
                logArea.setCaretPosition(doc.getLength());
            } catch (BadLocationException e) {
                // ignore
            }
        });
    }

    public void clear() {
        SwingUtilities.invokeLater(() -> {
            try {
                logArea.getDocument().remove(0, logArea.getDocument().getLength());
            } catch (BadLocationException e) { /* ignore */ }
        });
    }
}
