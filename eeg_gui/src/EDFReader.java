/*
 * EDFReader - 读取EDF(European Data Format)文件
 * 支持标准EDF(256)格式，返回double[][] [channel][sample]
 * v3: 修复编码 + 安全解析 + 调试输出 + 空值处理
 */
import java.io.*;
import java.nio.*;
import java.nio.charset.StandardCharsets;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.regex.*;

public class EDFReader {

    public static class EDFData {
        public double[][] data;        // [nChannels][nSamples]
        public double sampleRate;      // 采样率
        public int numChannels;
        public int numSamples;
        public String[] channelLabels;
        public Date startDate;
        public String patientInfo;
        public String recordingInfo;
    }

    private static final boolean DEBUG = true;

    // === 安全数字解析 ===
    private static int safeParseInt(String s, String fieldName) {
        s = s.trim();
        if (s.isEmpty()) {
            System.out.println("[EDFReader] 警告: " + fieldName + " 为空，使用默认值 0");
            return 0;
        }
        try {
            return Integer.parseInt(s);
        } catch (NumberFormatException e) {
            Matcher m = Pattern.compile("[-+]?\\d+").matcher(s);
            if (m.find()) {
                int val = Integer.parseInt(m.group());
                System.out.println("[EDFReader] 警告: " + fieldName + "=\"" + s + "\" → " + val);
                return val;
            }
            System.out.println("[EDFReader] 警告: " + fieldName + "=\"" + s + "\" → 默认 0");
            return 0;
        }
    }

    private static double safeParseDouble(String s, String fieldName) {
        s = s.trim();
        if (s.isEmpty()) {
            System.out.println("[EDFReader] 警告: " + fieldName + " 为空，使用默认值 0.0");
            return 0.0;
        }
        try {
            return Double.parseDouble(s);
        } catch (NumberFormatException e) {
            Matcher m = Pattern.compile("[-+]?\\d+\\.?\\d*").matcher(s);
            if (m.find()) {
                double val = Double.parseDouble(m.group());
                System.out.println("[EDFReader] 警告: " + fieldName + "=\"" + s + "\" → " + val);
                return val;
            }
            System.out.println("[EDFReader] 警告: " + fieldName + "=\"" + s + "\" → 默认 0.0");
            return 0.0;
        }
    }

    private static String readString(byte[] b) {
        return new String(b, StandardCharsets.ISO_8859_1).trim();
    }

    private static String readStringRaw(byte[] b) {
        return new String(b, StandardCharsets.ISO_8859_1);
    }

    public static EDFData read(File file) throws IOException {
        try (RandomAccessFile raf = new RandomAccessFile(file, "r")) {
            EDFData result = new EDFData();
            long fileLen = file.length();
            System.out.println("[EDFReader] 文件: " + file.getName() + " (" + fileLen + " bytes)");

            byte[] buf8 = new byte[8];
            byte[] buf80 = new byte[80];

            // === 固定Header ===
            raf.read(buf8);
            String version = readString(buf8);
            if (DEBUG) System.out.println("[EDFReader] version=\"" + version + "\" pos=" + raf.getFilePointer());

            raf.read(buf80);
            result.patientInfo = readString(buf80);
            if (DEBUG) System.out.println("[EDFReader] patient=\"" + result.patientInfo.substring(0, Math.min(40, result.patientInfo.length())) + "...\"");

            raf.read(buf80);
            result.recordingInfo = readString(buf80);

            raf.read(buf8);
            String dateStr = readString(buf8);

            raf.read(buf8);
            String timeStr = readString(buf8);

            try {
                SimpleDateFormat sdf = new SimpleDateFormat("dd.MM.yy HH.mm.ss");
                result.startDate = sdf.parse(dateStr + " " + timeStr);
            } catch (Exception e) {
                result.startDate = new Date();
            }

            raf.read(buf8);
            int headerBytes = safeParseInt(readString(buf8), "headerBytes");
            if (DEBUG) System.out.println("[EDFReader] headerBytes=" + headerBytes + " pos=" + raf.getFilePointer());

            raf.skipBytes(44);

            raf.read(buf8);
            int numRecords = safeParseInt(readString(buf8), "numRecords");
            if (DEBUG) System.out.println("[EDFReader] numRecords=" + numRecords + " pos=" + raf.getFilePointer());

            raf.read(buf8);
            double recordDuration = safeParseDouble(readString(buf8), "recordDuration");
            if (DEBUG) System.out.println("[EDFReader] recordDuration=" + recordDuration + " pos=" + raf.getFilePointer());

            raf.read(buf8);
            result.numChannels = safeParseInt(readString(buf8), "numChannels");
            int ns = result.numChannels;
            if (DEBUG) System.out.println("[EDFReader] numChannels=" + ns + " pos=" + raf.getFilePointer());

            if (ns <= 0 || ns > 200) {
                throw new IOException("通道数异常: " + ns + "，可能不是标准EDF文件");
            }

            // === 通道信息Header ===
            // channel labels (16 bytes each)
            result.channelLabels = new String[ns];
            for (int c = 0; c < ns; c++) {
                byte[] buf16 = new byte[16];
                raf.read(buf16);
                result.channelLabels[c] = readString(buf16);
            }
            if (DEBUG) System.out.println("[EDFReader] 通道标签: " + String.join(", ", result.channelLabels));

            // transducer type (80 bytes each)
            raf.skipBytes(80 * ns);
            // physical dimension (8 bytes each)
            raf.skipBytes(8 * ns);

            long posBefore = raf.getFilePointer();
            if (DEBUG) System.out.println("[EDFReader] 通道数值字段起始 pos=" + posBefore);

            // physical minimum (8 bytes each)
            double[] physMin = new double[ns];
            for (int c = 0; c < ns; c++) {
                byte[] b = new byte[8];
                raf.read(b);
                physMin[c] = safeParseDouble(readString(b), "physMin[" + c + "]");
            }
            // physical maximum (8 bytes each)
            double[] physMax = new double[ns];
            for (int c = 0; c < ns; c++) {
                byte[] b = new byte[8];
                raf.read(b);
                physMax[c] = safeParseDouble(readString(b), "physMax[" + c + "]");
            }
            // digital minimum (8 bytes each)
            double[] digMin = new double[ns];
            for (int c = 0; c < ns; c++) {
                byte[] b = new byte[8];
                raf.read(b);
                digMin[c] = safeParseDouble(readString(b), "digMin[" + c + "]");
            }
            // digital maximum (8 bytes each)
            double[] digMax = new double[ns];
            for (int c = 0; c < ns; c++) {
                byte[] b = new byte[8];
                raf.read(b);
                digMax[c] = safeParseDouble(readString(b), "digMax[" + c + "]");
            }
            // prefiltering (80 bytes each)
            raf.skipBytes(80 * ns);

            // samples per data record (8 bytes each)
            int[] samplesPerRecord = new int[ns];
            for (int c = 0; c < ns; c++) {
                byte[] b = new byte[8];
                raf.read(b);
                samplesPerRecord[c] = safeParseInt(readString(b), "samplesPerRecord[" + c + "]");
            }
            // reserved (32 bytes each)
            raf.skipBytes(32 * ns);

            long dataStart = raf.getFilePointer();
            if (DEBUG) {
                System.out.println("[EDFReader] 数据区起始 pos=" + dataStart);
                System.out.println("[EDFReader] headerBytes=" + headerBytes + " 实际=" + dataStart);
            }

            // === 计算采样率 ===
            result.sampleRate = (recordDuration > 0) ? samplesPerRecord[0] / recordDuration : 0;

            // === 读取Data Records ===
            int totalSamples = samplesPerRecord[0] * numRecords;
            result.numSamples = totalSamples;
            result.data = new double[ns][totalSamples];

            if (DEBUG) System.out.println("[EDFReader] 开始读取数据: " + ns + "ch x " + totalSamples + " samples");

            for (int r = 0; r < numRecords; r++) {
                for (int c = 0; c < ns; c++) {
                    int n = samplesPerRecord[c];
                    if (n <= 0) continue;
                    byte[] raw = new byte[n * 2];
                    raf.read(raw);
                    ByteBuffer bb = ByteBuffer.wrap(raw).order(ByteOrder.LITTLE_ENDIAN);

                    double range = digMax[c] - digMin[c];
                    double scale = (range != 0) ? (physMax[c] - physMin[c]) / range : 1.0;
                    int offset = r * samplesPerRecord[c];

                    for (int s = 0; s < n; s++) {
                        short dig = bb.getShort();
                        result.data[c][offset + s] = (dig - digMin[c]) * scale + physMin[c];
                    }
                }
            }

            System.out.println("[EDFReader] 读取完成: " + ns + "通道, "
                + totalSamples + "样本, 采样率=" + result.sampleRate + "Hz");
            return result;
        }
    }
}
