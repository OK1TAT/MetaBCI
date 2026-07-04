/*
 * EDFWriter - 写入EDF(European Data Format)文件
 * 标准EDF格式，数据为16-bit整数
 */
import java.io.*;
import java.nio.*;
import java.text.SimpleDateFormat;
import java.util.Date;

public class EDFWriter {

    /**
     * 写入EDF文件
     * @param file 目标文件
     * @param data [nChannels][nSamples] 物理值数据
     * @param sampleRate 采样率
     * @param channelLabels 通道名称数组
     */
    public static void write(File file, double[][] data, double sampleRate, String[] channelLabels)
            throws IOException {
        int ns = data.length;
        int totalSamples = data[0].length;

        // 每个data record包含1秒的数据
        int samplesPerRecord = (int) sampleRate;
        int numRecords = totalSamples / samplesPerRecord;

        // 计算每个通道的物理值范围
        double[] physMin = new double[ns];
        double[] physMax = new double[ns];
        for (int c = 0; c < ns; c++) {
            physMin[c] = Double.MAX_VALUE;
            physMax[c] = -Double.MAX_VALUE;
            for (int s = 0; s < totalSamples; s++) {
                double v = data[c][s];
                if (v < physMin[c]) physMin[c] = v;
                if (v > physMax[c]) physMax[c] = v;
            }
            // 留一点余量避免恰好等于极值
            double range = physMax[c] - physMin[c];
            if (range < 1e-9) {
                physMin[c] -= 1;
                physMax[c] += 1;
            } else {
                physMin[c] -= range * 0.01;
                physMax[c] += range * 0.01;
            }
        }

        // Digital范围固定为16-bit
        double digMin = -32768;
        double digMax = 32767;

        try (FileOutputStream fos = new FileOutputStream(file);
             DataOutputStream dos = new DataOutputStream(new BufferedOutputStream(fos))) {

            // === 固定Header (256 bytes) ===
            // Version
            writeFixedString(dos, "0       ", 8);

            // Patient info
            SimpleDateFormat sdf = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss");
            writeFixedString(dos, sdf.format(new Date()), 80);

            // Recording info
            writeFixedString(dos, "EEG Recording", 80);

            // Start date (dd.mm.yy)
            SimpleDateFormat dateFmt = new SimpleDateFormat("dd.MM.yy");
            writeFixedString(dos, dateFmt.format(new Date()), 8);

            // Start time (hh.mm.ss)
            SimpleDateFormat timeFmt = new SimpleDateFormat("HH.mm.ss");
            writeFixedString(dos, timeFmt.format(new Date()), 8);

            // Number of bytes in header
            int headerBytes = 256 + ns * 256;
            writeFixedString(dos, String.valueOf(headerBytes), 8);

            // Reserved (44 bytes)
            writeFixedString(dos, "", 44);

            // Number of data records (-1 = unknown/continuous)
            writeFixedString(dos, String.valueOf(numRecords), 8);

            // Duration of data record (seconds)
            writeFixedString(dos, "1", 8);

            // Number of signals (channels)
            writeFixedString(dos, String.valueOf(ns), 8);

            // === 通道信息Header ===
            // Channel labels (16 bytes each)
            for (int c = 0; c < ns; c++) {
                writeFixedString(dos, channelLabels[c], 16);
            }
            // Transducer type (80 bytes each)
            for (int c = 0; c < ns; c++) {
                writeFixedString(dos, "AgAgCl", 80);
            }
            // Physical dimension (6 bytes each) - uV
            for (int c = 0; c < ns; c++) {
                writeFixedString(dos, "uV", 6);
            }
            // Physical minimum (8 bytes each)
            for (int c = 0; c < ns; c++) {
                writeFixedString(dos, String.format("%.4f", physMin[c]), 8);
            }
            // Physical maximum (8 bytes each)
            for (int c = 0; c < ns; c++) {
                writeFixedString(dos, String.format("%.4f", physMax[c]), 8);
            }
            // Digital minimum (8 bytes each)
            for (int c = 0; c < ns; c++) {
                writeFixedString(dos, String.format("%.0f", digMin), 8);
            }
            // Digital maximum (8 bytes each)
            for (int c = 0; c < ns; c++) {
                writeFixedString(dos, String.format("%.0f", digMax), 8);
            }
            // Prefiltering (80 bytes each)
            for (int c = 0; c < ns; c++) {
                writeFixedString(dos, "", 80);
            }
            // Samples per data record (8 bytes each)
            for (int c = 0; c < ns; c++) {
                writeFixedString(dos, String.valueOf(samplesPerRecord), 8);
            }
            // Reserved (32 bytes each)
            for (int c = 0; c < ns; c++) {
                writeFixedString(dos, "", 32);
            }

            // === Data Records ===
            byte[] sampleBuf = new byte[samplesPerRecord * 2];
            ByteBuffer bb = ByteBuffer.wrap(sampleBuf).order(ByteOrder.LITTLE_ENDIAN);

            double[] scale = new double[ns];
            for (int c = 0; c < ns; c++) {
                scale[c] = (digMax - digMin) / (physMax[c] - physMin[c]);
            }

            for (int r = 0; r < numRecords; r++) {
                for (int c = 0; c < ns; c++) {
                    bb.clear();
                    int offset = r * samplesPerRecord;
                    for (int s = 0; s < samplesPerRecord; s++) {
                        double phys = data[c][offset + s];
                        int dig = (int) Math.round((phys - physMin[c]) * scale[c] + digMin);
                        // Clamp to 16-bit range
                        if (dig < -32768) dig = -32768;
                        if (dig > 32767) dig = 32767;
                        bb.putShort((short) dig);
                    }
                    dos.write(sampleBuf);
                }
            }

            dos.flush();
        }
    }

    private static void writeFixedString(DataOutputStream dos, String s, int length) throws IOException {
        byte[] bytes = new byte[length];
        byte[] src = s.getBytes("ISO-8859-1");
        int n = Math.min(src.length, length);
        System.arraycopy(src, 0, bytes, 0, n);
        // 剩余部分填空格
        for (int i = n; i < length; i++) bytes[i] = ' ';
        dos.write(bytes);
    }
}
