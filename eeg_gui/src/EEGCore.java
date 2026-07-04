/*
 * EEGCore - 核心工具类：环形缓冲区、DFT、频带功率、分类、模拟器
 */
import java.util.Random;

public class EEGCore {

    // ==================== 频带定义 ====================
    public static final String[] BAND_NAMES = {"δ", "θ", "α", "β", "γ"};
    public static final double[] BAND_LOW  = {1, 4, 8, 13, 30};
    public static final double[] BAND_HIGH = {4, 8, 13, 30, 50};

    // ==================== 环形缓冲区 ====================
    public static class RingBuffer {
        public final int nChannels;
        public final int capacity;
        public final double[][] data;
        public int writePos = 0;
        public int validSamples = 0;

        public RingBuffer(int nChannels, int capacity) {
            this.nChannels = nChannels;
            this.capacity = capacity;
            this.data = new double[nChannels][capacity];
        }

        public synchronized void push(double[] sample) {
            for (int c = 0; c < nChannels && c < sample.length; c++) {
                data[c][writePos] = sample[c];
            }
            writePos = (writePos + 1) % capacity;
            if (validSamples < capacity) validSamples++;
        }

        public synchronized double[] getLatest(int nSamples, int channel) {
            nSamples = Math.min(nSamples, validSamples);
            double[] out = new double[nSamples];
            int start = (writePos - nSamples + capacity) % capacity;
            for (int i = 0; i < nSamples; i++) {
                out[i] = data[channel][(start + i) % capacity];
            }
            return out;
        }

        public synchronized void clear() {
            for (int c = 0; c < nChannels; c++)
                java.util.Arrays.fill(data[c], 0);
            writePos = 0;
            validSamples = 0;
        }
    }

    // ==================== DFT ====================
    public static double[] dftMagnitude(double[] signal, int nfft) {
        int N = Math.min(signal.length, nfft);
        int nBins = nfft / 2 + 1;
        double[] mag = new double[nBins];
        for (int k = 0; k < nBins; k++) {
            double re = 0, im = 0;
            for (int n = 0; n < N; n++) {
                double angle = 2.0 * Math.PI * k * n / nfft;
                // Hanning窗
                double w = 0.5 * (1.0 - Math.cos(2.0 * Math.PI * n / (N - 1)));
                re += signal[n] * w * Math.cos(angle);
                im -= signal[n] * w * Math.sin(angle);
            }
            mag[k] = (re * re + im * im) / (N * N);
        }
        return mag;
    }

    // ==================== 频带功率 ====================
    public static double[] bandPower(double[] signal, double fs, int nfft) {
        double[] mag = dftMagnitude(signal, nfft);
        double df = fs / nfft;
        double[] power = new double[5];
        for (int b = 0; b < 5; b++) {
            int lo = (int) Math.ceil(BAND_LOW[b] / df);
            int hi = (int) Math.floor(BAND_HIGH[b] / df);
            double sum = 0;
            int cnt = 0;
            for (int k = lo; k <= hi && k < mag.length; k++) {
                sum += mag[k];
                cnt++;
            }
            power[b] = cnt > 0 ? sum / cnt : 0;
        }
        return power;
    }

    // 80维特征：5频带 × 16通道
    public static double[] extractFeatures(double[][] buf, int nCh, int writePos, int capacity, int validSamples, double fs) {
        int nfft = 256;
        int nUse = Math.min(nfft, validSamples);
        double[] features = new double[5 * nCh];
        for (int c = 0; c < nCh; c++) {
            double[] seg = new double[nUse];
            int start = (writePos - nUse + capacity) % capacity;
            for (int i = 0; i < nUse; i++) {
                seg[i] = buf[c][(start + i) % capacity];
            }
            double[] bp = bandPower(seg, fs, nfft);
            for (int b = 0; b < 5; b++) {
                features[b * nCh + c] = bp[b];
            }
        }
        return features;
    }

    // ==================== RMS能量 ====================
    public static double[] channelRMS(double[][] buf, int nCh, int writePos, int capacity, int validSamples) {
        int nUse = Math.min(1000, validSamples);
        double[] rms = new double[nCh];
        for (int c = 0; c < nCh; c++) {
            double sum = 0;
            int start = (writePos - nUse + capacity) % capacity;
            for (int i = 0; i < nUse; i++) {
                double v = buf[c][(start + i) % capacity];
                sum += v * v;
            }
            rms[c] = Math.sqrt(sum / nUse);
        }
        return rms;
    }

    // ==================== 线性分类 + Softmax ====================
    public static double[] linearClassify(double[] features, double[][] weights, double[] biases) {
        int nClasses = weights.length;
        double[] scores = new double[nClasses];
        for (int c = 0; c < nClasses; c++) {
            scores[c] = biases[c];
            for (int f = 0; f < features.length && f < weights[c].length; f++) {
                scores[c] += weights[c][f] * features[f];
            }
        }
        return softmax(scores);
    }

    public static double[] softmax(double[] x) {
        double max = Double.NEGATIVE_INFINITY;
        for (double v : x) if (v > max) max = v;
        double sum = 0;
        double[] out = new double[x.length];
        for (int i = 0; i < x.length; i++) {
            out[i] = Math.exp(x[i] - max);
            sum += out[i];
        }
        for (int i = 0; i < x.length; i++) out[i] /= sum;
        return out;
    }

    // ==================== 滤波器 ====================
    // 简单滑动平均低通
    public static double[] lowPass(double[] sig, int windowSize) {
        double[] out = new double[sig.length];
        int half = windowSize / 2;
        for (int i = 0; i < sig.length; i++) {
            double sum = 0;
            int cnt = 0;
            for (int j = Math.max(0, i - half); j <= Math.min(sig.length - 1, i + half); j++) {
                sum += sig[j];
                cnt++;
            }
            out[i] = sum / cnt;
        }
        return out;
    }

    // 50Hz陷波（简单差分）
    public static double[] notch50(double[] sig, double fs) {
        double[] out = new double[sig.length];
        int delay = (int) Math.round(fs / 50.0);
        for (int i = 0; i < sig.length; i++) {
            if (i >= delay) {
                out[i] = sig[i] - 0.5 * sig[i - delay];
            } else {
                out[i] = sig[i];
            }
        }
        return out;
    }

    // ==================== 模拟器信号生成 ====================
    private static final Random rng = new Random(42);
    private static final double[] CH_FREQS = {10, 12, 8, 10, 6, 8, 20, 20, 2, 2, 10, 12, 6, 8, 10, 10};

    public static double[] simulateSample(double t, int nCh) {
        double[] s = new double[nCh];
        for (int c = 0; c < nCh; c++) {
            double f1 = CH_FREQS[c];
            s[c] = 20 * Math.sin(2 * Math.PI * f1 * t)
                 + 8 * Math.sin(2 * Math.PI * (f1 * 2) * t + 0.5)
                 + 5 * Math.sin(2 * Math.PI * 6 * t + c * 0.3)
                 + 3 * Math.sin(2 * Math.PI * 2 * t)
                 + 15 * rng.nextGaussian()
                 + 10 * Math.sin(2 * Math.PI * 50 * t); // 50Hz工频
        }
        return s;
    }
}
