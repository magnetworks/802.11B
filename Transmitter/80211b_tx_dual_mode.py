
import numpy as np

class IEEE80211bTransmitterDualMode:
    """
    802.11b (1 Mbps DBPSK + Barker) transmitter in either floating-point or fixed-point.
    - Proper upsampling by sps
    - SRRC pulse shaping
    - Fixed-point path with Q1.15 data/coeffs, explicit integer FIR MAC, saturation
    - Returns FLOAT arrays in BOTH modes so you can compare directly
    - Helper to compute SNR between float and fixed outputs
    """
    def __init__(self, sps=8, rrc_span_chips=6, rrc_alpha=0.35,
                 data_bits=10, coeff_bits=12, acc_bits=24):
        # TX params
        self.sps = int(sps)                 # samples per chip
        self.rrc_span_chips = int(rrc_span_chips)
        self.rrc_alpha = float(rrc_alpha)
        # Fixed-point specs
        self.data_bits = int(data_bits)     # Q1.(data_bits-1)
        self.coeff_bits = int(coeff_bits)   # Q1.(coeff_bits-1)
        self.acc_bits = int(acc_bits)       # signed accumulator width
        self.data_frac = self.data_bits - 1
        self.coeff_frac = self.coeff_bits - 1

        # Constants
        self.barker11 = np.array([+1, -1, +1, +1, -1, +1, +1, +1, -1, -1, -1], dtype=np.float64)

        # Build SRRC taps (float) at chip rate; each "symbol" here is a chip
        self.h_float = self._srrc(self.rrc_alpha, self.rrc_span_chips, self.sps)

        # Quantized taps for fixed mode (Q1.15 ints)
        self.h_int = self._float_to_q(self.h_float, self.coeff_bits, self.coeff_frac, np.int32)

    # ---------------- Fixed-point helpers ----------------
    @staticmethod
    def _saturate_signed(x, bits):
        maxv = (1 << (bits - 1)) - 1
        minv = -(1 << (bits - 1))
        if isinstance(x, np.ndarray):
            x = np.where(x > maxv, maxv, x)
            x = np.where(x < minv, minv, x)
            return x
        else:
            return max(min(x, maxv), minv)

    @staticmethod
    def _float_to_q(x, total_bits, frac_bits, dtype=np.int32):
        scale = 1 << frac_bits
        maxv = (1 << (total_bits - 1)) - 1
        minv = -(1 << (total_bits - 1))
        xi = np.round(x * scale)
        xi = np.clip(xi, minv, maxv).astype(dtype)
        return xi

    @staticmethod
    def _q_to_float(xi, frac_bits):
        return xi.astype(np.float64) / (1 << frac_bits)

    # ---------------- Signal processing blocks ----------------
    @staticmethod
    def _lfsr_scramble(bits, seed=0x7F):
        """7-bit LFSR: polynomial x^7 + x^4 + 1. Returns scrambled bits (0/1)."""
        state = seed & 0x7F
        out = np.empty_like(bits, dtype=np.int8)
        for i, b in enumerate(bits):
            fb = ((state >> 3) ^ (state >> 6)) & 1  # taps 4 and 7 (1-indexed)
            out[i] = b ^ fb
            state = ((state << 1) & 0x7E) | fb
        return out

    @staticmethod
    def _dbpsk_diff_encode(bits):
        """Return complex symbols of unit magnitude with DBPSK differential encoding."""
        phase = 0.0
        syms = np.empty(len(bits), dtype=np.complex128)
        for i, b in enumerate(bits):
            if b:  # bit 1 -> pi phase jump
                phase += np.pi
            syms[i] = np.exp(1j * phase)  # unit circle
        return syms

    def _spread_barker(self, symbols):
        chips = np.empty(len(symbols) * 11, dtype=np.complex128)
        idx = 0
        for s in symbols:
            chips[idx:idx+11] = s * self.barker11
            idx += 11
        return chips

    @staticmethod
    def _upsample_by_sps(chips, sps):
        y = np.zeros(len(chips) * sps, dtype=chips.dtype)
        y[::sps] = chips
        return y

    @staticmethod
    def _srrc(alpha, span_chips, sps):
        """Square-root raised cosine filter (discrete-time) with chip as 'symbol' period."""
        N = int(span_chips * sps)
        if N % 2 == 0:
            N += 1
        t = np.arange(-N//2, N//2 + 1, dtype=np.float64) / sps
        h = np.zeros_like(t)
        for i, ti in enumerate(t):
            if abs(ti) < 1e-12:
                h[i] = 1.0 - alpha + (4*alpha/np.pi)
            elif alpha != 0 and abs(abs(ti) - 1/(4*alpha)) < 1e-12:
                h[i] = (alpha/np.sqrt(2))*(((1+2/np.pi)*np.sin(np.pi/(4*alpha))) +
                                           ((1-2/np.pi)*np.cos(np.pi/(4*alpha))))
            else:
                num = np.sin(np.pi*ti*(1-alpha)) + 4*alpha*ti*np.cos(np.pi*ti*(1+alpha))
                den = np.pi*ti*(1 - (4*alpha*ti)**2)
                h[i] = num/den
        # Normalize energy
        h = h / np.sqrt(np.sum(h**2))
        return h

    # ---------------- FIR engines ----------------
    def _fir_float(self, x):
        """Complex float FIR with self.h_float. Returns len(x) + len(h) - 1 samples."""
        yi = np.convolve(np.real(x), self.h_float, mode='full')
        yq = np.convolve(np.imag(x), self.h_float, mode='full')
        return yi + 1j*yq

    def _fir_fixed_q15(self, x):
        """
        Complex fixed-point FIR:
        - x is complex float, will be quantized to Q1.15 ints.
        - h_int are Q1.15 ints.
        - Accumulate in acc_bits signed integer.
        - Right-shift by coeff_frac to bring back to Q1.15.
        - Return float (converted from Q1.15).
        """
        xi_int = self._float_to_q(np.real(x), self.data_bits, self.data_frac, np.int32)
        xq_int = self._float_to_q(np.imag(x), self.data_bits, self.data_frac, np.int32)
        h_int = self.h_int  # Q1.15

        Lx = len(xi_int)
        Lh = len(h_int)
        Ly = Lx + Lh - 1

        yi_int = np.zeros(Ly, dtype=np.int64)
        yq_int = np.zeros(Ly, dtype=np.int64)

        for n in range(Ly):
            acc_i = np.int64(0)
            acc_q = np.int64(0)
            # FIR sum
            kmin = max(0, n - (Lx - 1))
            kmax = min(n, Lh - 1)
            for k in range(kmin, kmax + 1):
                x_idx = n - k
                prod_i = np.int64(xi_int[x_idx]) * np.int64(h_int[k])  # scale: 2^(data_frac+coeff_frac)
                prod_q = np.int64(xq_int[x_idx]) * np.int64(h_int[k])
                acc_i += prod_i
                acc_q += prod_q
                # Optional: saturate per addition (simulate narrower DSP slice)
                acc_i = self._saturate_signed(acc_i, self.acc_bits)
                acc_q = self._saturate_signed(acc_q, self.acc_bits)
            # Align back to Q1.15 by shifting right coeff_frac (keep data_frac scaling)
            acc_i = acc_i >> self.coeff_frac
            acc_q = acc_q >> self.coeff_frac
            # Final saturation to 16-bit output
            acc_i = self._saturate_signed(acc_i, self.data_bits)
            acc_q = self._saturate_signed(acc_q, self.data_bits)
            yi_int[n] = acc_i
            yq_int[n] = acc_q

        # Convert Q1.15 ints to float
        y_float = self._q_to_float(yi_int, self.data_frac) + 1j*self._q_to_float(yq_int, self.data_frac)
        return y_float

    # ---------------- Public API ----------------
    def transmit(self, bits, fixed_point=False, seed=0x7F):
        """
        bits: 0/1 array-like
        fixed_point: False => float pipeline, True => fixed pipeline
        Returns: complex float waveform (same units), and dict with metadata
        """
        bits = np.asarray(bits).astype(np.int8).flatten()
        # Scramble
        scram = self._lfsr_scramble(bits, seed=seed)
        # Differential DBPSK symbols (complex unit)
        syms = self._dbpsk_diff_encode(scram)
        # Spread by Barker-11
        chips = self._spread_barker(syms)
        # Upsample by sps
        up = self._upsample_by_sps(chips, self.sps)

        if fixed_point:
            y = self._fir_fixed_q15(up)  # returns complex float
            meta = {"mode": "fixed", "q_format": "Q1.15", "acc_bits": self.acc_bits}
        else:
            y = self._fir_float(up)      # complex float
            meta = {"mode": "float"}

        # Remove group delay to roughly align with "lfilter"-style latency
        gd = (len(self.h_float) - 1) // 2
        y = y[gd:]
        return y, meta

    @staticmethod
    def snr_db(ref, test, eps=1e-20):
        """Compute SNR(ref vs test) in dB: 10*log10(||ref||^2 / ||ref-test||^2)."""
        L = min(len(ref), len(test))
        ref = ref[:L]
        test = test[:L]
        num = np.sum(np.abs(ref)**2) + eps
        den = np.sum(np.abs(ref - test)**2) + eps
        return 10.0 * np.log10(num / den)


if __name__ == "__main__":
    # Quick self-test: compare float vs fixed and print SNR
    np.random.seed(0)
    tx = IEEE80211bTransmitterDualMode(sps=8, rrc_span_chips=6, rrc_alpha=0.35,
                                       data_bits=16, coeff_bits=16, acc_bits=32)
    bits = np.random.randint(0, 2, 200)

    y_float, _ = tx.transmit(bits, fixed_point=False)
    y_fixed, meta = tx.transmit(bits, fixed_point=True)

    snr = tx.snr_db(y_float, y_fixed)
    print(f"Fixed-point vs Float SNR: {snr:.2f} dB   (mode={meta['mode']}, {meta.get('q_format','')}, acc_bits={meta.get('acc_bits','-')})")
