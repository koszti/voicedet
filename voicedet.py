import math
import numpy
import sounddevice

from scipy.io.wavfile import write
from scipy.signal import fftconvolve

DEFAULT_SAMPLERATE = 44100
DEFAULT_CHANNELS = 2


class NoteDetection:
    def __init__(self, samplerate=DEFAULT_SAMPLERATE, channels=DEFAULT_CHANNELS):
        self.samplerate = samplerate
        self.channels = channels
        self.output = "output.wav"

    @staticmethod
    def parabolic(f, x):
        xv = 1 / 2.0 * (f[x - 1] - f[x + 1]) / (f[x - 1] - 2 * f[x] + f[x + 1]) + x
        yv = f[x] - 1 / 4.0 * (f[x - 1] - f[x + 1]) * (xv - x)
        return (xv, yv)

    @staticmethod
    def find(condition):
        (res,) = numpy.nonzero(numpy.ravel(condition))
        return res

    def record(self, seconds):
        print(f"Starting to record {seconds} seconds...")
        rec = sounddevice.rec(int(seconds * self.samplerate), samplerate=self.samplerate, channels=self.channels)
        sounddevice.wait()

        return rec

    def play(self, rec):
        print(f"Playing...")
        sounddevice.play(rec, samplerate=self.samplerate)
        sounddevice.wait()

    def loudness(self, rec):
        data = numpy.array(rec, dtype=float) / 32768.0
        ms = math.sqrt(numpy.sum(data**2.0) / len(data))
        if ms < 10e-8:
            ms = 10e-8
        return 10.0 * math.log(ms, 10.0)

    def note(self, rec):

        try:
            corr = fftconvolve(rec, rec[::-1], mode="full")
            corr = corr[int(len(corr) / 2) :]
            d = numpy.diff(corr)
            start = NoteDetection.find(d > 0)[0]
            peak = numpy.argmax(corr[start:]) + start
            px, py = NoteDetection.parabolic(corr, peak)
            note = self.samplerate / px
        except Exception as exc:
            print(exc)
            note = 0

        return note


if __name__ == "__main__":
    nd = NoteDetection()

    while True:
        rec = nd.record(seconds=0.5)
        signal_level = nd.loudness(rec)
        inputnote = nd.note(rec)

        print(f"signal_level: {signal_level} - inputnote: {inputnote}")

        nd.play(rec)
