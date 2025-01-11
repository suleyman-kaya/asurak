from rtlsdr import RtlSdr
import numpy as np
import matplotlib.pyplot as plt
import time
from threading import Thread
from scipy.io.wavfile import write

fft_size = 512

# RTL-SDR ayarları
def configure_sdr():
    sdr = RtlSdr()
    sdr.sample_rate = 2.048e6  # 2.048 MHz
    sdr.center_freq = 145500000  # 145.500 MHz
    sdr.freq_correction = 60  # PPM düzeltmesi
    sdr.gain = 'auto'  # Otomatik kazanç
    return sdr

def capture_spectrum(sdr, duration_sec, fft_size=512):
    num_samples = int(duration_sec * sdr.sample_rate)
    collected_samples = []

    print(f"Spektrum analizi başlıyor: {duration_sec} saniye boyunca dinleniyor...")
    start_time = time.time()

    while len(collected_samples) < num_samples:
        samples = sdr.read_samples(fft_size)
        collected_samples.extend(samples)

        elapsed_time = time.time() - start_time
        if elapsed_time >= duration_sec:
            break

    collected_samples = np.array(collected_samples[:num_samples])  # Fazla örnekleri kes
    return collected_samples

def save_wave_file(samples, sample_rate, filename="spectrum_audio.wav"):
    print("Sinyaller .wav dosyası olarak kaydediliyor...")
    # Veriyi normalize edip 16-bit PCM formatına dönüştür
    samples_normalized = np.int16(np.real(samples) / np.max(np.abs(np.real(samples))) * 32767)
    write(filename, int(sample_rate), samples_normalized)
    print(f".wav dosyası kaydedildi: {filename}")

def save_spectrum_figure(spectrogram, sdr, filename="spectrum_analysis.png"):
    extent = [
        (sdr.center_freq - sdr.sample_rate*10 / 2) / 1e6,  # Sol frekans sınırı (MHz)
        (sdr.center_freq + sdr.sample_rate*10 / 2) / 1e6,  # Sağ frekans sınırı (MHz)
        0,  # Zaman başlangıcı
        spectrogram.shape[0] / (sdr.sample_rate / fft_size)  # Zaman (saniye)
    ]

    # Spektrum sınırlarını normalize et ve renk skalasını ayarla
    spectrogram = np.clip(spectrogram, -120, 60)  # Minimum ve maksimum değerler

    plt.figure(figsize=(10, 6))
    plt.imshow(spectrogram, aspect='auto', extent=extent)
    plt.xlabel("Frekans [MHz]")
    plt.ylabel("Zaman [s]")
    plt.title("145.500 MHz Spektrum Analizi")
    plt.savefig(filename, bbox_inches='tight')
    plt.close()
    print(f"Spektrum analizi kaydedildi: {filename}")

def display_timer(duration_sec):
    for remaining in range(duration_sec, 0, -1):
        print(f"Kalan süre: {remaining} saniye", end="\r")
        time.sleep(1)
    print("\nDinleme tamamlandı.")

def main():
    sdr = configure_sdr()

    try:
        duration_sec = 12

        # Sayaç için ayrı bir thread başlat
        timer_thread = Thread(target=display_timer, args=(duration_sec,))
        timer_thread.start()

        # Spektrum verisini ve sinyalleri topla
        samples = capture_spectrum(sdr, duration_sec=duration_sec)

        # Sayaç tamamlanana kadar bekle
        timer_thread.join()

        # Sonuçları kaydet
        save_wave_file(samples, sdr.sample_rate, filename="spectrum_145500MHz.wav")
    finally:
        # SDR cihazını kapat
        sdr.close()

if __name__ == "__main__":
    main()
