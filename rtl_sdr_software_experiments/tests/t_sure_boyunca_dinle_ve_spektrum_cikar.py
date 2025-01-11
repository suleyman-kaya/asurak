from rtlsdr import RtlSdr
import numpy as np
import matplotlib.pyplot as plt
import time
from threading import Thread

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
    num_rows = int((duration_sec * sdr.sample_rate) / fft_size)
    spectrogram = np.zeros((num_rows, fft_size))

    print(f"Spektrum analizi başlıyor: {duration_sec} saniye boyunca dinleniyor...")
    start_time = time.time()

    for i in range(num_rows):
        samples = sdr.read_samples(fft_size)
        spectrum = 10 * np.log10(np.abs(np.fft.fftshift(np.fft.fft(samples))) + 1e-9)  # Epsilon eklendi
        spectrogram[i, :] = spectrum

        elapsed_time = time.time() - start_time
        if elapsed_time >= duration_sec:
            break

    return spectrogram

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
        # Sayaç için ayrı bir thread başlat
        duration_sec = 12
        timer_thread = Thread(target=display_timer, args=(duration_sec,))
        timer_thread.start()

        # Spektrum verisini topla
        spectrogram = capture_spectrum(sdr, duration_sec=duration_sec)

        # Sayaç tamamlanana kadar bekle
        timer_thread.join()

        # Sonucu kaydet
        save_spectrum_figure(spectrogram, sdr, filename="spectrum_145500MHz.png")
    finally:
        # SDR cihazını kapat
        sdr.close()

if __name__ == "__main__":
    main()
