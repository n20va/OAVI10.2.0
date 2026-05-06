from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from scipy.signal import stft, resample_poly

SR = 22050
CROSSFADE_MS = 90
TRIM_TOP_DB = 28
EDGE_FADE_MS = 10

BASE_DIR = Path(__file__).resolve().parent
PHONEMES_DIR = BASE_DIR / "phonemes"
RESULTS_DIR = BASE_DIR / "results"

PHONEMES = [
    "a", "ae", "aa", "schwa", "e", "eh",
    "i", "y", "o", "yo", "u", "uu",
    "a_red", "schwa2", "i_red", "y_red",
    "y_diph", "y_mid", "y_c", "u_red",
    "u_soft", "i_short", "u_short",
    "b", "b_soft", "v", "v_soft",
    "g", "g_soft", "d", "d_soft",
    "zh", "zh_soft", "z", "z_soft",
    "j", "k", "k_soft", "l",
    "l_soft", "m", "m_soft", "n",
    "n_soft", "p", "p_soft", "r",
    "r_soft", "s", "s_soft", "t",
    "t_soft", "f", "f_soft", "h",
    "h_soft", "dz", "ch", "dj",
    "c", "sh", "sch"
]

TRANSCRIPTION = [
    "h", "a_red", "r", "a_red", "sh", "o",
    "zh", "y_mid", "v_soft", "o", "t",
    "n", "a_red",
    "s", "v_soft", "e", "t_soft", "i_red",
    "v_soft", "i", "n", "n_soft", "i_red",
    "p", "u", "h"
]


def normalize(audio: np.ndarray) -> np.ndarray:
    peak = float(np.max(np.abs(audio))) if len(audio) else 0.0
    if peak > 0:
        audio = audio / peak
    return audio.astype(np.float32)


def trim_silence(audio: np.ndarray, top_db: float = TRIM_TOP_DB) -> np.ndarray:
    """Удаляет тишину в начале и конце файла без зависимости от librosa."""
    if len(audio) == 0:
        return audio

    peak = float(np.max(np.abs(audio)))
    if peak <= 0:
        return audio

    threshold = peak * (10 ** (-top_db / 20))
    active = np.where(np.abs(audio) > threshold)[0]

    if len(active) == 0:
        return audio

    start = int(active[0])
    end = int(active[-1]) + 1

    padding = int(0.008 * SR)
    start = max(0, start - padding)
    end = min(len(audio), end + padding)

    return audio[start:end]


def apply_edge_fade(audio: np.ndarray, fade_ms: int = EDGE_FADE_MS) -> np.ndarray:
    """Добавляет короткие fade-in/fade-out, чтобы убрать щелчки на стыках."""
    fade = int(SR * fade_ms / 1000)

    if fade <= 1 or len(audio) <= fade * 2:
        return audio.astype(np.float32)

    result = audio.copy().astype(np.float32)
    result[:fade] *= np.linspace(0, 1, fade, dtype=np.float32)
    result[-fade:] *= np.linspace(1, 0, fade, dtype=np.float32)

    return result


def load_wav(path: Path) -> np.ndarray:
    audio, sr = sf.read(path)

    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    audio = audio.astype(np.float32)

    if sr != SR:
        gcd = np.gcd(sr, SR)
        audio = resample_poly(audio, SR // gcd, sr // gcd).astype(np.float32)

    audio = trim_silence(audio)
    audio = normalize(audio)
    audio = apply_edge_fade(audio)
    audio = normalize(audio)

    return audio.astype(np.float32)


def load_phonemes() -> dict[str, np.ndarray]:
    missing = []

    for phoneme in PHONEMES:
        path = PHONEMES_DIR / f"{phoneme}.wav"
        if not path.exists():
            missing.append(path.name)

    if missing:
        raise FileNotFoundError(
            "Не найдены файлы фонем в каталоге phonemes:\n" +
            "\n".join(missing)
        )

    return {
        phoneme: load_wav(PHONEMES_DIR / f"{phoneme}.wav")
        for phoneme in PHONEMES
    }


def concatenate_simple(
        sequence: list[str],
        phoneme_audio: dict[str, np.ndarray]
) -> np.ndarray:
    return normalize(np.concatenate([phoneme_audio[p] for p in sequence]))


def concatenate_crossfade(
        sequence: list[str],
        phoneme_audio: dict[str, np.ndarray],
        crossfade_ms: int = CROSSFADE_MS
) -> np.ndarray:
    crossfade = int(SR * crossfade_ms / 1000)
    result = phoneme_audio[sequence[0]].copy()

    for phoneme in sequence[1:]:
        next_audio = phoneme_audio[phoneme].copy()
        current_crossfade = min(crossfade, len(result), len(next_audio))

        if current_crossfade <= 1:
            result = np.concatenate([result, next_audio])
            continue

        left = result[-current_crossfade:]
        right = next_audio[:current_crossfade]

        fade_out = np.linspace(1, 0, current_crossfade, dtype=np.float32)
        fade_in = np.linspace(0, 1, current_crossfade, dtype=np.float32)
        mixed = left * fade_out + right * fade_in

        result = np.concatenate([
            result[:-current_crossfade],
            mixed,
            next_audio[current_crossfade:]
        ])

    return normalize(result)


def save_spectrogram(audio: np.ndarray, path: Path, title: str) -> None:
    f, t, z = stft(
        audio,
        fs=SR,
        window="hann",
        nperseg=2048,
        noverlap=1536
    )

    magnitude = 20 * np.log10(np.abs(z) + 1e-10)
    f_plot = f.copy()

    if len(f_plot) > 1 and f_plot[0] == 0:
        f_plot[0] = f_plot[1] / 2

    plt.figure(figsize=(14, 6))
    plt.pcolormesh(t, f_plot, magnitude, shading="gouraud")
    plt.yscale("log")
    plt.xlabel("Время, с")
    plt.ylabel("Частота, Гц")
    plt.title(title)
    plt.colorbar(label="Амплитуда, dB")
    plt.tight_layout()
    plt.savefig(path, dpi=250)
    plt.close()


def save_waveform(audio: np.ndarray, path: Path, title: str) -> None:
    t = np.arange(len(audio)) / SR

    plt.figure(figsize=(14, 5))
    plt.plot(t, audio, linewidth=0.7)
    plt.xlabel("Время, с")
    plt.ylabel("Амплитуда")
    plt.title(title)
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(path, dpi=250)
    plt.close()


def save_waveform_compare(
        simple_audio: np.ndarray,
        crossfade_audio: np.ndarray,
        path: Path
) -> None:
    t1 = np.arange(len(simple_audio)) / SR
    t2 = np.arange(len(crossfade_audio)) / SR

    plt.figure(figsize=(14, 6))
    plt.plot(t1, simple_audio, linewidth=0.7, label="Простая конкатенация")
    plt.plot(t2, crossfade_audio, linewidth=0.7, alpha=0.8, label="Crossfade")
    plt.xlabel("Время, с")
    plt.ylabel("Амплитуда")
    plt.title("Сравнение синтезированных сигналов")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=250)
    plt.close()


def create_report(simple_audio: np.ndarray, crossfade_audio: np.ndarray) -> str:
    transcription_text = "-".join(TRANSCRIPTION)

    return f"""# Лабораторная работа №10
# Обработка голоса
## Вариант 2 — Синтезатор речи



## Исходные данные

| Параметр | Значение |
|---|---:|
| Формат аудио | WAV, mono |
| Частота дискретизации | {SR} Гц |
| Количество фонем и аллофонов | {len(PHONEMES)} |
| Длительность crossfade | {CROSSFADE_MS} мс |
| Порог удаления тишины | {TRIM_TOP_DB} dB |
| Синтезируемая фраза | Хорошо живёт на свете Винни-Пух |

---


## Фонетическая цепочка

```text
{transcription_text}
```

---

## Полученные аудиофайлы

| Метод | Файл | Длительность |
|---|---|---:|
| Простая конкатенация | `results/simple_concat.wav` | {len(simple_audio) / SR:.3f} с |
| Crossfade | `results/crossfade_concat.wav` | {len(crossfade_audio) / SR:.3f} с |

---

## Осциллограмма простой конкатенации

![Простая конкатенация](waveform_simple.png)

---

## Осциллограмма crossfade

![Crossfade](waveform_crossfade.png)

---

## Сравнение сигналов

![Сравнение сигналов](waveforms_compare.png)

---

## Спектрограмма простой конкатенации

![Спектрограмма простой конкатенации](spectrogram_simple.png)

---

## Спектрограмма crossfade

![Спектрограмма crossfade](spectrogram_crossfade.png)

---

## Сравнение методов

| Метод | Преимущества | Недостатки |
|---|---|---|
| Простая конкатенация | Простой алгоритм, высокая скорость | На границах фонем могут быть слышны паузы и щелчки |
| Crossfade | Плавные переходы, меньше слышимых стыков | Требуется дополнительная обработка сигнала |

---

## Вывод

В ходе лабораторной работы был реализован синтезатор речи на основе записанных фонем и аллофонов русского языка. Фраза «Хорошо живёт на свете Винни-Пух» была синтезирована двумя способами: простой конкатенацией и методом crossfade.

После добавления удаления тишины, нормализации и плавного затухания пробелы между фонемами стали менее заметными. Наиболее качественный результат получен при использовании crossfade, так как он обеспечивает более плавные переходы между соседними звуковыми фрагментами.
"""


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)

    phoneme_audio = load_phonemes()

    simple_audio = concatenate_simple(TRANSCRIPTION, phoneme_audio)
    crossfade_audio = concatenate_crossfade(TRANSCRIPTION, phoneme_audio)

    sf.write(RESULTS_DIR / "simple_concat.wav", simple_audio, SR)
    sf.write(RESULTS_DIR / "crossfade_concat.wav", crossfade_audio, SR)

    save_waveform(simple_audio, RESULTS_DIR / "waveform_simple.png", "Осциллограмма простой конкатенации")
    save_waveform(crossfade_audio, RESULTS_DIR / "waveform_crossfade.png", "Осциллограмма crossfade")
    save_waveform_compare(simple_audio, crossfade_audio, RESULTS_DIR / "waveforms_compare.png")

    save_spectrogram(simple_audio, RESULTS_DIR / "spectrogram_simple.png", "Спектрограмма простой конкатенации")
    save_spectrogram(crossfade_audio, RESULTS_DIR / "spectrogram_crossfade.png", "Спектрограмма crossfade")

    report = create_report(simple_audio, crossfade_audio)
    (RESULTS_DIR / "report.md").write_text(report, encoding="utf-8")

    print("Лабораторная работа выполнена.")
    print(f"Фонем загружено: {len(phoneme_audio)}")
    print(f"Простая конкатенация: {RESULTS_DIR / 'simple_concat.wav'}")
    print(f"Crossfade: {RESULTS_DIR / 'crossfade_concat.wav'}")
    print(f"Отчёт: {RESULTS_DIR / 'report.md'}")


if __name__ == "__main__":
    main()
