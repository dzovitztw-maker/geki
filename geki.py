#!/usr/bin/env python3

import colorsys
import json
import math
import os
import plistlib
import select
import shutil
import struct
import subprocess
import termios
import sys
import threading
import time
import tty
from urllib.parse import urlparse, parse_qs

import numpy as np
import sounddevice as sd
import soundfile as sf


# ═════════════════════════════════════════════════════════════
# GEKI
# Terminal Audio Visualizer
#
# v0.2.2 — Corrected FFT / Audio Analysis
# ═════════════════════════════════════════════════════════════

FPS = 30

BLOCK = "█"
SOFT = "░"


# ═════════════════════════════════════════════════════════════
# TERMINAL
# ═════════════════════════════════════════════════════════════

def clear():
    """Efface le terminal."""
    sys.stdout.write("\033[2J\033[H")
    sys.stdout.flush()


def hide_cursor():
    """Cache le curseur."""
    sys.stdout.write("\033[?25l")
    sys.stdout.flush()


def show_cursor():
    """Affiche le curseur."""
    sys.stdout.write("\033[?25h")
    sys.stdout.flush()


def terminal_size():
    """Retourne la taille actuelle du terminal."""
    return shutil.get_terminal_size((100, 30))


def color(r, g, b, text):
    """Couleur TrueColor ANSI."""
    return f"\033[38;2;{r};{g};{b}m{text}\033[0m"


# ═════════════════════════════════════════════════════════════
# AUDIO ENGINE
# ═════════════════════════════════════════════════════════════

class AudioEngine:

    def __init__(self, filename=None, sample_rate=None):

        # ─────────────────────────────────────────
        # Chargement du fichier
        # ─────────────────────────────────────────

        if filename is not None:
            self.data, self.sample_rate = sf.read(
                filename,
                dtype="float32",
                always_2d=True
            )

            # Conversion en mono pour l'analyse FFT.
            self.mono = np.mean(
                self.data,
                axis=1
            )
            self.total_samples = len(self.mono)
        else:
            self.data = None
            self.sample_rate = sample_rate
            self.mono = None
            self.total_samples = 0

        self.position = 0
        self.playing = True

        # ─────────────────────────────────────────
        # FFT
        # ─────────────────────────────────────────

        self.fft_size = 2048

        # ─────────────────────────────────────────
        # Valeurs audio
        # ─────────────────────────────────────────

        self.bass = 0.0
        self.mid = 0.0
        self.treble = 0.0
        self.energy = 0.0

        # Sensibilité globale.
        self.sensitivity = 1.15

        # Plafond adaptatif par bande (AGC).
        # Suit le pic le plus fort récent et redescend
        # lentement — voir adaptive_normalize() dans analyze().
        self.bass_ceiling = -40.0
        self.mid_ceiling = -40.0
        self.treble_ceiling = -40.0

        # ─────────────────────────────────────────
        # Spectre multi-bandes (pour le starburst)
        #
        # 48 bandes log-espacées de 30 Hz à 16 kHz.
        # Beaucoup plus de détail que bass/mid/treble,
        # avec sa propre normalisation adaptative par bin
        # et un peak-hold pour l'effet "coup" visuel.
        # ─────────────────────────────────────────

        self.num_bins = 48

        freqs_full = np.fft.rfftfreq(
            self.fft_size,
            1 / self.sample_rate
        )

        edges = np.geomspace(
            30,
            min(16000, self.sample_rate / 2 - 1),
            self.num_bins + 1
        )

        self.bin_masks = []

        for i in range(self.num_bins):

            lo = edges[i]
            hi = edges[i + 1]

            mask = (freqs_full >= lo) & (freqs_full < hi)

            if not mask.any():

                # Tranche trop étroite pour contenir un bin FFT (ça
                # arrive côté graves : l'écart log y est parfois plus
                # fin que la résolution de la FFT). Sans ce filet,
                # cette tranche resterait à 0 en permanence, quel que
                # soit le morceau. On la rattache au bin FFT le plus
                # proche de son centre pour qu'elle reste toujours
                # vivante.

                center = (lo + hi) / 2.0
                nearest_index = int(np.argmin(np.abs(freqs_full - center)))

                mask = np.zeros_like(freqs_full, dtype=bool)
                mask[nearest_index] = True

            self.bin_masks.append(mask)

        self.spectrum = np.zeros(self.num_bins)
        self.spectrum_ceiling = np.full(self.num_bins, -40.0)
        self.spectrum_peak = np.zeros(self.num_bins)

        # État pour la détection de kick (thème "shockwave").
        self.prev_bass_raw = 0.0
        self.last_kick_time = 0.0
        self.shockwaves = []

        # ─────────────────────────────────────────
        # Audio output
        # ─────────────────────────────────────────

        if filename is not None:
            self.stream = sd.OutputStream(
                samplerate=self.sample_rate,
                channels=self.data.shape[1],
                dtype="float32",
                callback=self.audio_callback,
                blocksize=1024
            )

    # ═════════════════════════════════════════════
    # AUDIO CALLBACK
    # ═════════════════════════════════════════════

    def audio_callback(
        self,
        outdata,
        frames,
        time_info,
        status
    ):

        if not self.playing:
            outdata.fill(0)
            return

        end = self.position + frames

        chunk = self.data[
            self.position:end
        ]

        # Fin du morceau.
        if len(chunk) < frames:

            if len(chunk) > 0:
                outdata[:len(chunk)] = chunk

            outdata[len(chunk):] = 0

            self.position = self.total_samples
            self.playing = False

            return

        outdata[:] = chunk
        self.position = end

    # ═════════════════════════════════════════════
    # FFT / AUDIO ANALYSIS
    # ═════════════════════════════════════════════

    def analysis_chunk(self):
        start = max(0, self.position - self.fft_size)
        return self.mono[start:self.position]

    def analyze(self):

        # Récupération de la fenêtre audio récente.
        chunk = self.analysis_chunk()

        if len(chunk) < self.fft_size:
            return

        # Fenêtre Hann.
        window = np.hanning(
            len(chunk)
        )

        signal = chunk * window
        n = len(signal)

        # ─────────────────────────────────────────
        # FFT correctement normalisée
        # ─────────────────────────────────────────

        spectrum = np.fft.rfft(
            signal
        )

        magnitude = (
            np.abs(spectrum)
            / n
        )

        # Compensation approximative du gain
        # perdu avec la fenêtre Hann.
        magnitude *= 2.0

        frequencies = np.fft.rfftfreq(
            n,
            1 / self.sample_rate
        )

        # ─────────────────────────────────────────
        # Bandes de fréquences
        # ─────────────────────────────────────────

        bass_mask = (
            (frequencies >= 30)
            &
            (frequencies < 250)
        )

        mid_mask = (
            (frequencies >= 250)
            &
            (frequencies < 4000)
        )

        treble_mask = (
            (frequencies >= 4000)
            &
            (frequencies < 16000)
        )

        # ─────────────────────────────────────────
        # Mesure RMS par bande
        # ─────────────────────────────────────────

        def band_level(mask):

            values = magnitude[mask]

            if len(values) == 0:
                return 0.0

            return float(
                np.sqrt(
                    np.mean(
                        values ** 2
                    )
                )
            )

        bass_raw = band_level(
            bass_mask
        )

        mid_raw = band_level(
            mid_mask
        )

        treble_raw = band_level(
            treble_mask
        )

        # ─────────────────────────────────────────
        # Conversion en dB
        # ─────────────────────────────────────────

        def to_db(value):

            return 20.0 * math.log10(
                max(
                    value,
                    1e-7
                )
            )

        bass_db = to_db(
            bass_raw
        )

        mid_db = to_db(
            mid_raw
        )

        treble_db = to_db(
            treble_raw
        )

        # ─────────────────────────────────────────
        # Normalisation adaptative (AGC par bande)
        #
        # Le plafond ("ceiling") suit le niveau le plus
        # fort récemment observé sur chaque bande, et
        # redescend lentement quand il n'est plus atteint.
        # Le visualizer s'adapte ainsi automatiquement à la
        # dynamique de n'importe quel morceau, au lieu de
        # saturer en permanence sur les basses avec un
        # seuil fixe.
        # ─────────────────────────────────────────

        floor = -70.0
        min_ceiling = -45.0
        ceiling_release = 0.15  # dB perdus par frame sans nouveau pic

        if bass_db > self.bass_ceiling:
            self.bass_ceiling = bass_db
        else:
            self.bass_ceiling = max(
                self.bass_ceiling - ceiling_release,
                min_ceiling
            )

        bass = float(
            np.clip(
                (bass_db - floor) / (self.bass_ceiling - floor),
                0.0,
                1.0
            )
        )

        if mid_db > self.mid_ceiling:
            self.mid_ceiling = mid_db
        else:
            self.mid_ceiling = max(
                self.mid_ceiling - ceiling_release,
                min_ceiling
            )

        mid = float(
            np.clip(
                (mid_db - floor) / (self.mid_ceiling - floor),
                0.0,
                1.0
            )
        )

        if treble_db > self.treble_ceiling:
            self.treble_ceiling = treble_db
        else:
            self.treble_ceiling = max(
                self.treble_ceiling - ceiling_release,
                min_ceiling
            )

        treble = float(
            np.clip(
                (treble_db - floor) / (self.treble_ceiling - floor),
                0.0,
                1.0
            )
        )

        # ─────────────────────────────────────────
        # Courbe de sensibilité
        # ─────────────────────────────────────────

        bass = (
            bass ** 0.75
        ) * self.sensitivity

        mid = (
            mid ** 0.82
        ) * self.sensitivity

        treble = (
            treble ** 0.90
        ) * self.sensitivity

        bass = float(
            np.clip(
                bass,
                0.0,
                1.0
            )
        )

        mid = float(
            np.clip(
                mid,
                0.0,
                1.0
            )
        )

        treble = float(
            np.clip(
                treble,
                0.0,
                1.0
            )
        )

        # ─────────────────────────────────────────
        # Smoothing
        #
        # Montée rapide, descente plus lente.
        # ─────────────────────────────────────────

        attack = 0.35
        release = 0.08

        def smooth(old, new):

            if new > old:
                speed = attack
            else:
                speed = release

            return (
                old * (1.0 - speed)
                +
                new * speed
            )

        self.bass = smooth(
            self.bass,
            bass
        )

        self.mid = smooth(
            self.mid,
            mid
        )

        self.treble = smooth(
            self.treble,
            treble
        )

        # Énergie globale.
        self.energy = (
            self.bass * 0.55
            +
            self.mid * 0.30
            +
            self.treble * 0.15
        )

        # ─────────────────────────────────────────
        # Spectre multi-bandes (starburst)
        # ─────────────────────────────────────────

        spectrum_raw = np.array([
            np.sqrt(np.mean(magnitude[mask] ** 2)) if mask.any() else 0.0
            for mask in self.bin_masks
        ])

        spectrum_db = 20.0 * np.log10(np.maximum(spectrum_raw, 1e-7))

        spec_min_ceiling = -45.0
        spec_release = 0.20

        rising = spectrum_db > self.spectrum_ceiling

        self.spectrum_ceiling = np.where(
            rising,
            spectrum_db,
            np.maximum(self.spectrum_ceiling - spec_release, spec_min_ceiling)
        )

        spectrum_norm = np.clip(
            (spectrum_db - floor) / (self.spectrum_ceiling - floor),
            0.0,
            1.0
        )

        # Un peu de contraste (fait ressortir les pics nets).
        spectrum_norm = spectrum_norm ** 0.8

        # Lissage percussif : montée quasi instantanée,
        # descente franche (pas de "flottement" comme le blob).
        spec_attack = 0.75
        spec_release_smooth = 0.30

        rising_smooth = spectrum_norm > self.spectrum

        self.spectrum = np.where(
            rising_smooth,
            self.spectrum * (1 - spec_attack) + spectrum_norm * spec_attack,
            self.spectrum * (1 - spec_release_smooth) + spectrum_norm * spec_release_smooth
        )

        # Peak-hold : petit marqueur qui grimpe avec le pic
        # et retombe lentement (effet VU-mètre / "coup").
        peak_fall = 0.025

        self.spectrum_peak = np.maximum(
            self.spectrum,
            self.spectrum_peak - peak_fall
        )

        # ─────────────────────────────────────────
        # Détection de kick (front montant du bass BRUT)
        #
        # Utilisée par le thème "shockwave". Important : on utilise
        # ici la variable locale `bass` (avant le lissage attack/
        # release qui alimente self.bass), parce que self.bass a
        # une release lente et redescend rarement sous le seuil
        # entre deux kicks rapprochés — le front montant ne se
        # redéclencherait quasiment jamais. Le bass brut, lui, suit
        # l'énergie instantanée frame par frame et retombe bien
        # entre deux coups.
        # ─────────────────────────────────────────

        now = time.perf_counter()

        kick_threshold = 0.5
        kick_cooldown = 0.12

        if (
            bass > kick_threshold
            and self.prev_bass_raw <= kick_threshold
            and (now - self.last_kick_time) > kick_cooldown
        ):
            self.shockwaves.append(now)
            self.last_kick_time = now

        self.prev_bass_raw = bass

        # Purge des anneaux dont la durée de vie visuelle est dépassée.
        self.shockwaves = [
            spawn_time for spawn_time in self.shockwaves
            if now - spawn_time < 1.2
        ]

    # ═════════════════════════════════════════════
    # TEMPS
    # ═════════════════════════════════════════════

    def position_seconds(self):

        return (
            self.position
            /
            self.sample_rate
        )

    def duration_seconds(self):

        return (
            self.total_samples
            /
            self.sample_rate
        )

    # ═════════════════════════════════════════════
    # PLAYBACK
    # ═════════════════════════════════════════════

    def start(self):
        self.stream.start()

    def stop(self):
        self.stream.stop()
        self.stream.close()


# ═════════════════════════════════════════════════════════════
# HELPERS
# ═════════════════════════════════════════════════════════════

def format_time(seconds):

    seconds = int(
        max(0, seconds)
    )

    minutes = seconds // 60
    seconds = seconds % 60

    return (
        f"{minutes:02d}:"
        f"{seconds:02d}"
    )


# ═════════════════════════════════════════════════════════════
# GEKI STARBURST
# ═════════════════════════════════════════════════════════════
#
# Un noyau central qui pulse sur les basses, entouré de pointes
# radiales (une par bin de fréquence) qui jaillissent du centre.
# Montée quasi instantanée + peak-hold => rendu "coup" plutôt
# qu'organique/lissé.
# ═════════════════════════════════════════════════════════════

def _spike_color(bin_fraction, intensity):
    """Couleur d'une pointe selon sa position fréquentielle (0=grave, 1=aigu)."""

    r = 255 * max(0.0, 1.0 - bin_fraction * 1.5)
    g = 255 * max(0.0, 1.0 - abs(bin_fraction - 0.5) * 1.7)
    b = 255 * max(0.0, (bin_fraction - 0.25) * 1.4)

    r = 60 + r * 0.75
    g = 25 + g * 0.75
    b = 90 + b * 0.75

    brightness = 0.45 + 0.55 * intensity

    return (
        int(min(255, r * brightness)),
        int(min(255, g * brightness)),
        int(min(255, b * brightness))
    )


def make_visualizer_starburst(width, height, audio, t):

    cx = width / 2.0
    cy = height / 2.0

    buffer = [[None] * width for _ in range(height)]

    def plot(x, y, char, rgb):
        xi = int(round(x))
        yi = int(round(y))
        if 0 <= xi < width and 0 <= yi < height:
            buffer[yi][xi] = (char, rgb)

    max_rx = width * 0.47
    max_ry = height * 0.47

    # ─────────────────────────────────────────
    # Noyau central (pulse sur les basses)
    # ─────────────────────────────────────────

    core_energy = (audio.bass * 0.6 + audio.mid * 0.25 + audio.treble * 0.15) ** 0.75

    core_fx = 0.05 + core_energy * 0.11
    core_rx = max(2, int(max_rx * core_fx))
    core_ry = max(1, int(max_ry * core_fx))

    for y in range(max(0, int(cy - core_ry) - 1), min(height, int(cy + core_ry) + 2)):
        for x in range(max(0, int(cx - core_rx) - 1), min(width, int(cx + core_rx) + 2)):

            dx = (x - cx) / max(core_rx, 1)
            dy = (y - cy) / max(core_ry, 1)
            dist = math.sqrt(dx * dx + dy * dy)

            if dist <= 1.0:
                boost = 1.0 - dist * 0.4
                r = int(min(255, (200 + audio.bass * 55) * boost))
                g = int(min(255, (60 + audio.mid * 120) * boost))
                b = int(min(255, (200 + audio.treble * 55) * boost))
                plot(x, y, BLOCK, (r, g, b))

    # ─────────────────────────────────────────
    # Pointes radiales (une par bin du spectre)
    # ─────────────────────────────────────────

    num_bins = audio.num_bins
    inner_fraction = core_fx * 1.35

    # Rotation continue — vitesse et sens ajustables.
    # Le bass ajoute un petit boost de vitesse sur les gros coups,
    # pour que la rotation "accélère" avec l'énergie du morceau.
    rotation_speed = 0.6
    rotation = t * rotation_speed + audio.bass * 0.15

    for i in range(num_bins):

        level = audio.spectrum[i]
        peak = audio.spectrum_peak[i]
        bin_fraction = i / max(1, num_bins - 1)

        angle = -math.pi / 2 + (2 * math.pi * i / num_bins) + rotation
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)

        tip_fraction = inner_fraction + level * (1.0 - inner_fraction) * 0.9
        color_rgb = _spike_color(bin_fraction, level)

        steps = max(2, int((tip_fraction - inner_fraction) * max(max_rx, max_ry) * 2))

        for s in range(steps + 1):
            f = inner_fraction + (tip_fraction - inner_fraction) * (s / steps)
            x = cx + cos_a * f * max_rx
            y = cy + sin_a * f * max_ry

            char = BLOCK if f < tip_fraction - 0.03 else SOFT
            plot(x, y, char, color_rgb)

        # Peak-hold : petit éclat qui reste en l'air puis retombe.
        peak_fraction = inner_fraction + peak * (1.0 - inner_fraction) * 0.9

        if peak_fraction > tip_fraction + 0.01:
            px = cx + cos_a * peak_fraction * max_rx
            py = cy + sin_a * peak_fraction * max_ry
            plot(px, py, "•", (255, 255, 255))

    # ─────────────────────────────────────────
    # Sérialisation
    # ─────────────────────────────────────────

    lines = []

    for y in range(height):
        line = ""
        for x in range(width):
            cell = buffer[y][x]
            if cell is None:
                line += " "
            else:
                char, rgb = cell
                line += color(rgb[0], rgb[1], rgb[2], char)
        lines.append(line)

    return lines


# ═════════════════════════════════════════════════════════════
# GEKI FLOWER
# ═════════════════════════════════════════════════════════════
#
# Des pétales arrondis (pas des pointes fines) qui poussent
# depuis un cœur central chaud. Chaque pétale répond à un
# groupe de bins du spectre, avec le même peak-hold que le
# starburst pour garder le côté "coup".
# ═════════════════════════════════════════════════════════════

def _petal_color(petal_fraction, intensity):
    """Couleur d'un pétale selon sa position dans le cercle (rose/violet/magenta)."""

    r = 190 + 60 * math.sin(petal_fraction * math.pi * 2)
    g = 90 + 60 * math.sin(petal_fraction * math.pi * 2 + 2.0)
    b = 170 + 70 * math.sin(petal_fraction * math.pi * 2 + 4.0)

    brightness = 0.5 + 0.5 * intensity

    return (
        int(min(255, max(40, r) * brightness)),
        int(min(255, max(20, g) * brightness)),
        int(min(255, max(60, b) * brightness))
    )


def make_visualizer_flower(width, height, audio, t):

    cx = width / 2.0
    cy = height / 2.0

    buffer = [[None] * width for _ in range(height)]

    def plot(x, y, char, rgb):
        xi = int(round(x))
        yi = int(round(y))
        if 0 <= xi < width and 0 <= yi < height:
            buffer[yi][xi] = (char, rgb)

    max_rx = width * 0.47
    max_ry = height * 0.47

    # ─────────────────────────────────────────
    # Cœur central (chaud, pulse sur les basses)
    # ─────────────────────────────────────────

    core_energy = (audio.bass * 0.6 + audio.mid * 0.25 + audio.treble * 0.15) ** 0.75

    core_fx = 0.05 + core_energy * 0.09
    core_rx = max(2, int(max_rx * core_fx))
    core_ry = max(1, int(max_ry * core_fx))

    for y in range(max(0, int(cy - core_ry) - 1), min(height, int(cy + core_ry) + 2)):
        for x in range(max(0, int(cx - core_rx) - 1), min(width, int(cx + core_rx) + 2)):

            dx = (x - cx) / max(core_rx, 1)
            dy = (y - cy) / max(core_ry, 1)
            dist = math.sqrt(dx * dx + dy * dy)

            if dist <= 1.0:
                boost = 1.0 - dist * 0.4
                r = int(min(255, (255 + audio.mid * 0) * boost))
                g = int(min(255, (200 + audio.bass * 55) * boost))
                b = int(min(255, (110 + audio.treble * 60) * boost))
                plot(x, y, BLOCK, (r, g, b))

    # ─────────────────────────────────────────
    # Pétales (groupes de bins du spectre)
    # ─────────────────────────────────────────

    petal_count = 10
    bins_per_petal = max(1, audio.num_bins // petal_count)
    inner_fraction = core_fx * 1.25

    rotation = t * 0.35 + audio.bass * 0.08

    for i in range(petal_count):

        start_bin = i * bins_per_petal
        end_bin = min(audio.num_bins, start_bin + bins_per_petal)

        level = float(np.mean(audio.spectrum[start_bin:end_bin]))
        peak = float(np.mean(audio.spectrum_peak[start_bin:end_bin]))

        petal_fraction = i / petal_count

        angle = -math.pi / 2 + (2 * math.pi * i / petal_count) + rotation
        dir_x = math.cos(angle)
        dir_y = math.sin(angle)
        perp_x = -math.sin(angle)
        perp_y = math.cos(angle)

        length_fraction = inner_fraction + level * (1.0 - inner_fraction) * 0.85
        max_half_width = 0.045 + level * 0.05

        color_rgb = _petal_color(petal_fraction, level)

        steps_along = max(6, int((length_fraction - inner_fraction) * max(max_rx, max_ry) * 2))

        for s in range(steps_along + 1):

            progress = s / steps_along
            f_along = inner_fraction + (length_fraction - inner_fraction) * progress

            # Forme "pétale" : étroit à la base et à la pointe, large au milieu.
            taper = math.sin(math.pi * progress)
            half_width = max_half_width * taper

            steps_across = max(1, int(half_width * max(max_rx, max_ry) * 2))

            for c in range(-steps_across, steps_across + 1):

                across = (c / steps_across) * half_width if steps_across else 0.0

                x = cx + dir_x * f_along * max_rx + perp_x * across * max_rx
                y = cy + dir_y * f_along * max_ry + perp_y * across * max_ry

                char = BLOCK if abs(c) < steps_across * 0.72 else SOFT
                plot(x, y, char, color_rgb)

        # Peak-hold : petit éclat à la pointe qui retombe lentement.
        peak_length_fraction = inner_fraction + peak * (1.0 - inner_fraction) * 0.85

        if peak_length_fraction > length_fraction + 0.01:
            px = cx + dir_x * peak_length_fraction * max_rx
            py = cy + dir_y * peak_length_fraction * max_ry
            plot(px, py, "•", (255, 255, 255))

    # ─────────────────────────────────────────
    # Sérialisation
    # ─────────────────────────────────────────

    lines = []

    for y in range(height):
        line = ""
        for x in range(width):
            cell = buffer[y][x]
            if cell is None:
                line += " "
            else:
                char, rgb = cell
                line += color(rgb[0], rgb[1], rgb[2], char)
        lines.append(line)

    return lines


# ═════════════════════════════════════════════════════════════
# GEKI GALAXY
# ═════════════════════════════════════════════════════════════
#
# Des bras spiralés d'étoiles qui tournent lentement, densité/
# éclat réactifs au spectre. Pas de BLOCK/SOFT ici : on utilise
# une palette de "points" (· ∘ ○ ✦ ★) dont la densité augmente
# avec l'intensité, pour un rendu texturé plutôt que solide.
# ═════════════════════════════════════════════════════════════

STAR_CHARS = ["·", "∘", "○", "✦", "★"]


def _star_char(intensity):
    """Choisit un caractère d'étoile selon l'intensité (0-1)."""

    index = min(
        len(STAR_CHARS) - 1,
        int(intensity * len(STAR_CHARS))
    )

    return STAR_CHARS[index]


def _galaxy_color(radius_fraction, intensity):
    """Dégradé violet/bleu au centre vers cyan/blanc en périphérie."""

    r = 140 + 60 * (1.0 - radius_fraction)
    g = 90 + 130 * radius_fraction
    b = 200 + 55 * radius_fraction

    brightness = 0.35 + 0.65 * intensity

    return (
        int(min(255, r * brightness)),
        int(min(255, g * brightness)),
        int(min(255, b * brightness))
    )


def make_visualizer_galaxy(width, height, audio, t):

    cx = width / 2.0
    cy = height / 2.0

    buffer = [[None] * width for _ in range(height)]

    def plot(x, y, char, rgb):
        xi = int(round(x))
        yi = int(round(y))
        if 0 <= xi < width and 0 <= yi < height:
            buffer[yi][xi] = (char, rgb)

    max_rx = width * 0.48
    max_ry = height * 0.48

    # ─────────────────────────────────────────
    # Cœur / noyau lumineux (pulse sur les basses)
    # ─────────────────────────────────────────

    core_energy = (audio.bass * 0.6 + audio.mid * 0.25 + audio.treble * 0.15) ** 0.75

    core_fx = 0.04 + core_energy * 0.05
    core_rx = max(1, int(max_rx * core_fx))
    core_ry = max(1, int(max_ry * core_fx))

    for y in range(max(0, int(cy - core_ry) - 1), min(height, int(cy + core_ry) + 2)):
        for x in range(max(0, int(cx - core_rx) - 1), min(width, int(cx + core_rx) + 2)):

            dx = (x - cx) / max(core_rx, 1)
            dy = (y - cy) / max(core_ry, 1)
            dist = math.sqrt(dx * dx + dy * dy)

            if dist <= 1.0:
                boost = 1.0 - dist * 0.3
                r = int(min(255, (225 + audio.bass * 30) * boost))
                g = int(min(255, (210 + audio.mid * 30) * boost))
                b = int(min(255, (255) * boost))
                char = "★" if dist < 0.5 else "✦"
                plot(x, y, char, (r, g, b))

    # ─────────────────────────────────────────
    # Bras spiralés
    # ─────────────────────────────────────────

    arm_count = 3
    turns = 1.5
    stars_per_arm = max(20, int(max(max_rx, max_ry) * 1.1))

    bins_per_arm = max(1, audio.num_bins // arm_count)
    inner_fraction = core_fx * 1.4

    rotation = t * 0.18 + audio.bass * 0.05

    for arm in range(arm_count):

        arm_offset = arm * (2 * math.pi / arm_count)

        for s in range(1, stars_per_arm + 1):

            progress = s / stars_per_arm

            # Densité d'étoiles plus forte près du centre (comme une vraie galaxie).
            radius_fraction = inner_fraction + (progress ** 0.55) * (1.0 - inner_fraction)

            angle = arm_offset + progress * turns * 2 * math.pi + rotation

            bin_index = arm * bins_per_arm + min(
                bins_per_arm - 1,
                int(progress * bins_per_arm)
            )

            level = float(audio.spectrum[bin_index])

            x = cx + math.cos(angle) * radius_fraction * max_rx
            y = cy + math.sin(angle) * radius_fraction * max_ry

            char = _star_char(level)
            color_rgb = _galaxy_color(radius_fraction, level)

            plot(x, y, char, color_rgb)

    # ─────────────────────────────────────────
    # Sérialisation
    # ─────────────────────────────────────────

    lines = []

    for y in range(height):
        line = ""
        for x in range(width):
            cell = buffer[y][x]
            if cell is None:
                line += " "
            else:
                char, rgb = cell
                line += color(rgb[0], rgb[1], rgb[2], char)
        lines.append(line)

    return lines


# ═════════════════════════════════════════════════════════════
# GEKI CLASSIC
# ═════════════════════════════════════════════════════════════
#
# Le vrai visualizer historique : le spectre en barres verticales,
# depuis le bas, sur toute la largeur — avec peak-hold caps qui
# retombent doucement.
# ═════════════════════════════════════════════════════════════

def make_visualizer_classic(width, height, audio, t):

    buffer = [[None] * width for _ in range(height)]

    def plot(x, y, char, rgb):
        if 0 <= x < width and 0 <= y < height:
            buffer[y][x] = (char, rgb)

    # Moins de barres que de bins, mais plus larges — look "EQ" classique.
    bar_count = max(12, min(audio.num_bins, width // 3))
    bins_per_bar = max(1, audio.num_bins // bar_count)

    bar_width = max(1, (width // bar_count) - 1)
    total_span = bar_count * (bar_width + 1)
    start_x = max(0, (width - total_span) // 2)

    baseline = height - 1
    usable_height = max(1, height - 1)

    for i in range(bar_count):

        start_bin = i * bins_per_bar
        end_bin = min(audio.num_bins, start_bin + bins_per_bar)

        level = float(np.mean(audio.spectrum[start_bin:end_bin]))
        peak = float(np.mean(audio.spectrum_peak[start_bin:end_bin]))

        bin_fraction = i / max(1, bar_count - 1)
        color_rgb = _spike_color(bin_fraction, level)

        bar_height = int(level * usable_height)
        bar_height = max(0, min(usable_height, bar_height))

        x0 = start_x + i * (bar_width + 1)

        for row in range(bar_height):

            y = baseline - row

            for dx in range(bar_width):
                plot(x0 + dx, y, BLOCK, color_rgb)

        # Peak-hold : petit trait blanc qui retombe lentement.
        peak_height = int(peak * usable_height)
        peak_height = max(0, min(usable_height, peak_height))

        if peak_height > bar_height:

            y = baseline - peak_height

            for dx in range(bar_width):
                plot(x0 + dx, y, "▔", (255, 255, 255))

    lines = []

    for y in range(height):
        line = ""
        for x in range(width):
            cell = buffer[y][x]
            if cell is None:
                line += " "
            else:
                char, rgb = cell
                line += color(rgb[0], rgb[1], rgb[2], char)
        lines.append(line)

    return lines


# ═════════════════════════════════════════════════════════════
# GEKI DANCE
# ═════════════════════════════════════════════════════════════
#
# Un champ d'astéroïdes dispersés à l'écran (pas radial depuis
# le centre comme les autres thèmes — mécanique différente).
# Chaque roche a sa propre couleur (dégradé arc-en-ciel complet),
# une silhouette irrégulière, et une taille qui pulse fort selon
# sa propre tranche de fréquence. Position de base stable (spirale
# à angle d'or), avec juste un léger flottement organique.
# ═════════════════════════════════════════════════════════════

ASTEROID_COUNT = 12
GOLDEN_ANGLE = math.pi * (3 - math.sqrt(5))


def _asteroid_color(index, count, intensity):

    hue = index / count
    saturation = 0.78
    value = 0.55 + 0.45 * intensity

    r, g, b = colorsys.hsv_to_rgb(hue, saturation, value)

    return (
        int(r * 255),
        int(g * 255),
        int(b * 255)
    )


def make_visualizer_dance(width, height, audio, t):

    cx = width / 2.0
    cy = height / 2.0

    buffer = [[None] * width for _ in range(height)]

    def plot(x, y, char, rgb):
        xi = int(round(x))
        yi = int(round(y))
        if 0 <= xi < width and 0 <= yi < height:
            buffer[yi][xi] = (char, rgb)

    max_rx = width * 0.44
    max_ry = height * 0.44

    bins_per_asteroid = max(1, audio.num_bins // ASTEROID_COUNT)

    for i in range(ASTEROID_COUNT):

        # Position de base stable, dispersée façon spirale à angle
        # d'or (répartition organique mais toujours la même d'une
        # frame à l'autre — pas de scintillement de position).
        scatter_r = math.sqrt((i + 0.5) / ASTEROID_COUNT)
        scatter_angle = i * GOLDEN_ANGLE

        base_x = cx + math.cos(scatter_angle) * scatter_r * max_rx
        base_y = cy + math.sin(scatter_angle) * scatter_r * max_ry

        # Léger flottement organique (pas un vrai déplacement, juste
        # une respiration douce dans l'espace).
        drift_x = math.sin(t * 0.3 + i * 1.7) * width * 0.01
        drift_y = math.cos(t * 0.25 + i * 2.3) * height * 0.01

        ax = base_x + drift_x
        ay = base_y + drift_y

        start_bin = i * bins_per_asteroid
        end_bin = min(audio.num_bins, start_bin + bins_per_asteroid)

        level = float(np.mean(audio.spectrum[start_bin:end_bin]))
        peak = float(np.mean(audio.spectrum_peak[start_bin:end_bin]))

        # La taille pulse fort avec le niveau, et gonfle encore un
        # peu de plus sur un pic frais (peak proche du niveau actuel).
        radius_x_cells = 1.1 + level * 3.4 + max(0.0, peak - level) * 2.0
        radius_y_cells = max(1.0, radius_x_cells * 0.55)

        color_rgb = _asteroid_color(i, ASTEROID_COUNT, level)

        # Silhouette irrégulière ("roche") : rayon modulé par deux
        # harmoniques fixes propres à chaque astéroïde.
        seed = i * 13.37

        box_x = int(radius_x_cells) + 2
        box_y = int(radius_y_cells) + 2

        for dy in range(-box_y, box_y + 1):
            for dx in range(-box_x, box_x + 1):

                nx = dx / max(radius_x_cells, 0.001)
                ny = dy / max(radius_y_cells, 0.001)

                dist = math.sqrt(nx * nx + ny * ny)
                angle = math.atan2(ny, nx)

                wobble = (
                    1.0
                    + 0.18 * math.sin(angle * 3 + seed)
                    + 0.12 * math.sin(angle * 5 - seed * 1.3)
                )

                if dist < wobble:
                    char = BLOCK if dist < wobble * 0.65 else SOFT
                    plot(ax + dx, ay + dy, char, color_rgb)

    # ─────────────────────────────────────────
    # Sérialisation
    # ─────────────────────────────────────────

    lines = []

    for y in range(height):
        line = ""
        for x in range(width):
            cell = buffer[y][x]
            if cell is None:
                line += " "
            else:
                char, rgb = cell
                line += color(rgb[0], rgb[1], rgb[2], char)
        lines.append(line)

    return lines


# ═════════════════════════════════════════════════════════════
# THÈMES
#
# Chaque thème est une fonction (width, height, audio, t) -> lignes.
# Pour en ajouter un nouveau : écrire la fonction, puis l'ajouter
# ici avec son nom (utilisé comme --nom en ligne de commande).
# ═════════════════════════════════════════════════════════════

THEMES = {
    "starburst": make_visualizer_starburst,
    "flower": make_visualizer_flower,
    "galaxy": make_visualizer_galaxy,
    "classic": make_visualizer_classic,
    "dance": make_visualizer_dance,
}

DEFAULT_THEME = "starburst"


# ═════════════════════════════════════════════════════════════
# INTERFACE
# ═════════════════════════════════════════════════════════════

def draw(
    audio,
    t,
    title,
    visualizer_fn
):

    width, height = terminal_size()

    # Espace réservé à l'interface.
    visual_height = max(
        8,
        height - 12
    )

    visual = visualizer_fn(
        width,
        visual_height,
        audio,
        t
    )

    # ─────────────────────────────────────────
    # Temps
    # ─────────────────────────────────────────

    current = audio.position_seconds()
    duration = audio.duration_seconds()

    progress = current / duration if duration else 0

    progress = min(
        max(progress, 0.0),
        1.0
    )

    # ─────────────────────────────────────────
    # Progress bar
    # ─────────────────────────────────────────

    bar_width = min(
        42,
        max(
            10,
            width - 35
        )
    )

    if duration is None:
        progress_bar = f"● LIVE · {format_time(current)}"
    else:
        filled = int(progress * bar_width)
        progress_bar = (
            "━" * filled
            + "●"
            + "━" * max(0, bar_width - filled - 1)
        )

    # ─────────────────────────────────────────
    # UI
    # ─────────────────────────────────────────

    output = []

    output.append(
        color(
            120,
            220,
            255,
            "  GEKI"
        )
        +
        color(
            120,
            120,
            140,
            " // AUDIO VISUALIZER"
        )
    )

    output.append("")

    output.extend(visual)

    output.append("")

    # Track — couleurs pastel assorties au disque plutôt qu'un
    # rose saturé.

    output.append(
        color(
            180,
            180,
            200,
            "  ♪ "
        )
        +
        color(
            255,
            185,
            215,
            title
        )
    )

    # Progression.
    output.append(
        color(
            100,
            100,
            120,
            "  "
        )
        +
        color(
            195,
            175,
            255,
            progress_bar
        )
        +
        " "
        +
        color(
            180,
            185,
            210,
            (
                f"{format_time(current)}"
                if duration is None
                else f"{format_time(current)} / {format_time(duration)}"
            )
        )
    )

    # ─────────────────────────────────────────
    # Draw
    #
    # "\033[K" après chaque ligne efface les résidus d'une
    # frame précédente plus large (resize à chaud).
    # "\033[J" en fin d'écriture efface tout ce qui traînerait
    # en dessous (frame précédente plus haute).
    # ─────────────────────────────────────────

    sys.stdout.write("\033[H")

    sys.stdout.write(
        "\033[K\n".join(output)
        +
        "\033[K\033[J"
    )

    sys.stdout.flush()


# ═════════════════════════════════════════════════════════════
# YT-DLP
# ═════════════════════════════════════════════════════════════

CACHE_DIR = os.path.expanduser("~/.geki/cache")
LIVE_HELPER_APP = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    ".geki-live",
    "GekiAudioTap.app"
)


def ensure_live_helper():
    """Compile/sign the small Core Audio tap app the first time live mode runs."""
    if sys.platform != "darwin":
        raise RuntimeError("La capture d’applications est disponible uniquement sur macOS.")

    source = os.path.join(os.path.dirname(os.path.abspath(__file__)), "geki_audio_tap.m")
    contents = os.path.join(LIVE_HELPER_APP, "Contents")
    executable = os.path.join(contents, "MacOS", "GekiAudioTap")
    info_plist = os.path.join(contents, "Info.plist")

    if not os.path.isfile(source):
        raise RuntimeError(f"Source du module audio introuvable : {source}")

    needs_build = (
        not os.path.isfile(executable)
        or not os.path.isfile(info_plist)
        or os.path.getmtime(source) > os.path.getmtime(executable)
    )
    if not needs_build:
        return executable

    os.makedirs(os.path.dirname(executable), exist_ok=True)
    bundle_info = {
        "CFBundleExecutable": "GekiAudioTap",
        "CFBundleIdentifier": "org.geki.LiveAudioTap",
        "CFBundleName": "GEKI Live Audio",
        "CFBundlePackageType": "APPL",
        "CFBundleShortVersionString": "1.0",
        "LSUIElement": True,
        "NSAudioCaptureUsageDescription": (
            "GEKI capture le son de l’application choisie pour l’analyser et l’afficher en direct."
        ),
    }
    with open(info_plist, "wb") as plist_file:
        plistlib.dump(bundle_info, plist_file)

    compile_command = [
        "clang", "-fobjc-arc", "-O2",
        "-framework", "AppKit",
        "-framework", "CoreAudio",
        "-framework", "Foundation",
        source,
        "-o", executable,
    ]
    try:
        subprocess.run(compile_command, check=True, capture_output=True, text=True)
        subprocess.run(
            ["codesign", "--force", "--deep", "--sign", "-", LIVE_HELPER_APP],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        details = getattr(error, "stderr", None) or str(error)
        raise RuntimeError(f"Impossible de préparer la capture audio native :\n{details}") from error

    return executable


def list_live_processes(helper):
    result = subprocess.run(
        [helper, "--list"],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)


class LiveAudioEngine(AudioEngine):
    """AudioEngine alimenté par un flux Core Audio vivant et borné en mémoire."""

    def __init__(self, process_info, helper):
        self.capture = subprocess.Popen(
            [helper, "--stream", str(process_info["object_id"])],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
        )
        header = bytearray()
        while len(header) < 8:
            part = self.capture.stdout.read(8 - len(header))
            if not part:
                break
            header.extend(part)
        if len(header) != 8 or header[:4] != b"GEKI":
            error_text = self.capture.stderr.read().decode("utf-8", errors="replace").strip()
            self.capture.wait(timeout=2)
            raise RuntimeError(
                error_text or "La capture n’a pas démarré. Vérifie l’autorisation audio de macOS."
            )

        sample_rate = struct.unpack("=I", header[4:])[0]
        if sample_rate <= 0:
            self.capture.terminate()
            raise RuntimeError("Le flux Core Audio a fourni une fréquence d’échantillonnage invalide.")

        super().__init__(sample_rate=sample_rate)
        self.is_live = True
        self.process_name = process_info["name"]
        self._ring = np.zeros(self.fft_size * 8, dtype=np.float32)
        self._write_index = 0
        self._buffered_samples = 0
        self._ring_lock = threading.Lock()
        self._reader = threading.Thread(target=self._read_capture, daemon=True)
        self._reader.start()

    def _read_capture(self):
        pending = b""
        try:
            while True:
                block = self.capture.stdout.read(8192)
                if not block:
                    break
                data = pending + block
                aligned_size = len(data) - (len(data) % 4)
                samples = np.frombuffer(data[:aligned_size], dtype=np.float32)
                pending = data[aligned_size:]
                if samples.size == 0:
                    continue

                with self._ring_lock:
                    first_part = min(samples.size, self._ring.size - self._write_index)
                    self._ring[self._write_index:self._write_index + first_part] = samples[:first_part]
                    remaining = samples.size - first_part
                    if remaining:
                        self._ring[:remaining] = samples[first_part:]
                    self._write_index = (self._write_index + samples.size) % self._ring.size
                    self._buffered_samples = min(self._ring.size, self._buffered_samples + samples.size)
                    self.position += samples.size
        except (OSError, ValueError):
            pass
        finally:
            self.playing = False

    def analysis_chunk(self):
        with self._ring_lock:
            if self._buffered_samples < self.fft_size:
                return np.empty(0, dtype=np.float32)
            start = (self._write_index - self.fft_size) % self._ring.size
            end = self._write_index
            if start < end:
                return self._ring[start:end].copy()
            return np.concatenate((self._ring[start:], self._ring[:end]))

    def duration_seconds(self):
        return None

    def start(self):
        # Le flux natif commence avant la boucle d’affichage.
        pass

    def stop(self):
        if self.capture.poll() is None:
            try:
                self.capture.terminate()
            except ProcessLookupError:
                pass
            try:
                self.capture.wait(timeout=2)
            except subprocess.TimeoutExpired:
                try:
                    self.capture.kill()
                except ProcessLookupError:
                    pass
                self.capture.wait()
        self.playing = False
        self._reader.join(timeout=2)
        if self.capture.stdout:
            self.capture.stdout.close()
        if self.capture.stderr:
            self.capture_error = self.capture.stderr.read().decode("utf-8", errors="replace").strip()
            self.capture.stderr.close()


def live_source_menu():
    """Sélectionne une application qui émet actuellement du son."""
    if not sys.stdin.isatty():
        print("GEKI: le mode live nécessite un terminal interactif.")
        return None

    try:
        helper = ensure_live_helper()
        processes = list_live_processes(helper)
    except (OSError, RuntimeError, subprocess.CalledProcessError, json.JSONDecodeError) as error:
        print(f"GEKI: impossible de lister les applications audio : {error}")
        return None

    selected = 0
    theme_names = list(THEMES)
    selected_theme = theme_names.index(DEFAULT_THEME)
    message = "↑/↓ application   ←/→ style   Entrée démarrer   r actualiser   q quitter"
    fd = sys.stdin.fileno()
    previous = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        while True:
            clear()
            print(color(120, 220, 255, "GEKI // CAPTURE LIVE"))
            print("Applications avec une sortie audio active")
            print(f"Style : {theme_names[selected_theme]}\n")
            if not processes:
                print("Aucune sortie audio active. Lance du son puis appuie sur r.")
            for index, process in enumerate(processes):
                marker = "❯ " if index == selected else "  "
                label = f"{marker}{process['name']}  (PID {process['pid']})"
                print(color(255, 185, 215, label) if index == selected else label)
            print("\n" + message)

            key = os.read(fd, 1).decode("utf-8", errors="ignore")
            if not processes and key not in ("r", "R", "q", "Q", "\x03"):
                continue
            if key == "\x1b":
                if select.select([sys.stdin], [], [], 0.08)[0]:
                    sequence = os.read(fd, 2).decode("utf-8", errors="ignore")
                    if sequence == "[A":
                        selected = (selected - 1) % len(processes)
                    elif sequence == "[B":
                        selected = (selected + 1) % len(processes)
                    elif sequence == "[C":
                        selected_theme = (selected_theme + 1) % len(theme_names)
                    elif sequence == "[D":
                        selected_theme = (selected_theme - 1) % len(theme_names)
                continue
            if key.lower() == "r":
                try:
                    processes = list_live_processes(helper)
                    selected = min(selected, max(0, len(processes) - 1))
                    message = "Liste actualisée.  " + message
                except (OSError, RuntimeError, subprocess.CalledProcessError, json.JSONDecodeError) as error:
                    message = f"Actualisation impossible : {error}"
                if not processes:
                    message = "Aucune sortie audio active. Lance du son puis appuie sur r."
                continue
            if key in ("q", "Q", "\x03"):
                return None
            if key in ("\r", "\n"):
                return processes[selected], theme_names[selected_theme]
    except KeyboardInterrupt:
        return None
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, previous)
        clear()


def metadata_path(audio_path):
    return os.path.splitext(audio_path)[0] + ".json"


def save_track_metadata(audio_path, info):
    """Sauvegarde les informations utiles à la bibliothèque locale."""
    metadata = {
        "title": info.get("title") or os.path.basename(audio_path),
        "id": info.get("id"),
        "url": info.get("webpage_url"),
        "uploader": info.get("uploader") or info.get("channel"),
        "duration": info.get("duration"),
    }
    path = metadata_path(audio_path)
    temporary_path = path + ".tmp"
    with open(temporary_path, "w", encoding="utf-8") as metadata_file:
        json.dump(metadata, metadata_file, ensure_ascii=False, indent=2)
    os.replace(temporary_path, path)


def load_library():
    """Liste les MP3 du cache, avec compatibilité pour les anciens fichiers."""
    if not os.path.isdir(CACHE_DIR):
        return []

    tracks = []
    for name in os.listdir(CACHE_DIR):
        if not name.lower().endswith(".mp3"):
            continue
        path = os.path.join(CACHE_DIR, name)
        if not os.path.isfile(path):
            continue
        title = os.path.splitext(name)[0]
        metadata = {}
        try:
            with open(metadata_path(path), encoding="utf-8") as metadata_file:
                metadata = json.load(metadata_file)
            title = metadata.get("title") or title
        except (OSError, ValueError, TypeError, AttributeError):
            pass
        tracks.append({"path": path, "title": title, "metadata": metadata})
    return sorted(tracks, key=lambda track: track["title"].casefold())


def library_menu():
    """Affiche la bibliothèque et retourne le morceau et le thème choisis."""
    if not sys.stdin.isatty():
        print("GEKI: la bibliothèque interactive nécessite un terminal.")
        return None

    tracks = load_library()
    if not tracks:
        print("GEKI: aucun morceau dans la bibliothèque.")
        print(f"Dossier: {CACHE_DIR}")
        return None

    selected = 0
    theme_names = list(THEMES)
    selected_theme = theme_names.index(DEFAULT_THEME)
    message = "↑/↓ morceau   ←/→ thème   Entrée lire   d supprimer   q quitter"
    fd = sys.stdin.fileno()
    previous = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        while True:
            clear()
            print(color(120, 220, 255, "GEKI // BIBLIOTHÈQUE"))
            print(f"{len(tracks)} morceau(x) — {CACHE_DIR}")
            print(f"Style : {theme_names[selected_theme]}\n")
            for index, track in enumerate(tracks):
                marker = "❯ " if index == selected else "  "
                metadata = track["metadata"]
                detail = metadata.get("uploader") or os.path.basename(track["path"])
                label = f"{marker}{track['title']}  [{detail}]"
                if index == selected:
                    print(color(255, 185, 215, label))
                else:
                    print(label)
            print("\n" + message)

            key = os.read(fd, 1).decode("utf-8", errors="ignore")
            if key == "\x1b":
                if select.select([sys.stdin], [], [], 0.08)[0]:
                    sequence = os.read(fd, 2).decode("utf-8", errors="ignore")
                    if sequence == "[A":
                        selected = (selected - 1) % len(tracks)
                    elif sequence == "[B":
                        selected = (selected + 1) % len(tracks)
                    elif sequence == "[C":
                        selected_theme = (selected_theme + 1) % len(theme_names)
                    elif sequence == "[D":
                        selected_theme = (selected_theme - 1) % len(theme_names)
                continue
            if key in ("q", "Q", "\x03"):
                return None
            if key in ("\r", "\n"):
                return tracks[selected], theme_names[selected_theme]
            if key.lower() == "d":
                track = tracks[selected]
                sys.stdout.write(f"\nSupprimer « {track['title']} » ? (o/N) ")
                sys.stdout.flush()
                answer = os.read(fd, 1).decode("utf-8", errors="ignore")
                if answer.lower() in ("o", "y"):
                    os.remove(track["path"])
                    try:
                        os.remove(metadata_path(track["path"]))
                    except FileNotFoundError:
                        pass
                    tracks.pop(selected)
                    if not tracks:
                        return None
                    selected = min(selected, len(tracks) - 1)
                    message = "Morceau supprimé."
                else:
                    message = "Suppression annulée."
    except KeyboardInterrupt:
        return None
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, previous)
        clear()


def is_url(source):
    """Détecte si l'argument fourni est un lien plutôt qu'un fichier local."""
    return source.startswith("http://") or source.startswith("https://")


def download_audio(url):
    """Télécharge l'audio d'un lien via yt-dlp, avec cache local.

    Retourne (chemin_fichier, titre_affichable)."""

    try:
        import yt_dlp
    except ImportError:

        print("GEKI: yt-dlp n'est pas installé.")
        print()
        print("Installe-le avec :")
        print("  pip3 install yt-dlp")
        print()
        print("Et si ce n'est pas déjà fait, ffmpeg (requis pour")
        print("extraire l'audio) :")
        print("  brew install ffmpeg")

        sys.exit(1)

    os.makedirs(CACHE_DIR, exist_ok=True)

    # Récupère les infos sans télécharger, pour connaître
    # l'id (clé de cache) et le titre avant de lancer quoi que ce soit.
    probe_opts = {
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
    }

    try:
        with yt_dlp.YoutubeDL(probe_opts) as ydl:
            info = ydl.extract_info(url, download=False)
    except Exception as error:

        print("GEKI: impossible de lire ce lien.")
        print()
        print(f"Erreur: {error}")

        sys.exit(1)

    video_id = info.get("id", "audio")
    title = info.get("title", video_id)

    cached_path = os.path.join(
        CACHE_DIR,
        f"{video_id}.mp3"
    )

    if os.path.exists(cached_path):
        save_track_metadata(cached_path, info)
        print(f"GEKI: '{title}' déjà en cache, lecture directe.")
        return cached_path, title

    print(f"GEKI: téléchargement de '{title}'...")

    def progress_hook(d):

        if d["status"] == "downloading":

            pct = d.get("_percent_str", "").strip()
            sys.stdout.write(f"\r  {pct}   ")
            sys.stdout.flush()

        elif d["status"] == "finished":

            sys.stdout.write("\r  conversion audio...        \n")
            sys.stdout.flush()

    download_opts = {
        "format": "bestaudio/best",
        "outtmpl": os.path.join(CACHE_DIR, "%(id)s.%(ext)s"),
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "mp3",
            "preferredquality": "192",
        }],
        "progress_hooks": [progress_hook],
        "quiet": True,
        "no_warnings": True,
    }

    try:
        with yt_dlp.YoutubeDL(download_opts) as ydl:
            ydl.download([url])
    except Exception as error:

        print()
        print("GEKI: échec du téléchargement.")
        print()
        print(f"Erreur: {error}")
        print()
        print("Vérifie que ffmpeg est installé (brew install ffmpeg).")

        sys.exit(1)

    print()

    if os.path.exists(cached_path):
        save_track_metadata(cached_path, info)

    return cached_path, title


def build_queue(source):
    """Retourne une liste de sources à lire à la suite.

    Pour un fichier local ou un lien vers une seule vidéo : liste
    à un seul élément. Pour un lien de playlist : liste de liens,
    un par morceau, dans l'ordre de la playlist."""

    if not is_url(source):
        return [source]

    try:
        import yt_dlp
    except ImportError:

        print("GEKI: yt-dlp n'est pas installé.")
        print()
        print("Installe-le avec :")
        print("  pip3 install yt-dlp")

        sys.exit(1)

    # "extract_flat" : scan léger et rapide (juste id/titre/url par
    # entrée), sans aller chercher tous les formats de chaque vidéo
    # un par un — sinon une grosse playlist mettrait un temps fou
    # juste à être détectée.
    probe_opts = {
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
        "extract_flat": "in_playlist",
    }

    try:
        with yt_dlp.YoutubeDL(probe_opts) as ydl:
            info = ydl.extract_info(source, download=False)
    except Exception as error:

        print("GEKI: impossible de lire ce lien.")
        print()
        print(f"Erreur: {error}")

        sys.exit(1)

    is_playlist = (
        info.get("_type") == "playlist"
        or "entries" in info
    )

    if not is_playlist:
        return [source]

    entries = [e for e in info.get("entries", []) if e]

    if not entries:
        print("GEKI: playlist vide ou inaccessible.")
        sys.exit(1)

    # Lien de "Mix" YouTube (watch?v=...&list=RD...&index=N) : le
    # paramètre index indique le morceau sur lequel on a cliqué.
    # Sans ça, on repartirait toujours du tout début du mix.
    query = parse_qs(urlparse(source).query)

    if "index" in query:

        try:
            start_index = max(1, int(query["index"][0]))
        except ValueError:
            start_index = 1

        entries = entries[start_index - 1:]

        if not entries:
            print("GEKI: index hors limites pour ce mix, lecture depuis le début.")
            entries = [e for e in info.get("entries", []) if e]

    playlist_title = info.get("title", "playlist")

    print(f"GEKI: playlist '{playlist_title}' — {len(entries)} morceau(x).")

    queue = []

    for entry in entries:

        entry_url = (
            entry.get("url")
            or entry.get("webpage_url")
            or entry.get("id")
        )

        if not entry_url:
            continue

        if entry_url.startswith("http"):
            queue.append(entry_url)
        else:
            # extract_flat renvoie parfois juste l'id de la vidéo.
            queue.append(f"https://www.youtube.com/watch?v={entry_url}")

    return queue


def resolve_source(source):
    """Retourne (chemin_fichier, titre_affichable) pour un fichier
    local ou un lien (téléchargé via yt-dlp au besoin)."""

    if is_url(source):
        return download_audio(source)

    return source, os.path.basename(source)


# ═════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════

def main():

    if len(sys.argv) < 2:

        print(
            "GEKI // Terminal Audio Visualizer"
        )

        print()
        print("Usage:")
        print("  geki -library")
        print("  geki -live [--theme]")
        print("  python3 geki.py <audio-file-or-url-or-playlist> [--theme]")
        print()
        print("Example:")
        print('  python3 geki.py "music.mp3"')
        print('  python3 geki.py "music.mp3" --flower')
        print('  python3 geki.py "https://youtube.com/watch?v=..."')
        print('  python3 geki.py "https://youtube.com/playlist?list=..."')
        print()
        print(f"Thèmes disponibles : {', '.join(THEMES)}")

        sys.exit(1)

    args = sys.argv[1:]

    live_selection = None
    library_title = None
    if args and args[0] == "-live":
        if len(args) > 2:
            print("GEKI: usage: geki -live [--theme]")
            sys.exit(1)
        live_selection = live_source_menu()
        if live_selection is None:
            return
        source = None
        theme_name = live_selection[1]
        args = args[1:]
    elif args and args[0] == "-library":
        if len(args) > 2:
            print("GEKI: usage: geki -library [--theme]")
            sys.exit(1)
        library_selection = library_menu()
        if library_selection is None:
            return
        selection, library_theme = library_selection
        source = selection["path"]
        library_title = selection["title"]
        theme_name = library_theme
        args = args[1:]
    else:
        source = None
        theme_name = DEFAULT_THEME

    flags = [a for a in args if a.startswith("--")]
    positional = [a for a in args if not a.startswith("--")]

    if flags:

        requested = flags[0][2:]

        if requested not in THEMES:

            print(f"GEKI: thème inconnu '{requested}'")
            print(f"Thèmes disponibles : {', '.join(THEMES)}")

            sys.exit(1)

        theme_name = requested

    if source is None and not positional and live_selection is None:

        print("GEKI: aucun fichier ou lien fourni.")

        sys.exit(1)

    visualizer_fn = THEMES[theme_name]

    if live_selection is not None:
        process_info, _ = live_selection
        audio = None
        error_message = None
        try:
            print("GEKI: démarrage de la capture. Accepte la demande macOS si elle apparaît.")
            audio = LiveAudioEngine(process_info, ensure_live_helper())
            audio.start()
            hide_cursor()
            start = time.perf_counter()
            while audio.playing:
                now = time.perf_counter()
                audio.analyze()
                draw(audio, now - start, audio.process_name, visualizer_fn)
                time.sleep(1 / FPS)
        except KeyboardInterrupt:
            pass
        except Exception as error:
            error_message = str(error)
        finally:
            if audio is not None:
                audio.stop()
                if audio.capture.returncode and not error_message:
                    error_message = audio.capture_error or (
                        f"Le helper audio s’est arrêté (code {audio.capture.returncode})."
                    )
            show_cursor()
            clear()
        if error_message:
            print(f"GEKI: la capture live a échoué.\n{error_message}")
        return

    if source is None:
        source = positional[0]

    queue = build_queue(source)

    if not queue:

        print("GEKI: rien à lire.")

        sys.exit(1)

    clear()
    hide_cursor()

    try:

        for index, item in enumerate(queue):

            filename, title = resolve_source(item)

            if library_title is not None:
                title = library_title

            if not os.path.exists(filename):

                print(
                    f"GEKI: fichier introuvable, morceau ignoré: {filename}"
                )

                continue

            try:

                audio = AudioEngine(
                    filename
                )

            except Exception as error:

                print(
                    "GEKI: impossible de charger "
                    "ce morceau, il est ignoré."
                )

                print()
                print(f"Erreur: {error}")

                continue

            display_title = title

            if len(queue) > 1:
                display_title = f"({index + 1}/{len(queue)}) {title}"

            audio.start()

            start = time.perf_counter()

            while audio.playing:

                now = time.perf_counter()

                audio.analyze()

                draw(
                    audio,
                    now - start,
                    display_title,
                    visualizer_fn
                )

                time.sleep(
                    1 / FPS
                )

            audio.stop()

    except KeyboardInterrupt:

        pass

    finally:

        show_cursor()
        clear()


if __name__ == "__main__":
    main()
