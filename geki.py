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
    """Clear the terminal screen."""
    sys.stdout.write("\033[2J\033[H")
    sys.stdout.flush()


def hide_cursor():
    """Hide the terminal cursor."""
    sys.stdout.write("\033[?25l")
    sys.stdout.flush()


def show_cursor():
    """Show the terminal cursor."""
    sys.stdout.write("\033[?25h")
    sys.stdout.flush()


def terminal_size():
    """Return the current terminal dimensions."""
    return shutil.get_terminal_size((100, 30))


def color(r, g, b, text):
    """Wrap text in an ANSI TrueColor sequence."""
    return f"\033[38;2;{r};{g};{b}m{text}\033[0m"


# ═════════════════════════════════════════════════════════════
# AUDIO ENGINE
# ═════════════════════════════════════════════════════════════

class AudioEngine:

    def __init__(self, filename=None, sample_rate=None):

        # ─────────────────────────────────────────
        # Load the audio file.
        # ─────────────────────────────────────────

        if filename is not None:
            self.data, self.sample_rate = sf.read(
                filename,
                dtype="float32",
                always_2d=True
            )

            # Convert to mono for FFT analysis.
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
        # Audio values.
        # ─────────────────────────────────────────

        self.bass = 0.0
        self.mid = 0.0
        self.treble = 0.0
        self.energy = 0.0

        # Global sensitivity.
        self.sensitivity = 1.15

        # Adaptive per-band ceiling (AGC).
        # Tracks the strongest recent peak and decays slowly;
        # see adaptive_normalize() in analyze().
        self.bass_ceiling = -40.0
        self.mid_ceiling = -40.0
        self.treble_ceiling = -40.0

        # ─────────────────────────────────────────
        # Multi-band spectrum (used by starburst).
        #
        # 48 logarithmically spaced bands from 30 Hz to 16 kHz.
        # This provides more detail than bass/mid/treble, with
        # per-bin adaptive normalization and a peak hold effect.
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

                # This band is too narrow to contain an FFT bin. This
                # can happen in the bass range when logarithmic spacing
                # is finer than the FFT resolution. Attach the nearest
                # FFT bin to its center so the band can still respond.

                center = (lo + hi) / 2.0
                nearest_index = int(np.argmin(np.abs(freqs_full - center)))

                mask = np.zeros_like(freqs_full, dtype=bool)
                mask[nearest_index] = True

            self.bin_masks.append(mask)

        self.spectrum = np.zeros(self.num_bins)
        self.spectrum_ceiling = np.full(self.num_bins, -40.0)
        self.spectrum_peak = np.zeros(self.num_bins)

        # State used for kick detection.
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

        # End of the track.
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

        # Get the latest audio window.
        chunk = self.analysis_chunk()

        if len(chunk) < self.fft_size:
            return

        # Apply a Hann window.
        window = np.hanning(
            len(chunk)
        )

        signal = chunk * window
        n = len(signal)

        # ─────────────────────────────────────────
        # Properly normalized FFT.
        # ─────────────────────────────────────────

        spectrum = np.fft.rfft(
            signal
        )

        magnitude = (
            np.abs(spectrum)
            / n
        )

        # Approximate compensation for gain lost through
        # the Hann window.
        magnitude *= 2.0

        frequencies = np.fft.rfftfreq(
            n,
            1 / self.sample_rate
        )

        # ─────────────────────────────────────────
        # Frequency bands.
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
        # RMS measurement per band.
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
        # Convert to decibels.
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
        # Adaptive normalization (per-band AGC).
        #
        # The ceiling tracks the strongest recent level in each band
        # and slowly falls when that level is no longer reached. This
        # lets the visualizer adapt to each track's dynamics instead
        # of constantly saturating the bass with a fixed threshold.
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
        # Sensitivity curve.
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
        # Fast attack, slower release.
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

        # Overall energy.
        self.energy = (
            self.bass * 0.55
            +
            self.mid * 0.30
            +
            self.treble * 0.15
        )

        # ─────────────────────────────────────────
        # Multi-band spectrum (starburst).
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

        # Add contrast to make sharp peaks stand out.
        spectrum_norm = spectrum_norm ** 0.8

        # Percussive smoothing: near-instant attack and a quick
        # release, without the floating effect of the blob.
        spec_attack = 0.75
        spec_release_smooth = 0.30

        rising_smooth = spectrum_norm > self.spectrum

        self.spectrum = np.where(
            rising_smooth,
            self.spectrum * (1 - spec_attack) + spectrum_norm * spec_attack,
            self.spectrum * (1 - spec_release_smooth) + spectrum_norm * spec_release_smooth
        )

        # Peak hold: a small marker rises with the peak and falls
        # slowly, like a VU meter.
        peak_fall = 0.025

        self.spectrum_peak = np.maximum(
            self.spectrum,
            self.spectrum_peak - peak_fall
        )

        # ─────────────────────────────────────────
        # Kick detection (rising edge of the raw bass level).
        #
        # Use the local `bass` value before attack/release smoothing
        # updates self.bass. Its slower release may not fall below the
        # threshold between close kicks, preventing a new rising edge.
        # The raw value follows frame-by-frame energy and falls between
        # beats, so it can reliably detect each kick.
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

        # Remove rings whose visual lifetime has expired.
        self.shockwaves = [
            spawn_time for spawn_time in self.shockwaves
            if now - spawn_time < 1.2
        ]

    # ═════════════════════════════════════════════
    # TIME
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
# A central core pulses with the bass, surrounded by radial spikes
# (one per frequency bin) that shoot outward. A near-instant attack
# and peak hold create sharp hits instead of a smooth, organic look.
# ═════════════════════════════════════════════════════════════

def _spike_color(bin_fraction, intensity):
    """Return a spike color based on frequency position (0 = bass, 1 = treble)."""

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
    # Central core (pulses with the bass).
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
    # Radial spikes (one per spectrum bin).
    # ─────────────────────────────────────────

    num_bins = audio.num_bins
    inner_fraction = core_fx * 1.35

    # Continuous rotation; adjust its speed and direction here.
    # Strong bass hits add a small speed boost so the rotation
    # accelerates with the track's energy.
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

        # Peak hold: a small spark lingers, then falls.
        peak_fraction = inner_fraction + peak * (1.0 - inner_fraction) * 0.9

        if peak_fraction > tip_fraction + 0.01:
            px = cx + cos_a * peak_fraction * max_rx
            py = cy + sin_a * peak_fraction * max_ry
            plot(px, py, "•", (255, 255, 255))

    # ─────────────────────────────────────────
    # Serialize the frame.
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
# Rounded petals grow from a warm central core. Each petal responds
# to a group of spectrum bins and uses the starburst peak hold to
# preserve its sharp-hit character.
# ═════════════════════════════════════════════════════════════

def _petal_color(petal_fraction, intensity):
    """Return a petal color based on its position in the pink-purple-magenta palette."""

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
    # Warm central core (pulses with the bass).
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
    # Petals (groups of spectrum bins).
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

            # Petal shape: narrow at the base and tip, wide in the middle.
            taper = math.sin(math.pi * progress)
            half_width = max_half_width * taper

            steps_across = max(1, int(half_width * max(max_rx, max_ry) * 2))

            for c in range(-steps_across, steps_across + 1):

                across = (c / steps_across) * half_width if steps_across else 0.0

                x = cx + dir_x * f_along * max_rx + perp_x * across * max_rx
                y = cy + dir_y * f_along * max_ry + perp_y * across * max_ry

                char = BLOCK if abs(c) < steps_across * 0.72 else SOFT
                plot(x, y, char, color_rgb)

        # Peak hold: a small spark at the tip falls slowly.
        peak_length_fraction = inner_fraction + peak * (1.0 - inner_fraction) * 0.85

        if peak_length_fraction > length_fraction + 0.01:
            px = cx + dir_x * peak_length_fraction * max_rx
            py = cy + dir_y * peak_length_fraction * max_ry
            plot(px, py, "•", (255, 255, 255))

    # ─────────────────────────────────────────
    # Serialize the frame.
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
# Slowly rotating spiral arms of stars react to the spectrum through
# their density and brightness. Instead of BLOCK/SOFT, this theme uses
# dots (· ∘ ○ ✦ ★) that grow denser with intensity for a textured look.
# ═════════════════════════════════════════════════════════════

STAR_CHARS = ["·", "∘", "○", "✦", "★"]


def _star_char(intensity):
    """Choose a star character based on intensity (0-1)."""

    index = min(
        len(STAR_CHARS) - 1,
        int(intensity * len(STAR_CHARS))
    )

    return STAR_CHARS[index]


def _galaxy_color(radius_fraction, intensity):
    """Create a purple-blue gradient at the center fading to cyan-white at the edges."""

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
    # Bright core (pulses with the bass).
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
    # Spiral arms.
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

            # Increase star density near the center, like a real galaxy.
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
    # Serialize the frame.
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
# GEKI PRISM
# ═════════════════════════════════════════════════════════════
#
# A rotating, audio-shaped icosahedron floats over a mirrored
# kaleidoscope. Bass hits trigger a brief, restrained color wash.
# ═════════════════════════════════════════════════════════════

PRISM_PHI = (1 + math.sqrt(5)) / 2
PRISM_VERTICES = [
    (-1, PRISM_PHI, 0), (1, PRISM_PHI, 0), (-1, -PRISM_PHI, 0), (1, -PRISM_PHI, 0),
    (0, -1, PRISM_PHI), (0, 1, PRISM_PHI), (0, -1, -PRISM_PHI), (0, 1, -PRISM_PHI),
    (PRISM_PHI, 0, -1), (PRISM_PHI, 0, 1), (-PRISM_PHI, 0, -1), (-PRISM_PHI, 0, 1),
]
PRISM_FACES = [
    (0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11),
    (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6), (7, 1, 8),
    (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9),
    (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1),
]
PRISM_FLASH_COLORS = [
    (255, 42, 55),
    (245, 248, 255),
    (42, 105, 255),
    (48, 255, 135),
]
PRISM_GLYPHS = ["·", "✦", "◇", "✧", "•"]


def _prism_face_color(index, level):
    hue = (0.52 + index * 0.137) % 1.0
    red, green, blue = colorsys.hsv_to_rgb(hue, 0.78, 0.42 + 0.58 * level)
    return int(red * 255), int(green * 255), int(blue * 255)


def make_visualizer_prism(width, height, audio, t):
    """Render a reactive polyhedron inside a mirrored color field."""
    now = time.perf_counter()
    seen_kick = getattr(audio, "prism_seen_kick", 0.0)
    new_kicks = [stamp for stamp in audio.shockwaves if stamp > seen_kick]
    if new_kicks:
        latest_kick = max(new_kicks)
        audio.prism_seen_kick = latest_kick
        previous_flash = getattr(audio, "prism_flash_start", 0.0)
        if latest_kick - previous_flash >= 0.42:
            audio.prism_flash_start = latest_kick
            audio.prism_flash_index = (getattr(audio, "prism_flash_index", -1) + 1) % len(PRISM_FLASH_COLORS)

    flash_age = now - getattr(audio, "prism_flash_start", 0.0)
    flash_strength = max(0.0, 1.0 - flash_age / 0.52) ** 2
    flash_strength = min(1.0, flash_strength)
    flash_color = PRISM_FLASH_COLORS[getattr(audio, "prism_flash_index", 0)]
    flash_mix = 0.23 * flash_strength
    background_rgb = tuple(
        int(base * (1 - flash_mix) + flash * flash_mix)
        for base, flash in zip((4, 5, 12), flash_color)
    )

    cx = width / 2.0
    cy = height / 2.0
    max_rx = width * 0.31
    max_ry = height * 0.42
    buffer = [[(" ", None) for _ in range(width)] for _ in range(height)]

    def plot(x, y, char, rgb):
        xi = int(round(x))
        yi = int(round(y))
        if 0 <= xi < width and 0 <= yi < height:
            buffer[yi][xi] = (char, rgb)

    # Fill the field with a mirrored, frequency-colored pattern.
    wedge = (2 * math.pi) / 10
    palette = [
        (75, 35, 150), (32, 72, 190), (20, 145, 185), (35, 155, 105),
        (155, 85, 35), (190, 40, 95), (115, 35, 170), (35, 110, 160),
    ]
    for y in range(height):
        for x in range(width):
            dx = (x - cx) / max(max_rx, 1)
            dy = (y - cy) / max(max_ry, 1)
            radius = math.sqrt(dx * dx + dy * dy)
            angle = math.atan2(dy, dx) - t * 0.12
            folded = abs((angle + wedge / 2) % wedge - wedge / 2)
            ripple = math.sin(radius * 19 - folded * 15 + t * 0.8)
            facet = math.cos(radius * 8 + folded * 25 - t * 0.25)
            strength = max(0.0, ripple * 0.55 + facet * 0.32 - 0.24)
            band = int((folded / wedge * 7 + radius * 3) * 2) % audio.num_bins
            level = float(audio.spectrum[band])
            if strength > 0.06:
                base = palette[(int(folded / wedge * len(palette)) + int(radius * 3)) % len(palette)]
                brightness = 0.28 + 0.52 * strength + 0.2 * level
                rgb = tuple(min(255, int(channel * brightness + level * 36)) for channel in base)
                glyph = PRISM_GLYPHS[min(len(PRISM_GLYPHS) - 1, int(strength * len(PRISM_GLYPHS)))]
                buffer[y][x] = (glyph, rgb)

    # Rotate the vertices and let nearby frequency bands deform them.
    rotation_x = t * 0.38 + audio.treble * 0.06
    rotation_y = t * 0.57 + audio.mid * 0.08
    rotation_z = t * 0.21 + audio.bass * 0.05
    scale = 1.0 + audio.bass * 0.07
    projected = []
    for index, vertex in enumerate(PRISM_VERTICES):
        vx, vy, vz = vertex
        norm = math.sqrt(vx * vx + vy * vy + vz * vz)
        bin_index = int(index * audio.num_bins / len(PRISM_VERTICES))
        deformation = 1.0 + float(audio.spectrum[bin_index]) * 0.17
        vx, vy, vz = (component / norm * deformation for component in (vx, vy, vz))

        y1 = vy * math.cos(rotation_x) - vz * math.sin(rotation_x)
        z1 = vy * math.sin(rotation_x) + vz * math.cos(rotation_x)
        x2 = vx * math.cos(rotation_y) + z1 * math.sin(rotation_y)
        z2 = -vx * math.sin(rotation_y) + z1 * math.cos(rotation_y)
        x3 = x2 * math.cos(rotation_z) - y1 * math.sin(rotation_z)
        y3 = x2 * math.sin(rotation_z) + y1 * math.cos(rotation_z)
        perspective = 2.75 / (3.4 - z2 * 0.36)
        px = cx + x3 * perspective * scale * max_rx
        py = cy + y3 * perspective * scale * max_ry
        projected.append((px, py, z2))

    face_order = sorted(
        enumerate(PRISM_FACES),
        key=lambda item: sum(projected[index][2] for index in item[1]) / 3,
    )
    face_glyphs = ["░", "▒", "▓", "█"]
    for face_index, face in face_order:
        points = [projected[index] for index in face]
        a, b, c = points
        min_x = max(0, int(math.floor(min(a[0], b[0], c[0]))))
        max_x = min(width - 1, int(math.ceil(max(a[0], b[0], c[0]))))
        min_y = max(0, int(math.floor(min(a[1], b[1], c[1]))))
        max_y = min(height - 1, int(math.ceil(max(a[1], b[1], c[1]))))
        denominator = (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1])
        if abs(denominator) < 0.0001:
            continue

        band_start = int(face_index * audio.num_bins / len(PRISM_FACES))
        level = float(np.mean(audio.spectrum[band_start:band_start + 2]))
        face_color = _prism_face_color(face_index, level)
        glyph = face_glyphs[min(len(face_glyphs) - 1, int(level * len(face_glyphs)))]
        for y in range(min_y, max_y + 1):
            for x in range(min_x, max_x + 1):
                w1 = ((b[1] - c[1]) * (x - c[0]) + (c[0] - b[0]) * (y - c[1])) / denominator
                w2 = ((c[1] - a[1]) * (x - c[0]) + (a[0] - c[0]) * (y - c[1])) / denominator
                w3 = 1.0 - w1 - w2
                if w1 >= 0 and w2 >= 0 and w3 >= 0:
                    plot(x, y, glyph, face_color)

    # Draw bright edges and audio-reactive vertex sparks over the facets.
    for face_index, face in face_order:
        edge_color = _prism_face_color(face_index, 1.0)
        for start_index, end_index in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])):
            start = projected[start_index]
            end = projected[end_index]
            steps = max(1, int(max(abs(end[0] - start[0]), abs(end[1] - start[1]))))
            delta_x = end[0] - start[0]
            delta_y = end[1] - start[1]
            if abs(delta_x) > abs(delta_y) * 1.7:
                edge_glyph = "─"
            elif abs(delta_y) > abs(delta_x) * 1.7:
                edge_glyph = "│"
            else:
                edge_glyph = "╱" if delta_x * delta_y < 0 else "╲"
            for step in range(steps + 1):
                ratio = step / steps
                plot(start[0] + (end[0] - start[0]) * ratio,
                     start[1] + (end[1] - start[1]) * ratio,
                     edge_glyph, edge_color)

    for index, (x, y, _depth) in enumerate(projected):
        band_index = int(index * audio.num_bins / len(PRISM_VERTICES))
        level = float(audio.spectrum[band_index])
        if level > 0.22:
            plot(x, y, "✦" if level > 0.65 else "•", (225, 245, 255))

    # Run-length encode foreground colors while keeping the background wash.
    lines = []
    for row in buffer:
        parts = [f"\033[48;2;{background_rgb[0]};{background_rgb[1]};{background_rgb[2]}m"]
        active_color = None
        for char, rgb in row:
            if rgb != active_color:
                if rgb is None:
                    parts.append("\033[39m")
                else:
                    parts.append(f"\033[38;2;{rgb[0]};{rgb[1]};{rgb[2]}m")
                active_color = rgb
            parts.append(char)
        parts.append("\033[0m")
        lines.append("".join(parts))
    return lines


# ═════════════════════════════════════════════════════════════
# GEKI CLASSIC
# ═════════════════════════════════════════════════════════════
#
# The original visualizer: a full-width spectrum of vertical bars
# rising from the bottom, with peak-hold caps that fall slowly.
# ═════════════════════════════════════════════════════════════

def make_visualizer_classic(width, height, audio, t):

    buffer = [[None] * width for _ in range(height)]

    def plot(x, y, char, rgb):
        if 0 <= x < width and 0 <= y < height:
            buffer[y][x] = (char, rgb)

    # Fewer, wider bars than spectrum bins for a classic EQ look.
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

        # Peak hold: a small white marker that falls slowly.
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
# A field of asteroids scattered across the screen, rather than
# radiating from the center like the other themes. Each rock has its
# own rainbow color, irregular silhouette, and size that pulses with
# its frequency band. Positions stay fixed on a golden-angle spiral,
# with a subtle organic drift.
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

        # Stable base positions follow a golden-angle spiral. This
        # creates an organic distribution without frame-to-frame jitter.
        scatter_r = math.sqrt((i + 0.5) / ASTEROID_COUNT)
        scatter_angle = i * GOLDEN_ANGLE

        base_x = cx + math.cos(scatter_angle) * scatter_r * max_rx
        base_y = cy + math.sin(scatter_angle) * scatter_r * max_ry

        # Add a subtle organic drift: a gentle breathing effect rather
        # than actual movement through space.
        drift_x = math.sin(t * 0.3 + i * 1.7) * width * 0.01
        drift_y = math.cos(t * 0.25 + i * 2.3) * height * 0.01

        ax = base_x + drift_x
        ay = base_y + drift_y

        start_bin = i * bins_per_asteroid
        end_bin = min(audio.num_bins, start_bin + bins_per_asteroid)

        level = float(np.mean(audio.spectrum[start_bin:end_bin]))
        peak = float(np.mean(audio.spectrum_peak[start_bin:end_bin]))

        # Size pulses with the level and grows a little more on a fresh
        # peak (when the peak is close to the current level).
        radius_x_cells = 1.1 + level * 3.4 + max(0.0, peak - level) * 2.0
        radius_y_cells = max(1.0, radius_x_cells * 0.55)

        color_rgb = _asteroid_color(i, ASTEROID_COUNT, level)

        # Create an irregular rock silhouette by modulating its radius
        # with two fixed harmonics unique to each asteroid.
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
    # Serialize the frame.
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
# THEMES
#
# Each theme is a function with signature (width, height, audio, t) -> lines.
# To add a theme, write the function and register it here with the name
# used by the corresponding --name command-line option.
# ═════════════════════════════════════════════════════════════

THEMES = {
    "starburst": make_visualizer_starburst,
    "flower": make_visualizer_flower,
    "galaxy": make_visualizer_galaxy,
    "prism": make_visualizer_prism,
    "classic": make_visualizer_classic,
    "dance": make_visualizer_dance,
}

DEFAULT_THEME = "starburst"


# ═════════════════════════════════════════════════════════════
# INTERFACE
# ═════════════════════════════════════════════════════════════

def _clip_text(text, limit):
    """Clip plain text to a terminal column limit."""
    text = str(text)
    if limit <= 0:
        return ""
    if len(text) > limit:
        return text[:max(0, limit - 1)] + "…"
    return text


def _frame_row(left, right, width):
    """Build a fixed-width row for the playback frame."""
    inner_width = max(0, width - 2)
    left = _clip_text(left, inner_width)
    right = _clip_text(right, max(0, inner_width - len(left)))
    gap = max(0, inner_width - len(left) - len(right))
    return (
        color(66, 82, 105, "│")
        + color(120, 220, 255, left)
        + " " * gap
        + color(185, 195, 215, right)
        + color(66, 82, 105, "│")
    )


def _frame_item(text, width, selected=False):
    """Build a selectable row for terminal menus."""
    inner_width = max(0, width - 4)
    text = _clip_text(text, inner_width)
    fill = " " * max(0, inner_width - len(text))
    item_color = (255, 185, 215) if selected else (175, 185, 205)
    return (
        color(66, 82, 105, "│ ")
        + color(*item_color, text)
        + fill
        + color(66, 82, 105, " │")
    )


def draw(audio, t, title, visualizer_fn):
    width, height = terminal_size()
    width = max(20, width)
    visual_height = max(4, height - 11)
    art_width = width - 2

    current = audio.position_seconds()
    duration = audio.duration_seconds()
    progress = current / duration if duration else 0
    progress = min(max(progress, 0.0), 1.0)
    mode = "LIVE CAPTURE" if duration is None else "FILE PLAYBACK"
    theme_name = visualizer_fn.__name__.replace("make_visualizer_", "").upper()

    top = color(66, 82, 105, "╭" + "─" * (width - 2) + "╮")
    divider = color(66, 82, 105, "├" + "─" * (width - 2) + "┤")
    bottom = color(66, 82, 105, "╰" + "─" * (width - 2) + "╯")

    visual = visualizer_fn(art_width, visual_height, audio, t)
    output = [
        top,
        _frame_row(" GEKI  /  TERMINAL AUDIO VISUALIZER", f"{mode}  ·  {theme_name} ", width),
        _frame_row("", "", width),
    ]
    output.extend(color(66, 82, 105, "│") + line + color(66, 82, 105, "│") for line in visual)
    output.extend([
        divider,
        _frame_row(" NOW PLAYING", "", width),
        _frame_row(f"  ♪  {_clip_text(title, width - 9)}", "", width),
    ])

    content_width = max(8, width - 6)
    if duration is None:
        progress_text = f"● LIVE   {format_time(current)}"
    else:
        time_text = f"{format_time(current)} / {format_time(duration)}"
        bar_width = max(4, content_width - len(time_text) - 3)
        filled = min(bar_width - 1, int(progress * bar_width))
        progress_bar = "━" * filled + "●" + "━" * max(0, bar_width - filled - 1)
        progress_text = f"{progress_bar}  {time_text}"
    output.append(_frame_row(f"  {progress_text}", "", width))
    output.extend([
        _frame_row("  CTRL+C  STOP PLAYBACK", "", width),
        bottom,
    ])

    # Clear each rendered line to remove leftovers after a terminal resize.
    sys.stdout.write("\033[H")
    sys.stdout.write("\033[K\n".join(output) + "\033[K\033[J")
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
        raise RuntimeError(f"Could not prepare native audio capture:\n{details}") from error

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
    """AudioEngine backed by a live Core Audio stream with a bounded buffer."""

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
                error_text or "Capture did not start. Check macOS audio permissions."
            )

        sample_rate = struct.unpack("=I", header[4:])[0]
        if sample_rate <= 0:
            self.capture.terminate()
            raise RuntimeError("The Core Audio stream returned an invalid sample rate.")

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
        # Start the native stream before entering the display loop.
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
    """Select an application that is currently playing audio."""
    if not sys.stdin.isatty():
        print("GEKI: live mode requires an interactive terminal.")
        return None

    try:
        helper = ensure_live_helper()
        processes = list_live_processes(helper)
    except (OSError, RuntimeError, subprocess.CalledProcessError, json.JSONDecodeError) as error:
        print(f"GEKI: could not list audio applications: {error}")
        return None

    selected = 0
    theme_names = list(THEMES)
    selected_theme = theme_names.index(DEFAULT_THEME)
    message = "↑/↓ select app   ←/→ select theme   Enter start   r refresh   q quit"
    fd = sys.stdin.fileno()
    previous = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        while True:
            clear()
            width = max(36, terminal_size()[0])
            top = color(66, 82, 105, "╭" + "─" * (width - 2) + "╮")
            divider = color(66, 82, 105, "├" + "─" * (width - 2) + "┤")
            bottom = color(66, 82, 105, "╰" + "─" * (width - 2) + "╯")
            print(top)
            print(_frame_row(" GEKI  /  LIVE CAPTURE", f"{len(processes)} SOURCES ", width))
            print(_frame_row(" Choose an app with active audio output", f"THEME  {theme_names[selected_theme].upper()} ", width))
            print(divider)
            if not processes:
                print(_frame_row(" No active audio output. Start playback, then press r to refresh.", "", width))
            for index, process in enumerate(processes):
                marker = "❯ " if index == selected else "  "
                label = f"{marker}{process['name']}  (PID {process['pid']})"
                print(_frame_item(label, width, selected=index == selected))
            print(divider)
            print(_frame_row(f" {message}", "", width))
            print(bottom)

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
                    message = "List refreshed.  " + message
                except (OSError, RuntimeError, subprocess.CalledProcessError, json.JSONDecodeError) as error:
                    message = f"Refresh failed: {error}"
                if not processes:
                    message = "No active audio output. Start playback, then press r to refresh."
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
    """Save metadata used by the local library."""
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
    """List cached MP3 files, including files without metadata."""
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
    """Show the library and return the selected track and theme."""
    if not sys.stdin.isatty():
        print("GEKI: the interactive library requires a terminal.")
        return None

    tracks = load_library()
    if not tracks:
        print("GEKI: the library is empty.")
        print(f"Folder: {CACHE_DIR}")
        return None

    selected = 0
    theme_names = list(THEMES)
    selected_theme = theme_names.index(DEFAULT_THEME)
    message = "↑/↓ select track   ←/→ select theme   Enter play   d delete   q quit"
    fd = sys.stdin.fileno()
    previous = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        while True:
            clear()
            width = max(36, terminal_size()[0])
            top = color(66, 82, 105, "╭" + "─" * (width - 2) + "╮")
            divider = color(66, 82, 105, "├" + "─" * (width - 2) + "┤")
            bottom = color(66, 82, 105, "╰" + "─" * (width - 2) + "╯")
            print(top)
            print(_frame_row(" GEKI  /  LIBRARY", f"{len(tracks)} TRACK(S) ", width))
            print(_frame_row(f" {CACHE_DIR}", f"THEME  {theme_names[selected_theme].upper()} ", width))
            print(divider)
            for index, track in enumerate(tracks):
                marker = "❯ " if index == selected else "  "
                metadata = track["metadata"]
                detail = metadata.get("uploader") or os.path.basename(track["path"])
                label = f"{marker}{track['title']}  [{detail}]"
                print(_frame_item(label, width, selected=index == selected))
            print(divider)
            print(_frame_row(f" {message}", "", width))
            print(bottom)

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
                sys.stdout.write(f"\nDelete ‘{track['title']}’? (y/N) ")
                sys.stdout.flush()
                answer = os.read(fd, 1).decode("utf-8", errors="ignore")
                if answer.lower() == "y":
                    os.remove(track["path"])
                    try:
                        os.remove(metadata_path(track["path"]))
                    except FileNotFoundError:
                        pass
                    tracks.pop(selected)
                    if not tracks:
                        return None
                    selected = min(selected, len(tracks) - 1)
                    message = "Track deleted."
                else:
                    message = "Deletion cancelled."
    except KeyboardInterrupt:
        return None
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, previous)
        clear()


def is_url(source):
    """Return whether the argument is a URL rather than a local file."""
    return source.startswith("http://") or source.startswith("https://")


def download_audio(url):
    """Download audio from a URL with yt-dlp and cache it locally.

    Returns (file_path, display_title)."""

    try:
        import yt_dlp
    except ImportError:

        print("GEKI: yt-dlp is not installed.")
        print()
        print("Install it with:")
        print("  pip3 install yt-dlp")
        print()
        print("If you have not already, install ffmpeg (required to")
        print("extract audio):")
        print("  brew install ffmpeg")

        sys.exit(1)

    os.makedirs(CACHE_DIR, exist_ok=True)

    # Fetch metadata without downloading so we know the ID (cache key)
    # and title before starting the download.
    probe_opts = {
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
    }

    try:
        with yt_dlp.YoutubeDL(probe_opts) as ydl:
            info = ydl.extract_info(url, download=False)
    except Exception as error:

        print("GEKI: could not read this URL.")
        print()
        print(f"Error: {error}")

        sys.exit(1)

    video_id = info.get("id", "audio")
    title = info.get("title", video_id)

    cached_path = os.path.join(
        CACHE_DIR,
        f"{video_id}.mp3"
    )

    if os.path.exists(cached_path):
        save_track_metadata(cached_path, info)
        print(f"GEKI: '{title}' is cached; playing it now.")
        return cached_path, title

    print(f"GEKI: downloading '{title}'...")

    def progress_hook(d):

        if d["status"] == "downloading":

            pct = d.get("_percent_str", "").strip()
            sys.stdout.write(f"\r  {pct}   ")
            sys.stdout.flush()

        elif d["status"] == "finished":

            sys.stdout.write("\r  converting audio...        \n")
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
        print("GEKI: download failed.")
        print()
        print(f"Error: {error}")
        print()
        print("Check that ffmpeg is installed (brew install ffmpeg).")

        sys.exit(1)

    print()

    if os.path.exists(cached_path):
        save_track_metadata(cached_path, info)

    return cached_path, title


def build_queue(source):
    """Return a list of sources to play in sequence.

    A local file or single video URL produces a one-item list. A
    playlist URL produces one URL per track, in playlist order."""

    if not is_url(source):
        return [source]

    try:
        import yt_dlp
    except ImportError:

        print("GEKI: yt-dlp is not installed.")
        print()
        print("Install it with:")
        print("  pip3 install yt-dlp")

        sys.exit(1)

    # "extract_flat" performs a quick scan for each entry's ID, title,
    # and URL without fetching every video's available formats. This
    # avoids long delays just to detect a large playlist.
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

        print("GEKI: could not read this URL.")
        print()
        print(f"Error: {error}")

        sys.exit(1)

    is_playlist = (
        info.get("_type") == "playlist"
        or "entries" in info
    )

    if not is_playlist:
        return [source]

    entries = [e for e in info.get("entries", []) if e]

    if not entries:
        print("GEKI: playlist is empty or unavailable.")
        sys.exit(1)

    # For YouTube Mix URLs (watch?v=...&list=RD...&index=N), the index
    # identifies the selected track. Without it, playback would always
    # restart at the beginning of the mix.
    query = parse_qs(urlparse(source).query)

    if "index" in query:

        try:
            start_index = max(1, int(query["index"][0]))
        except ValueError:
            start_index = 1

        entries = entries[start_index - 1:]

        if not entries:
            print("GEKI: mix index is out of range; starting from the beginning.")
            entries = [e for e in info.get("entries", []) if e]

    playlist_title = info.get("title", "playlist")

    print(f"GEKI: playlist '{playlist_title}' — {len(entries)} track(s).")

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
            # extract_flat sometimes returns only the video ID.
            queue.append(f"https://www.youtube.com/watch?v={entry_url}")

    return queue


def resolve_source(source):
    """Return (file_path, display_title) for a local file or URL.

    URLs are downloaded through yt-dlp when needed."""

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
        print(f"Available themes: {', '.join(THEMES)}")

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

            print(f"GEKI: unknown theme '{requested}'")
            print(f"Available themes: {', '.join(THEMES)}")

            sys.exit(1)

        theme_name = requested

    if source is None and not positional and live_selection is None:

        print("GEKI: no file or URL was provided.")

        sys.exit(1)

    visualizer_fn = THEMES[theme_name]

    if live_selection is not None:
        process_info, _ = live_selection
        audio = None
        error_message = None
        try:
            print("GEKI: starting capture. Accept the macOS prompt if it appears.")
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
                        f"The audio helper stopped (exit code {audio.capture.returncode})."
                    )
            show_cursor()
            clear()
        if error_message:
            print(f"GEKI: live capture failed.\n{error_message}")
        return

    if source is None:
        source = positional[0]

    queue = build_queue(source)

    if not queue:

        print("GEKI: nothing to play.")

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
                    f"GEKI: file not found; skipping track: {filename}"
                )

                continue

            try:

                audio = AudioEngine(
                    filename
                )

            except Exception as error:

                print(
                    "GEKI: could not load this track; "
                    "skipping it."
                )

                print()
                print(f"Error: {error}")

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
