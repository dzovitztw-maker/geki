# GEKI

**GEKI** is a terminal audio player and reactive visualizer. It can play a local audio file, download audio from a URL supported by `yt-dlp`, or visualize the live audio output of a running macOS application. Downloaded tracks are kept in a local library that you can browse, play, and delete from the keyboard.

## Contents

- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Library](#library)
- [Live application capture](#live-application-capture)
- [Visual themes](#visual-themes)
- [How it works](#how-it-works)
- [Troubleshooting](#troubleshooting)

## Requirements

- Python 3
- `numpy`, `sounddevice`, `soundfile`, and `yt-dlp` (listed in `requirements.txt`)
- A terminal that supports ANSI escape sequences and TrueColor
- An available audio output device
- macOS 14.2 or later and the Xcode Command Line Tools for live application capture

`yt-dlp` and FFmpeg are only needed to play URLs and playlists. They are not needed for local audio files.

## Installation

Install the Python dependencies:

```sh
python3 -m pip install -r requirements.txt
```

Install FFmpeg for URL and playlist downloads:

```sh
brew install ffmpeg
```

The `geki.py` script is executable. To run it from its directory:

```sh
./geki.py "music.mp3"
```

If the `geki` command is already available through your `PATH` or a shell alias, you can run it from any directory:

```sh
geki "music.mp3"
```

The setup that makes `geki` available globally depends on your machine. To see which command your shell runs:

```sh
command -v geki
```

## Usage

### Play a local file

```sh
geki "music.mp3"
geki "/Users/me/Music/My album/track 01.mp3"
```

Quotes are useful when a path contains spaces. Supported formats depend on the formats provided by the installed `soundfile` library.

### Play a URL

```sh
geki "https://www.youtube.com/watch?v=VIDEO_ID"
```

GEKI uses `yt-dlp` to retrieve the audio. Supported sites and URLs are those supported by the installed version of `yt-dlp`.

### Play a playlist

```sh
geki "https://www.youtube.com/playlist?list=PLAYLIST_ID"
```

Tracks play one after another. For playlists with an `index` parameter, GEKI starts at the corresponding entry.

### Choose a visual theme from the command line

Add the theme name after the source, using two hyphens:

```sh
geki "music.mp3" --prism
geki "https://www.youtube.com/watch?v=VIDEO_ID" --classic
```

Available themes are `--starburst`, `--flower`, `--galaxy`, `--prism`, `--classic`, and `--dance`. The default is `starburst`.

### Stop playback

Press `Ctrl+C`. GEKI restores the terminal cursor when it exits.

## Library

Open the library of downloaded tracks:

```sh
geki -library
```

| Key | Action |
| --- | --- |
| ↑ / ↓ | Select a track |
| ← / → | Cycle through visual themes |
| Enter | Play the selected track with the displayed theme |
| `d` | Request deletion of the selected track |
| `o` or `y` | Confirm deletion |
| Any other response | Cancel deletion |
| `q` | Quit the library |

You can also provide a theme on the command line, for example `geki -library --galaxy`. The theme selected with ←/→ is used by default; an explicit option after `-library` overrides it.

The library lists MP3 files in the GEKI cache. New downloads save the title and uploader/channel name in a small JSON sidecar file. Older downloads without saved metadata still appear, usually under their file ID.

Deleting a track removes its cached MP3 and its associated JSON file. It does not affect audio files stored elsewhere.

## Live application capture

The live mode visualizes audio currently playing from an application, such as a web browser or music player:

```sh
geki -live
```

Capture uses macOS Core Audio process taps and does not require a virtual audio driver. On first use, GEKI compiles and locally signs a small native helper in `.geki-live/`. macOS then asks for system audio capture permission; allow **GEKI Live Audio** in **System Settings → Privacy & Security** to continue. The helper is compiled from `geki_audio_tap.m` using `clang`, included with the Xcode Command Line Tools.

The menu lists applications that are outputting audio at that moment. If the list is empty, start audio playback and press `r` to refresh it.

| Key | Action |
| --- | --- |
| ↑ / ↓ | Select an application |
| ← / → | Choose a visual theme |
| Enter | Start visualizing the live stream |
| `r` | Refresh the list of active audio applications |
| `q` or `Ctrl+C` | Quit |

During the stream, `Ctrl+C` stops capture and restores the terminal. The application's sound continues to its normal audio output. GEKI does not save a recording: it keeps only a fixed-size circular buffer in memory for visualization, so memory usage does not grow during a long session.

You can set the initial theme with an option, for example `geki -live --galaxy`. Use ←/→ in the menu to choose another theme.

Live capture requires macOS 14.2 or later. It is not available on Windows or Linux.

## Visual themes

| Theme | Appearance |
| --- | --- |
| `starburst` | A central core and radial spikes that react to frequency bands. |
| `flower` | A central core and petals that respond to the spectrum. |
| `galaxy` | Rotating spiral arms of stars around a bright core. |
| `prism` | A rotating, audio-shaped icosahedron over a mirrored kaleidoscope, with subtle color washes on bass hits. |
| `classic` | Vertical equalizer bars with peak markers. |
| `dance` | A colorful field of asteroids that pulse independently. |

## How it works

### Audio analysis and display

GEKI reads the audio signal, converts it to mono for analysis, and regularly measures its frequency levels. The levels are smoothed and adaptively normalized so visualizations can respond to tracks at different volumes. Audio plays while GEKI redraws the terminal around 30 times per second.

### Downloads and cache

Downloaded files are stored in:

```text
~/.geki/cache
```

GEKI uses the media ID as the filename. It stores the MP3 and, for recent downloads, a neighboring JSON file with information such as the title, URL, uploader/channel, and duration. If the MP3 is already in the cache, GEKI plays it without downloading it again.

Browse the cache with `geki -library`. Delete tracks from that menu to free disk space.

### Project structure

`geki.py` contains the interface, file player, audio analysis, and visual themes. `geki_audio_tap.m` is the small native helper used only for live capture. The generated helper binary is ignored by Git and can be rebuilt from its source.

## Troubleshooting

### `yt-dlp` is missing

```sh
python3 -m pip install -r requirements.txt
```

### FFmpeg is missing or audio conversion fails

```sh
brew install ffmpeg
```

### No sound or an audio device error

Check that macOS audio output is working and that another application is not exclusively using the device. Some devices may not support a file's sample rate; try another file or output device.

### `geki` only works in some directories

Check the result of `command -v geki`. The script must be available through your `PATH` or configured shell alias.

### An older track displays a file ID instead of its title

Files already in the cache before metadata support was added may not have a saved title. New downloads automatically create the JSON file used by the library.
