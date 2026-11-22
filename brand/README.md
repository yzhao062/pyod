# PyOD Brand Assets

Logo and mark files for the PyOD ecosystem. The palette is cardinal `#990000` (the
flagged outlier) and navy `#2b5597` (the inlier distribution), matching
[pyod.dev](https://pyod.dev).

| File | Use |
|---|---|
| `pyod-icon.svg` | App-style icon on a light rounded tile. Stays legible on any background; used in the README header. |
| `pyod-mark.svg` / `pyod-mark-dark.svg` | Transparent mark (ring plus outlier dot). Use the `-dark` variant on dark backgrounds. |
| `pyod-logo.svg` / `pyod-logo-dark.svg` | Mark plus the "PyOD" wordmark. Use the `-dark` variant on dark backgrounds. |
| `team/core-team.png` | Team card shown in the Core Team section of the README and the docs landing page. Drawn at 2x (1800 px wide) for display at 900 px. |
| `team/build_team_image.py` | Rebuilds `core-team.png` with Pillow: `python brand/team/build_team_image.py`. Edit its `CORE` and `CONTRIBUTORS` lists to change the team; GitHub avatars are downloaded at build time. |
| `team/yue-zhao.jpg` / `team/xiyang-hu.jpg` / `team/winston-li.jpg` | 240 px headshots used as inputs to the team card; the first two come from the leads' homepages. |

The wordmark in the logo files uses a system monospace font. For a fixed rendering
everywhere, outline the text to paths before export.
