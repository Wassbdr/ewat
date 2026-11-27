# Terminal demo, rendered by showcase/make_showcase.sh (needs vhs + ttyd + ffmpeg).
# Live computation on SYNTHETIC data; the typer was trained in step 1 of make_showcase.sh.
Output showcase/.cache/terminal.mp4
Set Shell "bash"
Set FontSize 17
Set Width 1280
Set Height 720
Set Padding 24
Set Theme "Catppuccin Mocha"
Set TypingSpeed 35ms

Hide
Type "export PS1='$ ' && clear"
Enter
Show

Type "# EWAT · live run on a SYNTHETIC episode (seeded generator, no cluster data)"
Enter
Sleep 600ms
Type "ewat demo --scenario fail_slow_latency --seed 0 --delay 0.12"
Sleep 400ms
Enter
Sleep 9s
Type "# → chart: showcase/poster.png"
Sleep 1.5s
