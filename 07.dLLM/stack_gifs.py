"""Stack several diffusion GIFs vertically into one, with a labelled banner per panel.

    python stack_gifs.py   # uses the three GIFs in assets/ -> assets/diffusion_stacked.gif
"""

import argparse

from PIL import Image, ImageDraw

from visualize import BG, FG, MASK_FILL, NEW_BG, PROMPT_FG, EOS_OUTLINE, load_font

DEFAULT_PANELS = [
    ("assets/diffusion_prompt.gif", "A", "Prompted  (\"Once upon a time\")",
     "random-order unmasking  ·  MDLM ancestral sampler", (230, 120, 30)),
    ("assets/diffusion_uncond.gif", "B", "Unconditional  (no prompt)",
     "random-order unmasking  ·  MDLM ancestral sampler", (40, 150, 90)),
    ("assets/diffusion_block_confidence.gif", "C", "Prompted  (\"Once upon a time\")",
     "confidence-based unmasking  ·  semi-autoregressive blocks of 32 (LLaDA)", (120, 70, 200)),
]
CROP_TOP = 46        # drop each panel's own big title line (keep its step/progress line)
BANNER_H = 44
HEADER_H = 92
GAP = 14


def gif_frames(path):
    """Yield (RGB frame, duration_ms) sequentially."""
    im = Image.open(path)
    for i in range(im.n_frames):
        im.seek(i)
        yield im.convert("RGB"), im.info.get("duration", 50)


def draw_banner(d, y, w, letter, title, desc, color):
    d.rectangle([0, y, w, y + BANNER_H], fill=(238, 238, 234))
    d.rectangle([0, y, 10, y + BANNER_H], fill=color)
    # letter badge
    d.rounded_rectangle([26, y + 8, 54, y + 36], radius=6, fill=color)
    f_badge, f_title, f_desc = load_font(20, bold=True), load_font(19, bold=True), load_font(16)
    d.text((40, y + 22), letter, font=f_badge, fill=(255, 255, 255), anchor="mm")
    d.text((68, y + 11), title, font=f_title, fill=FG)
    d.text((68 + f_title.getlength(title) + 18, y + 14), desc, font=f_desc, fill=(90, 90, 90))


def draw_header(d, w):
    d.text((24, 16), "Text diffusion with miniGPT in Flax NNX (TinyStories): 3 sampling settings",
           font=load_font(24, bold=True), fill=FG)
    # legend
    f = load_font(15)
    x, y = 24, 58
    items = [("mask", "masked"), ("new", "revealed this step"), ("prompt", "prompt"), ("eos", "<eos> padding")]
    for kind, label in items:
        if kind == "mask":
            d.rounded_rectangle([x, y + 2, x + 24, y + 18], radius=4, fill=MASK_FILL)
            x += 30
        elif kind == "new":
            d.rounded_rectangle([x, y + 1, x + 34, y + 19], radius=4, fill=NEW_BG)
            d.text((x + 4, y + 1), "abc", font=f, fill=FG)
            x += 40
        elif kind == "prompt":
            d.text((x, y + 1), "abc", font=f, fill=PROMPT_FG)
            x += 32
        else:
            d.rounded_rectangle([x, y + 2, x + 11, y + 18], radius=3, outline=EOS_OUTLINE)
            x += 17
        d.text((x, y + 1), label, font=f, fill=(90, 90, 90))
        x += f.getlength(label) + 28


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="assets/diffusion_stacked.gif")
    ap.add_argument("--frame_ms", type=int, default=50)
    ap.add_argument("--hold_ms", type=int, default=4000)
    args = ap.parse_args()

    panels = DEFAULT_PANELS
    sizes, counts = [], []
    for p in panels:
        im = Image.open(p[0])
        sizes.append(im.size)
        counts.append(im.n_frames)
    W = max(w for w, _ in sizes)
    heights = [h - CROP_TOP for _, h in sizes]
    H = HEADER_H + sum(BANNER_H + h + GAP for h in heights)
    n_out = max(counts)

    # static background: header + banners
    base = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(base)
    draw_header(d, W)
    ys, y = [], HEADER_H
    for (_, letter, title, desc, color), h in zip(panels, heights):
        draw_banner(d, y, W, letter, title, desc, color)
        ys.append(y + BANNER_H)
        y += BANNER_H + h + GAP

    # Walk all GIFs in lock-step; shorter ones hold their last (final) frame.
    iters = [gif_frames(p[0]) for p in panels]
    current = [None] * len(panels)
    frames = []
    for i in range(n_out):
        for k, it in enumerate(iters):
            if i < counts[k]:
                current[k] = next(it)[0]
        canvas = base.copy()
        for k, fr in enumerate(current):
            canvas.paste(fr.crop((0, CROP_TOP, fr.width, fr.height)), (0, ys[k]))
        frames.append(canvas.quantize(colors=96, method=Image.Quantize.MEDIANCUT))

    durations = [600] + [args.frame_ms] * (n_out - 2) + [args.hold_ms]
    frames[0].save(args.out, save_all=True, append_images=frames[1:], duration=durations,
                   loop=0, optimize=True)
    print(f"saved {args.out}: {W}x{H}, {n_out} frames")


if __name__ == "__main__":
    main()
