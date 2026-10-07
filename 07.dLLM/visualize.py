"""Sample from a trained diffusion miniGPT (Flax NNX) and render the denoising process as a GIF.

    python visualize.py --out assets/diffusion_uncond.gif
    python visualize.py --prompt "Once upon a time, there was a little dog" --out assets/diffusion_prompt.gif
    python visualize.py --strategy random --out assets/diffusion_random.gif
"""

import argparse
import os

import jax
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from data import EOS_ID, MASK_ID, CompactTokenizer
from checkpoint import load_model
from diffusion import generate

# ---- style -----------------------------------------------------------------
W = 1100
MARGIN = 36
HEADER_H = 96
BG = (250, 250, 247)
FG = (40, 40, 40)
MASK_FILL = (214, 214, 214)
EOS_OUTLINE = (200, 200, 200)
NEW_BG = (255, 205, 70)
NEW_FG = (20, 20, 20)
PROMPT_FG = (25, 100, 200)
ACCENT = (230, 120, 30)


def load_font(size, bold=False):
    names = (["DejaVuSans-Bold.ttf"] if bold else ["DejaVuSans.ttf"])
    for d in ["/usr/share/fonts/truetype/dejavu/", "/Library/Fonts/", ""]:
        for n in names:
            try:
                return ImageFont.truetype(os.path.join(d, n), size)
            except OSError:
                pass
    return ImageFont.load_default()


def layout(pieces, font, line_h, width):
    """pieces: list of (text, kind). Returns list of (x, y, w, text, kind) and total height."""
    x, y = MARGIN, 0
    placed = []
    for text, kind in pieces:
        if kind in ("mask", "eos", "eos_new"):  # fixed-size boxes
            w = 26 if kind == "mask" else 12
            if x + w > MARGIN + width:
                x, y = MARGIN, y + line_h
            placed.append((x, y, w, "", kind))
            x += w + 3
            continue
        for j, part in enumerate(text.split("\n")):
            if j > 0:
                if x > MARGIN:  # collapse consecutive newlines into one paragraph break
                    x, y = MARGIN, y + int(line_h * 1.5)
                part = part.lstrip(" ")
            if not part:
                continue
            w = font.getlength(part)
            # word wrap -- only break before a new word, never inside one / before punctuation
            glued = not part.startswith(" ") and x > MARGIN and x + w <= W - 8
            if x + w > MARGIN + width and not glued:
                x, y = MARGIN, y + line_h
                part = part.lstrip(" ")
                w = font.getlength(part)
            placed.append((x, y, w, part, kind))
            x += w
    return placed, y + line_h


def frame_pieces(tok, tokens, new_mask, n_prompt):
    """Turn one diffusion state into (text, kind) pieces for drawing."""
    pieces = []
    for i, (t, new) in enumerate(zip(tokens.tolist(), new_mask.tolist())):
        if t == MASK_ID:
            pieces.append(("", "mask"))
        elif t == EOS_ID:
            pieces.append(("", "eos_new" if new else "eos"))
        else:
            kind = "prompt" if i < n_prompt else ("new" if new else "old")
            pieces.append((tok.token_str(t), kind))
    return pieces


def render(history, tok, n_prompt, title, subtitle):
    font = load_font(18)
    font_title = load_font(24, bold=True)
    font_small = load_font(15)
    line_h = 28
    text_w = W - 2 * MARGIN

    layouts = [layout(frame_pieces(tok, tokens, new, n_prompt), font, line_h, text_w)
               for tokens, new in history]
    H = HEADER_H + max(h for _, h in layouts) + MARGIN

    frames = []
    n_steps = max(1, len(history) - 2)  # last entry is the held, un-highlighted final frame
    L = len(history[0][0])
    for s, ((tokens, _), (placed, _)) in enumerate(zip(history, layouts)):
        s = min(s, n_steps)
        img = Image.new("RGB", (W, H), BG)
        d = ImageDraw.Draw(img)
        d.text((MARGIN, 18), title, font=font_title, fill=FG)
        n_masked = int((tokens == MASK_ID).sum())
        d.text((MARGIN, 52), f"{subtitle}   |   step {s:3d}/{n_steps}   |   masked {n_masked:3d}/{L}",
               font=font_small, fill=(110, 110, 110))
        # progress bar
        bx0, bx1, by = MARGIN, W - MARGIN, 78
        d.rounded_rectangle([bx0, by, bx1, by + 6], radius=3, fill=(225, 225, 225))
        d.rounded_rectangle([bx0, by, bx0 + (bx1 - bx0) * (1 - n_masked / L), by + 6], radius=3, fill=ACCENT)

        for x, y, w, text, kind in placed:
            yy = HEADER_H + y
            if kind == "mask":
                d.rounded_rectangle([x, yy + 5, x + w, yy + 23], radius=4, fill=MASK_FILL)
            elif kind == "eos_new":
                d.rounded_rectangle([x, yy + 5, x + w, yy + 23], radius=3, fill=NEW_BG)
            elif kind == "eos":
                d.rounded_rectangle([x, yy + 5, x + w, yy + 23], radius=3, outline=EOS_OUTLINE)
            elif kind == "new":
                d.rounded_rectangle([x - 1, yy + 2, x + w + 1, yy + 26], radius=4, fill=NEW_BG)
                d.text((x, yy + 3), text, font=font, fill=NEW_FG)
            elif kind == "prompt":
                d.text((x, yy + 3), text, font=font, fill=PROMPT_FG)
            else:
                d.text((x, yy + 3), text, font=font, fill=FG)
        frames.append(img)
    return frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_dir", default="out", help="directory holding ckpt/ and ckpt.json")
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--prompt", default="")
    ap.add_argument("--length", type=int, default=256)
    ap.add_argument("--steps", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top_k", type=int, default=None)
    ap.add_argument("--strategy", default="random", choices=["confidence", "random"])
    ap.add_argument("--block_len", type=int, default=None,
                    help="semi-autoregressive block size (None = denoise the whole sequence at once)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--frame_ms", type=int, default=50)
    ap.add_argument("--out", default="assets/diffusion.gif")
    args = ap.parse_args()

    tok = CompactTokenizer(os.path.join(args.data_dir, "vocab.json"))
    model, _ = load_model(args.ckpt_dir)

    prompt_ids = tok.encode(args.prompt) if args.prompt else None
    final, history = generate(model, prompt_ids, length=args.length, steps=args.steps,
                              temperature=args.temperature, top_k=args.top_k, strategy=args.strategy,
                              block_len=args.block_len, key=jax.random.PRNGKey(args.seed))
    print(tok.decode(final))

    title = "Text diffusion with miniGPT in Flax NNX  (trained on TinyStories)"
    subtitle = f"{args.strategy} unmasking"
    if args.block_len:
        subtitle += f", blocks of {args.block_len}"
    subtitle += f", T={args.temperature}"
    if args.prompt:
        subtitle += ", prompt in blue"
    # append a final clean frame (no highlights) that is held at the end
    history = history + [(final, np.zeros_like(final, dtype=bool))]
    frames = render(history, tok, len(prompt_ids or []), title, subtitle)
    durations = [600] + [args.frame_ms] * (len(frames) - 2) + [3500]

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    frames = [f.quantize(colors=64, method=Image.Quantize.MEDIANCUT) for f in frames]
    frames[0].save(args.out, save_all=True, append_images=frames[1:], duration=durations, loop=0,
                   optimize=True)
    print(f"saved {args.out} ({len(frames)} frames)")


if __name__ == "__main__":
    main()
