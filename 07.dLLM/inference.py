import argparse
import os
import subprocess
import tempfile

if (
    "JAX_PLATFORMS" not in os.environ
    and not os.path.exists("/dev/accel0")
    and not os.path.exists("/dev/vfio/0")
    and not os.path.exists("/dev/nvidia0")
):
    os.environ["JAX_PLATFORMS"] = "cpu"

import jax
import jax.numpy as jnp
import flax.nnx as nnx
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import train as tr


def build_mask_char_flags(text: str) -> list[bool]:
    flags = [False] * len(text)
    mask_token = "[MASK]"
    start = 0
    while True:
        idx = text.find(mask_token, start)
        if idx == -1:
            break
        for j in range(idx, min(len(text), idx + len(mask_token))):
            flags[j] = True
        start = idx + len(mask_token)
    for i, ch in enumerate(text):
        if ch == "_":
            flags[i] = True
    return flags


def load_model(checkpoint_path: str) -> tr.MiniGPT:
    model = tr.create_model(rngs=nnx.Rngs(0))
    tr.load_checkpoint(model, checkpoint_path)
    return model


def write_diffusion_gif(
    trace,
    prompt_len: int,
    output_path: str,
    frame_ms: int = 200,
    chars_per_line: int = 96,
) -> None:
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    font = ImageFont.load_default()
    pad = 20
    line_h = 18
    header_h = 70
    char_w = 8
    max_lines = max(
        1,
        (max(len(item["text"]) for item in trace) + chars_per_line - 1) // chars_per_line,
    )
    width = pad * 2 + chars_per_line * char_w
    height = header_h + pad * 2 + max_lines * line_h

    frames = []
    for item in trace:
        im = Image.new("RGB", (width, height), "#faf8f2")
        draw = ImageDraw.Draw(im)

        draw.text((pad, 14), "Diffusion Inference Trace", fill="#111827", font=font)
        draw.text(
            (pad, 34),
            f"Step: {item['label']}    Active masks: {item['masked']}",
            fill="#374151",
            font=font,
        )

        text = item["text"]
        mask_flags = build_mask_char_flags(text)
        for i, ch in enumerate(text):
            row = i // chars_per_line
            col = i % chars_per_line
            x = pad + col * char_w
            y = header_h + row * line_h

            if i < prompt_len:
                draw.text((x, y), ch, fill="#2563eb", font=font)
            elif mask_flags[i]:
                draw.rectangle(
                    [x - 1, y - 1, x + char_w, y + line_h - 4], fill="#fde68a"
                )
                draw.text((x, y), ch, fill="#92400e", font=font)
            else:
                draw.text((x, y), ch, fill="#000000", font=font)

        frames.append(im)

    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=max(20, frame_ms),
        loop=0,
    )


def write_diffusion_video(
    trace,
    prompt_len: int,
    output_path: str,
    frame_ms: int = 180,
    chars_per_line: int = 120,
) -> None:
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    width, height = 1280, 720
    fps = max(1, int(round(1000.0 / max(20, frame_ms))))

    font_paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationMono-Regular.ttf",
    ]
    font_title = None
    font_meta = None
    font_mono = None
    for path in font_paths:
        if os.path.exists(path):
            font_title = ImageFont.truetype(path, 40)
            font_meta = ImageFont.truetype(path, 22)
            font_mono = ImageFont.truetype(path, 24)
            break
    if font_title is None or font_meta is None or font_mono is None:
        font_title = ImageFont.load_default()
        font_meta = ImageFont.load_default()
        font_mono = ImageFont.load_default()

    def parse_t(label: str) -> int:
        if label == "init":
            return tr.T
        if label.startswith("t="):
            try:
                return int(label.split("=", 1)[1])
            except ValueError:
                return tr.T
        return tr.T

    mask_history = [item["masked"] for item in trace]
    max_masks = max(mask_history) if mask_history else 1

    # Layout
    margin = 44
    header_h = 138
    meta_h = 74
    text_top = margin + header_h + meta_h + 16
    text_h = height - text_top - margin
    left_w = int(width * 0.72)
    right_w = width - (2 * margin + left_w + 16)
    left_x0 = margin
    left_x1 = left_x0 + left_w
    right_x0 = left_x1 + 16
    right_x1 = right_x0 + right_w

    # Typography metrics
    char_w = max(8, font_mono.getbbox("M")[2] - font_mono.getbbox("M")[0])
    line_h = max(24, font_mono.getbbox("Ag")[3] - font_mono.getbbox("Ag")[1] + 8)
    inner_pad = 22
    max_chars = max(20, min(chars_per_line, (left_w - 2 * inner_pad) // char_w))
    max_lines = max(4, (text_h - 2 * inner_pad) // line_h)

    with tempfile.TemporaryDirectory(prefix="diff_trace_") as tmpdir:
        for frame_idx, item in enumerate(trace):
            im = Image.new("RGB", (width, height), "#f7f7f5")
            draw = ImageDraw.Draw(im)

            # Panels
            draw.rounded_rectangle(
                [margin, margin, width - margin, margin + header_h],
                radius=20,
                fill=(255, 255, 255),
                outline=(214, 214, 214),
                width=1,
            )
            draw.rounded_rectangle(
                [
                    margin,
                    margin + header_h + 10,
                    width - margin,
                    margin + header_h + meta_h + 10,
                ],
                radius=16,
                fill=(252, 252, 252),
                outline=(220, 220, 220),
                width=1,
            )
            draw.rounded_rectangle(
                [left_x0, text_top, left_x1, text_top + text_h],
                radius=18,
                fill=(255, 255, 255),
                outline=(220, 220, 220),
                width=1,
            )
            draw.rounded_rectangle(
                [right_x0, text_top, right_x1, text_top + text_h],
                radius=18,
                fill=(255, 255, 255),
                outline=(220, 220, 220),
                width=1,
            )

            # Header text
            draw.text(
                (margin + 24, margin + 20),
                "Diffusion Decoding",
                fill=(25, 25, 25),
                font=font_title,
            )

            # Progress bar
            current_t = parse_t(item["label"])
            progress = min(1.0, max(0.0, 1.0 - (current_t / max(1, tr.T))))
            bar_x0 = width - margin - 430
            bar_y0 = margin + 44
            bar_x1 = width - margin - 26
            bar_y1 = bar_y0 + 22
            draw.rounded_rectangle(
                [bar_x0, bar_y0, bar_x1, bar_y1], radius=11, fill=(234, 234, 234)
            )
            fill_w = int((bar_x1 - bar_x0) * progress)
            if fill_w > 0:
                draw.rounded_rectangle(
                    [bar_x0, bar_y0, bar_x0 + fill_w, bar_y1],
                    radius=11,
                    fill=(65, 133, 243),
                )
            draw.text(
                (bar_x0, bar_y1 + 10),
                f"progress {int(progress * 100):3d}%",
                fill=(95, 95, 95),
                font=font_meta,
            )

            # Meta row
            draw.text(
                (margin + 24, margin + header_h + 30),
                f"step {item['label']}    active_masks {item['masked']}    total_steps {tr.T}",
                fill=(60, 60, 60),
                font=font_meta,
            )

            text = item["text"]
            max_chars_total = max_chars * max_lines
            if len(text) > max_chars_total:
                text = text[: max_chars_total - 3] + "..."
            mask_flags = build_mask_char_flags(text)

            # Token grid
            for i, ch in enumerate(text):
                row = i // max_chars
                col = i % max_chars
                x = left_x0 + inner_pad + col * char_w
                y = text_top + inner_pad + row * line_h

                if i < prompt_len:
                    draw.text((x, y), ch, fill=(37, 99, 235), font=font_mono)
                elif mask_flags[i]:
                    draw.rounded_rectangle(
                        [x - 2, y - 2, x + char_w + 2, y + line_h - 6],
                        radius=4,
                        fill=(253, 230, 138),
                    )
                    draw.text((x, y), ch, fill=(146, 64, 14), font=font_mono)
                else:
                    draw.text((x, y), ch, fill=(0, 0, 0), font=font_mono)

            # Mask-count mini chart
            chart_pad = 18
            cx0 = right_x0 + chart_pad
            cy0 = text_top + chart_pad + 16
            cx1 = right_x1 - chart_pad
            cy1 = text_top + text_h - chart_pad - 24
            draw.text(
                (cx0, text_top + chart_pad - 4),
                "Mask Count Timeline",
                fill=(70, 70, 70),
                font=font_meta,
            )
            draw.rectangle([cx0, cy0, cx1, cy1], outline=(195, 195, 195), width=1)

            if len(mask_history) > 1:
                points = []
                for j, m in enumerate(mask_history):
                    tx = cx0 + int((cx1 - cx0) * (j / (len(mask_history) - 1)))
                    ty = cy1 - int((cy1 - cy0) * (m / max(1, max_masks)))
                    points.append((tx, ty))
                for p0, p1 in zip(points[:-1], points[1:]):
                    draw.line([p0, p1], fill=(65, 133, 243), width=3)

                cur_x = cx0 + int(
                    (cx1 - cx0) * (frame_idx / max(1, len(mask_history) - 1))
                )
                cur_y = cy1 - int((cy1 - cy0) * (item["masked"] / max(1, max_masks)))
                draw.ellipse(
                    [cur_x - 6, cur_y - 6, cur_x + 6, cur_y + 6], fill=(230, 129, 57)
                )
            draw.text(
                (cx0, cy1 + 8),
                f"current masks: {item['masked']}",
                fill=(95, 95, 95),
                font=font_meta,
            )

            frame_path = os.path.join(tmpdir, f"frame_{frame_idx:05d}.png")
            im.save(frame_path)

        cmd = [
            "ffmpeg",
            "-y",
            "-framerate",
            str(fps),
            "-i",
            os.path.join(tmpdir, "frame_%05d.png"),
            "-vcodec",
            "libx264",
            "-crf",
            "18",
            "-preset",
            "slow",
            "-pix_fmt",
            "yuv420p",
            output_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg failed:\n{result.stderr}")


def generate_from_prompt(
    model: tr.MiniGPT,
    prompt: str,
    gen_len: int = 128,
    temperature: float = 1.0,
    capture_trace: bool = False,
    trace_every: int = 1,
    seed: int = 1337,
):
    prompt_tokens = tr.encode(prompt)
    if len(prompt_tokens) == 0:
        raise ValueError("Prompt cannot be empty.")
    if len(prompt_tokens) >= tr.block_size:
        raise ValueError(f"Prompt too long. Max length is {tr.block_size - 1} tokens.")

    max_gen = tr.block_size - len(prompt_tokens)
    gen_len = max(1, min(gen_len, max_gen))
    total_len = len(prompt_tokens) + gen_len

    x_np = np.full((1, tr.block_size), tr.mask_token_id, dtype=np.int32)
    x_np[0, : len(prompt_tokens)] = np.asarray(prompt_tokens, dtype=np.int32)
    x = jnp.asarray(x_np)

    fixed_mask_np = np.ones((1, tr.block_size), dtype=bool)
    gen_slice = slice(len(prompt_tokens), total_len)
    fixed_mask_np[:, gen_slice] = False
    fixed_mask = jnp.asarray(fixed_mask_np)

    trace = []
    if capture_trace:
        x_host = np.asarray(x[0])
        trace.append(
            {
                "label": "init",
                "masked": int(np.sum(x_host[gen_slice] == tr.mask_token_id)),
                "text": tr.decode(x_host[:total_len].tolist()),
            }
        )

    rng = jax.random.PRNGKey(seed)
    temp_arr = jnp.asarray(temperature, dtype=jnp.float32)
    gen_positions = total_len - len(prompt_tokens)

    for t in reversed(range(1, tr.T + 1)):
        rng, step_key = jax.random.split(rng)
        t_tensor = jnp.asarray([t], dtype=jnp.int32)
        if t > 1:
            next_mask_ratio = 1.0 - tr.survival_prob(t - 1)
            k = int(next_mask_ratio * gen_positions)
        else:
            k = 0
        k_arr = jnp.asarray(k, dtype=jnp.int32)
        x = tr._reverse_diffusion_step(
            model, x, t_tensor, fixed_mask, k_arr, step_key, temp_arr
        )

        if capture_trace and (t % trace_every == 0 or t == 1):
            x_host = np.asarray(x[0])
            trace.append(
                {
                    "label": f"t={t}",
                    "masked": int(np.sum(x_host[gen_slice] == tr.mask_token_id)),
                    "text": tr.decode(x_host[:total_len].tolist()),
                }
            )

    t0 = jnp.asarray([0], dtype=jnp.int32)
    x = tr._final_denoise_step(model, x, t0, fixed_mask)

    x_host = np.asarray(x[0])
    output = tr.decode(x_host[:total_len].tolist())
    if capture_trace:
        trace.append(
            {
                "label": "t=0",
                "masked": int(np.sum(x_host[gen_slice] == tr.mask_token_id)),
                "text": output,
            }
        )

    prompt_prefix = tr.decode(prompt_tokens)
    return output, trace, len(prompt_prefix)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="artifacts/models/minigpt_tinystories_ckpt",
    )
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--gen-len", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--viz-gif", type=str, default="")
    parser.add_argument("--viz-video", type=str, default="")
    parser.add_argument("--trace-every", type=int, default=10)
    parser.add_argument("--gif-frame-ms", type=int, default=180)
    args = parser.parse_args()

    capture_trace = bool(args.viz_gif or args.viz_video)
    model = load_model(args.checkpoint)
    out, trace, prompt_len = generate_from_prompt(
        model,
        prompt=args.prompt,
        gen_len=args.gen_len,
        temperature=args.temperature,
        capture_trace=capture_trace,
        trace_every=max(1, args.trace_every),
        seed=args.seed,
    )
    print(out)
    if args.viz_gif:
        write_diffusion_gif(
            trace, prompt_len, args.viz_gif, frame_ms=args.gif_frame_ms
        )
        print(f"saved diffusion gif: {args.viz_gif}")
    if args.viz_video:
        write_diffusion_video(
            trace, prompt_len, args.viz_video, frame_ms=args.gif_frame_ms
        )
        print(f"saved diffusion video: {args.viz_video}")


if __name__ == "__main__":
    main()
