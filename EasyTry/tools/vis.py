from pathlib import Path
from PIL import Image, ImageDraw


def make_mock_frame(save_path: Path, sequence_index: int, step_id: int, action: str):
    save_path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (1280, 720), (18, 18, 22))
    draw = ImageDraw.Draw(img)

    draw.text((40, 40), f"sequence={sequence_index}", fill=(255, 255, 255))
    draw.text((40, 90), f"step={step_id}", fill=(255, 255, 255))
    draw.text((40, 140), f"action={action}", fill=(180, 255, 180))

    x0 = 200 + 40 * step_id
    x1 = min(x0 + 180, 1100)
    draw.rectangle((x0, 260, x1, 420), outline=(0, 255, 0), width=6)
    draw.line((100, 520, 1180, 520), fill=(120, 120, 120), width=5)

    img.save(save_path)
    return str(save_path)
