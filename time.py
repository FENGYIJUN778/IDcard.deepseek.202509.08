# -*- coding: utf-8 -*-
import os
import re
import csv
import json
import torch
import time
from PIL import Image, ImageDraw, ImageFont
from deepseek_vl2.models import DeepseekVLV2ForCausalLM, DeepseekVLV2Processor

# =========================
# 1) モデルロード / Model Load
# =========================
model_path = "deepseek-ai/deepseek-vl2-tiny"
print("🔄 モデルを読み込み中...")
processor = DeepseekVLV2Processor.from_pretrained(model_path)

# tokenizer 修正
if not hasattr(processor.tokenizer, "eos_token") or processor.tokenizer.eos_token is None:
    processor.tokenizer.eos_token = "<|endoftext|>"
if not hasattr(processor.tokenizer, "eos_token_id") or processor.tokenizer.eos_token_id is None:
    processor.tokenizer.eos_token_id = 100010

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32

model = DeepseekVLV2ForCausalLM.from_pretrained(
    model_path,
    torch_dtype=dtype,
    device_map="auto"
)

print(f"✅ モデル読み込み成功：device={model.device} / dtype={dtype}")

# =========================
# 2) フォルダ準備 / Folders
# =========================
image_folder = "images"
output_folder = "output"
os.makedirs(output_folder, exist_ok=True)

image_files = [f for f in os.listdir(image_folder)
               if f.lower().endswith((".jpg", ".jpeg", ".png"))]

if not image_files:
    print("❌ images フォルダに画像がありません")
    raise SystemExit(1)

# =========================
# 3) CSV 初期化 / CSV Init
# =========================
csv_path = "results.csv"
fieldnames = [
    "ファイル名", "在留カード番号", "氏名", "生年月日", "性別",
    "国籍", "住所", "在留資格", "在留期間", "許可年月日",
    "処理時間（秒）"
]

with open(csv_path, mode="w", newline="", encoding="utf-8-sig") as f:
    csv.DictWriter(f, fieldnames=fieldnames).writeheader()


# =========================
# 4) 信息提取 / Extract Logic
# =========================
CARDNO_RE = re.compile(r"[A-Z]{2}\d{8}[A-Z]{2}")

def _strip_bbox_noise(s: str):
    return re.sub(r"\[\d+\.\d+,\s*\d+\.\d+,\s*\d+\.\d+,\s*\d+\.\d+\]", "", s)

def extract_info(text: str):
    # JSON 解析
    json_match = re.search(r"\{.*\}", text, re.DOTALL)
    if json_match:
        try:
            data = json.loads(json_match.group(0))
            alias = {
                "在留カード番号": ["在留カード番号", "番号", "Card No", "Card Number"],
                "氏名": ["氏名", "名前"],
                "生年月日": ["生年月日", "誕生日"],
                "性別": ["性別", "Gender", "Sex"],
                "国籍": ["国籍", "Nationality"],
                "住所": ["住所", "Address"],
                "在留資格": ["在留資格", "Status"],
                "在留期間": ["在留期間", "Period"],
                "許可年月日": ["許可年月日", "許可日"]
            }
            out = {}
            for std, keys in alias.items():
                for k in keys:
                    if k in data and isinstance(data[k], str):
                        out[std] = _strip_bbox_noise(data[k].strip())
                        break
            # 卡号兜底
            if "在留カード番号" not in out:
                m = CARDNO_RE.search(text)
                if m:
                    out["在留カード番号"] = m.group(0)
            return out
        except:
            pass

    # 正则补充
    patterns = {
        "在留カード番号": r"[A-Z]{2}\d{8}[A-Z]{2}",
        "氏名": r"氏名[:：\s]*([^\n]+)",
        "生年月日": r"生年月日[:：\s]*([^\n]+)",
        "性別": r"(?:性別|Gender|Sex)[:：\s]*([^\n]+)",
        "国籍": r"(?:国籍|Nationality)[:：\s]*([^\n]+)",
        "住所": r"(?:住所|Address)[:：\s]*([^\n]+)",
        "在留資格": r"在留資格[:：\s]*([^\n]+)",
        "在留期間": r"(?:在留期間|Period)[:：\s]*([^\n]+)",
        "許可年月日": r"(?:許可日|許可年月日)[:：\s]*([^\n]+)"
    }

    out = {}
    for key, p in patterns.items():
        m = re.search(p, text)
        if m:
            out[key] = _strip_bbox_noise(m.group(1).strip()) if m.groups() else m.group(0)

    # 全局卡号兜底
    if "在留カード番号" not in out:
        m = CARDNO_RE.search(text)
        if m:
            out["在留カード番号"] = m.group(0)

    return out


# =========================
# 5) BBox 提取
# =========================
def extract_bboxes(text):
    pts = re.findall(r"\[(\d+\.\d+),\s*(\d+\.\d+),\s*(\d+\.\d+),\s*(\d+\.\d+)\]", text)
    out = []
    for p in pts:
        box = list(map(float, p))
        if all(0 <= v <= 1 for v in box):
            out.append(box)
    return out

def normalize_to_pixels(box, w, h):
    x1 = int(box[0] * w); y1 = int(box[1] * h)
    x2 = int(box[2] * w); y2 = int(box[3] * h)
    return (x1, y1, x2, y2)


# =========================
# 6) 解决 NoneType.sum BUG
# =========================
def fix_deepseek_masks(inputs):
    """为 DeepSeek-VL2 强制补齐缺失的 mask，避免 NoneType.sum 错误"""

    batch = inputs.input_ids.shape[0]
    seq_len = inputs.input_ids.shape[1]

    # attention_mask
    if getattr(inputs, "attention_mask", None) is None:
        inputs.attention_mask = torch.ones(
            (batch, seq_len), dtype=torch.long, device=model.device
        )

    # images_seq_mask
    if getattr(inputs, "images_seq_mask", None) is None:
        inputs.images_seq_mask = torch.ones(
            (batch, seq_len), dtype=torch.bool, device=model.device
        )

    # images_spatial_crop
    if getattr(inputs, "images_spatial_crop", None) is None:
        inputs.images_spatial_crop = torch.zeros(
            (batch, 4), dtype=torch.float32, device=model.device
        )

    return inputs


# =========================
# 7) 画像ループ（计时 + 输出）
# =========================
total_start = time.time()

for idx, filename in enumerate(sorted(image_files)):
    print(f"\n📂 処理中: {filename} ({idx+1}/{len(image_files)})")

    start_time = time.time()   # ★ 每张图计时

    img_path = os.path.join(image_folder, filename)
    image = Image.open(img_path).convert("RGB")
    original_image = image.copy()

    conversation = [
        {
            "role": "user",
            "content": "<image>\n在留カード情報を抽出してください。JSONのみで返答。"
        },
        {"role": "assistant", "content": ""}
    ]

    try:
        inputs = processor(
            conversations=conversation,
            images=[image],
            return_tensors="pt",
            force_batchify=True
        ).to(model.device)

        inputs.images = inputs.images.to(dtype)
        inputs = fix_deepseek_masks(inputs)  # ★ 修复 NoneType.sum

        output = model.generate(
            input_ids=inputs.input_ids,
            images=inputs.images,
            attention_mask=inputs.attention_mask,
            images_seq_mask=inputs.images_seq_mask,
            images_spatial_crop=inputs.images_spatial_crop,
            max_new_tokens=300,
            temperature=0.1,
            do_sample=False,
            pad_token_id=processor.tokenizer.eos_token_id,
            eos_token_id=processor.tokenizer.eos_token_id
        )

        raw_text = processor.tokenizer.decode(output[0], skip_special_tokens=False)

        # 抽取回复内容
        start_token = "<|Assistant|>"
        pos = raw_text.find(start_token)
        text = raw_text[pos + len(start_token):].strip()
        text = re.sub(r"<\|[^>]+\|>", "", text).strip()

        info = extract_info(text)

        # ===== BBox 可视化 =====
        bboxes = extract_bboxes(raw_text)
        draw = ImageDraw.Draw(original_image)
        w, h = original_image.size

        try:
            font = ImageFont.truetype("arial.ttf", 20)
        except:
            font = ImageFont.load_default()

        for i, b in enumerate(bboxes):
            x1, y1, x2, y2 = normalize_to_pixels(b, w, h)
            draw.rectangle((x1, y1, x2, y2), outline="red", width=3)
            draw.text((x1, y1 - 25), f"領域{i+1}", fill="red", font=font)

        out_path = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}_bbox.jpg")
        original_image.save(out_path)

        # ===== 写入 CSV =====
        elapsed = time.time() - start_time
        print(f"⏱ この画像の処理時間: {elapsed:.2f} 秒")

        with open(csv_path, "a", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            row = {"ファイル名": filename}
            for k in fieldnames[1:-1]:
                row[k] = info.get(k, "")
            row["処理時間（秒）"] = f"{elapsed:.2f}"
            writer.writerow(row)

        print(f"✅ 抽出成功: {filename}")

    except Exception as e:
        print(f"❌ エラー: {filename} -> {e}")

# ===== 全部图片耗时 =====
total_elapsed = time.time() - total_start
print(f"\n⏳ 全部処理時間: {total_elapsed:.2f} 秒")
print(f"📉 平均処理時間: {total_elapsed / len(image_files):.2f} 秒")

print("\n🎉 完了！")
print(f"📝 CSV: {csv_path}")
print(f"🖼 画像出力: {output_folder}")
