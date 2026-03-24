# -*- coding: utf-8 -*-
import os
import re
import csv
import json
import torch
from PIL import Image, ImageDraw, ImageFont
from deepseek_vl2.models import DeepseekVLV2ForCausalLM, DeepseekVLV2Processor
# ===========当前设备==============
print("🔥 CUDA 是否可用：", torch.cuda.is_available())
print("🖥️ 当前设备：", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
print("🚀 PyTorch 使用的设备：", torch.device("cuda" if torch.cuda.is_available() else "cpu"))

# =========================
# 1) モデルロード / Model Load
# =========================
model_path = "deepseek-ai/deepseek-vl2-tiny"
print("🔄 モデルを読み込み中...")
processor = DeepseekVLV2Processor.from_pretrained(model_path)

# 一部 tokenizer の EOS が None の環境対策
if not hasattr(processor.tokenizer, 'eos_token') or processor.tokenizer.eos_token is None:
    processor.tokenizer.eos_token = "<|endoftext|>"
if not hasattr(processor.tokenizer, 'eos_token_id') or processor.tokenizer.eos_token_id is None:
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
               if f.lower().endswith((".jpg", ".jpeg", ".png"))] if os.path.isdir(image_folder) else []
if not image_files:
    print("❌ 画像が見つかりません（images フォルダに .jpg/.png を入れてください）")
    raise SystemExit(1)

# =========================
# 3) CSV 初期化 / CSV Init
# =========================
csv_path = "results.csv"
fieldnames = [
    "ファイル名", "在留カード番号", "氏名", "生年月日", "性別",
    "国籍", "住所", "在留資格", "在留期間", "許可年月日"
]
with open(csv_path, mode="w", newline="", encoding="utf-8-sig") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

# ==========================================================
# 4) 情報抽出関数 / Info Extraction (JSON優先 + 正則表現 + 兜底)
# ==========================================================
CARDNO_RE = re.compile(r"[A-Z]{2}\d{8}[A-Z]{2}")

def _strip_bbox_noise(val: str) -> str:
    # 消除输出中类似 [0.12, 0.34, 0.56, 0.78] 的坐标噪声
    return re.sub(r"\[\d+\.\d+,\s*\d+\.\d+,\s*\d+\.\d+,\s*\d+\.\d+\]", "", val)

def extract_info(text: str):
    # --- ① JSON 解析を最優先 ---
    json_match = re.search(r"\{.*\}", text, re.DOTALL)
    if json_match:
        try:
            data = json.loads(json_match.group(0))
            # 鍵名の別名吸収
            alias_map = {
                "在留カード番号": ["在留カード番号", "カード番号", "番号", "Card No", "CardNo", "カードNo", "Card Number"],
                "氏名": ["氏名", "氏名（漢字）", "名前", "Name_ja"],
                "生年月日": ["生年月日", "誕生日", "Birthdate", "DOB"],
                "性別": ["性別", "Gender", "Sex"],
                "国籍": ["国籍", "国籍・地域", "Nationality"],
                "住所": ["住所", "住居地", "居住地", "Address"],
                "在留資格": ["在留資格", "Status"],
                "在留期間": ["在留期間", "在留期限", "Period"],
                "許可年月日": ["許可年月日", "許可日", "Date of permission"]
            }
            norm = {}
            for std_key, aliases in alias_map.items():
                for a in aliases:
                    if a in data and isinstance(data[a], str) and data[a].strip():
                        norm[std_key] = _strip_bbox_noise(data[a].strip())
                        break
            # JSONにカード番号がなければ、全文から兜底
            if "在留カード番号" not in norm:
                m = CARDNO_RE.search(text)
                if m:
                    norm["在留カード番号"] = m.group(0)
            return norm
        except Exception:
            pass

    # --- ② 正則表現フォールバック ---
    patterns = {
        "在留カード番号": r"(?:在留カード番号|カード番号|番号|Card\s*No\.?|Card\s*Number)\s*[：:\s]*([A-Z]{2}\d{8}[A-Z]{2})",
        "氏名": r"(氏名)[：:\s]*([^\n]+)",
        "生年月日": r"(生年月日)[：:\s]*([^\n]+)",
        # 性別：英語/一文字も拾う
        "性別": r"(?:性別|Sex|Gender)[：:\s]*([MF]|Male|Female|男|女|男性|女性|その他|他)\b",
        "国籍": r"(?:国籍|国籍[／/・]?\s*地域|Nationality)[：:\s]*([^\n]+)",
        "住所": r"(?:住所|住居地|居住地|Address)[：:\s]*([^\n]+)",
        "在留資格": r"(在留資格)[：:\s]*([^\n]+)",
        "在留期間": r"(?:在留期間|在留期限|Period)[：:\s]*([^\n]+)",
        "許可年月日": r"(?:許可年月日|許可日|Date\s*of\s*permission)[：:\s]*([^\n]+)"
    }
    result = {}
    for key, pattern in patterns.items():
        m = re.search(pattern, text)
        if m:
            # 通常は第2捕获组为值；卡号可能只有1个组
            value = m.group(2) if len(m.groups()) >= 2 else m.group(1)
            result[key] = _strip_bbox_noise(value.strip())

    # --- ③ 兜底：没有标签也全局搜卡号 ---
    if "在留カード番号" not in result:
        m = CARDNO_RE.search(text)
        if m:
            result["在留カード番号"] = m.group(0)

    return result if result else text

# ===================================================
# 5) 在留資格 归一化 / Status normalization (核心补丁)
# ===================================================
LEGAL_STATUSES = [
    "留学","技術・人文知識・国際業務","家族滞在","特定活動","永住者",
    "日本人の配偶者等","定住者","技能","短期滞在","文化活動",
    "研修","研究","介護","高度専門職","高度専門職1号","高度専門職2号",
    "企業内転勤","経営・管理","教授","教育","医療","興行","宗教","報道","芸術"
]

def normalize_status(raw: str) -> str:
    """把学生/Student 等俗称统一为留学；若命中法定名称直接返回"""
    if not isinstance(raw, str) or not raw.strip():
        return raw
    s = raw.strip()
    # 先直接命中合法名
    for legal in LEGAL_STATUSES:
        if legal in s:
            return legal
    # 常见俗称 → 留学
    if re.search(r"(学生|留学生|student|college\s*student)", s, re.I):
        return "留学"
    # 英文/误译兜底
    if re.search(r"(study|school)", s, re.I):
        return "留学"
    return s

def pick_status_from_text(full_text: str, current: str) -> str:
    """
    锚定“在留資格”这一行从原文再确认一次；
    如果文本中能明确命中法定名称，优先用它，否则对 current 做一次 normalize。
    """
    try:
        m = re.search(r"在留資格[：:\s]*([^\n\r]+)", full_text)
        cand = None
        if m:
            cand = m.group(1).strip()
        else:
            m2 = re.search(r"(在留資格[：:\s]*\n)([^\n\r]+)", full_text)
            if m2:
                cand = m2.group(2).strip()
        if cand:
            cand_norm = normalize_status(cand)
            if cand_norm in LEGAL_STATUSES:
                return cand_norm
    except:
        pass
    return normalize_status(current)

# ==========================
# 性別 归一化 / Sex normalization
# ==========================
def normalize_sex(raw: str, output="ja") -> str:
    """
    raw: 抽取到的性別（M/F/Male/Female/男/女/その他 等）
    output: 'ja' -> 日本語（男/女/その他/不詳）
            'en' -> 英語（Male/Female/Other/Unknown）
    """
    if not isinstance(raw, str) or not raw.strip():
        return "" if output == "ja" else "Unknown"

    s = raw.strip().lower()
    # よくある表記を正規化（記号や余分な語を除去）
    s = re.sub(r'[^a-z\u4e00-\u9fff]', '', s)  # 英字と漢字以外をざっくり除去（必要なら調整）

    male_tokens   = {"m", "male", "男", "男性"}
    female_tokens = {"f", "female", "女", "女性"}
    other_tokens  = {"その他", "他", "nonbinary", "nonbinarygender", "x"}

    if s in male_tokens:
        return "男" if output == "ja" else "Male"
    if s in female_tokens:
        return "女" if output == "ja" else "Female"
    if s in other_tokens:
        return "その他" if output == "ja" else "Other"

    if s.startswith("m"):
        return "男" if output == "ja" else "Male"
    if s.startswith("f"):
        return "女" if output == "ja" else "Female"

    return "不詳" if output == "ja" else "Unknown"

# ==============================
# 6) BBox 関連 / BBox utilities
# ==============================
def extract_bboxes(text: str):
    pattern = r"\[(\d+\.\d+),\s*(\d+\.\d+),\s*(\d+\.\d+),\s*(\d+\.\d+)\]"
    bboxes = []
    for m in re.finditer(pattern, text):
        coords = [float(m.group(i)) for i in range(1, 5)]
        if all(0.0 <= c <= 1.0 for c in coords):
            bboxes.append(coords)
    return bboxes

def normalize_to_pixels(bbox, w, h):
    # bbox: [x1, y1, x2, y2] (0~1)
    x1 = int(bbox[0] * w); y1 = int(bbox[1] * h)
    x2 = int(bbox[2] * w); y2 = int(bbox[3] * h)
    return (x1, y1, x2, y2)

# ==============================
# 7) 画像ループ / Process images
# ==============================
for idx, filename in enumerate(sorted(image_files)):
    print(f"\n📂 処理中: {filename} ({idx+1}/{len(image_files)})")
    image_path = os.path.join(image_folder, filename)
    image = Image.open(image_path).convert("RGB")
    original_image = image.copy()

    # 会話プロンプト：严格要求法定名称（禁止“学生”等一般語）
    conversation = [
        {
            "role": "user",
            "content": (
                "<image>\n"
                "この日本の在留カードの情報を抽出してください。日本の『在留資格』は"
                " 法定分類名で厳密に記載してください（例：留学、技術・人文知識・国際業務、家族滞在、特定活動、永住者、"
                " 日本人の配偶者等、定住者、技能、短期滞在、文化活動、研修、研究、介護、"
                " 高度専門職、高度専門職1号、高度専門職2号、企業内転勤、経営・管理、教授、教育、医療、興行、宗教、報道、芸術）。"
                " 「学生」などの一般語や翻訳語は使用不可。該当しない語は出力しないでください。\n"
                " 次のキーで日本語JSONを出力：在留カード番号, 氏名, 生年月日, 性別, 国籍, 住所, 在留資格, 在留期間, 許可年月日。\n"
                " 必ず JSON のみを返答してください。"
            )
        },
        {"role": "assistant", "content": ""}
    ]

    try:
        inputs = processor(
            conversations=conversation,
            images=[image],
            return_tensors="pt",
            force_batchify=True
        )
        inputs = inputs.to(model.device)
        # 画像テンソル dtype の明示
        inputs.images = inputs.images.to(dtype)

        model_inputs = {
            "input_ids": inputs.input_ids,
            "images": inputs.images,
        }
        if hasattr(inputs, "attention_mask") and inputs.attention_mask is not None:
            model_inputs["attention_mask"] = inputs.attention_mask
        if hasattr(inputs, "images_seq_mask") and inputs.images_seq_mask is not None:
            model_inputs["images_seq_mask"] = inputs.images_seq_mask
        if hasattr(inputs, "images_spatial_crop") and inputs.images_spatial_crop is not None:
            model_inputs["images_spatial_crop"] = inputs.images_spatial_crop

        with torch.no_grad():
            output = model.generate(
                **model_inputs,
                max_new_tokens=300,
                temperature=0.1,
                do_sample=False,
                pad_token_id=processor.tokenizer.eos_token_id,
                eos_token_id=processor.tokenizer.eos_token_id
            )

        raw_output = processor.tokenizer.decode(output[0], skip_special_tokens=False)

        # DeepSeek-VL2 形式：<|Assistant|> 以降が返答本体
        start_token = "<|Assistant|>"
        start_idx = raw_output.find(start_token)
        decoded_text = raw_output[start_idx + len(start_token):].strip() if start_idx != -1 else raw_output
        # <|...|> 系のspecial token除去
        cleaned_text = re.sub(r"<\|[^>]+\|>", "", decoded_text).strip()

        # ======= 信息抽取 =======
        extracted_info = extract_info(cleaned_text)

        # —— 对“在留資格”做强制归一化（学生→留学；优先用卡面“在留資格”行）
        if isinstance(extracted_info, dict):
            cur_status = extracted_info.get("在留資格", "")
            fixed_status = pick_status_from_text(cleaned_text, cur_status)
            if fixed_status:
                extracted_info["在留資格"] = fixed_status

            # ★ 性別を日本語に正規化（M/F/Male/Female → 男/女/その他/不詳）
            if extracted_info.get("性別"):
                extracted_info["性別"] = normalize_sex(extracted_info["性別"], output="ja")

        # ======= 可視化：BBox & 情報叠加 =======
        bboxes = extract_bboxes(raw_output)
        annotated_image = original_image.copy()
        draw = ImageDraw.Draw(annotated_image)
        width, height = annotated_image.size

        # フォント準備
        try:
            font = ImageFont.truetype("arial.ttf", 20)
        except Exception:
            font = ImageFont.load_default()

        # BBox 可視化
        for i, bbox in enumerate(bboxes):
            x1, y1, x2, y2 = normalize_to_pixels(bbox, width, height)
            draw.rectangle((x1, y1, x2, y2), outline="red", width=3)
            label_y = max(0, y1 - 25)
            draw.text((x1, label_y), f"領域{i+1}", fill="red", font=font)

        base = os.path.splitext(filename)[0]
        annotated_path = os.path.join(output_folder, f"{base}_bbox.jpg")
        annotated_image.save(annotated_path)

        # 情報叠加（卡号优先显示）
        if isinstance(extracted_info, dict):
            info_image = annotated_image.copy()
            draw2 = ImageDraw.Draw(info_image)
            y = 10
            keys = ["在留カード番号"] + [k for k in extracted_info.keys() if k != "在留カード番号"]
            for k in keys:
                v = extracted_info.get(k)
                if v:
                    draw2.text((10, y), f"{k}: {v}", fill="blue", font=font)
                    y += 30
            info_path = os.path.join(output_folder, f"{base}_info.jpg")
            info_image.save(info_path)

        # ======= CSV 書き込み =======
        with open(csv_path, mode="a", newline="", encoding="utf-8-sig") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            if isinstance(extracted_info, dict):
                row = {"ファイル名": filename}
                for k in fieldnames[1:]:
                    row[k] = extracted_info.get(k, "")
                writer.writerow(row)

        print(f"✅ 抽出成功: {filename}")

    except Exception as e:
        print(f"❌ 処理失敗: {filename} -> {e}")

print("\n🎉 すべての画像処理が完了しました / Done.")
print(f"📝 CSV: {csv_path}")
print(f"🖼 画像出力: {output_folder}")
