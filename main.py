import os
import re
import csv
import json
import torch
from PIL import Image, ImageDraw, ImageFont
from deepseek_vl2.models import DeepseekVLV2ForCausalLM, DeepseekVLV2Processor

# ===== 1. モデルロード =====
model_path = "deepseek-ai/deepseek-vl2-tiny"
print("🔄 モデルを読み込み中...")
processor = DeepseekVLV2Processor.from_pretrained(model_path)

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
print(f"✅ モデル読み込み成功：デバイス={model.device}、精度={dtype}")

# ===== 2. フォルダ準備 =====
image_folder = "images"
output_folder = "output"
os.makedirs(output_folder, exist_ok=True)

image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
if not image_files:
    print("❌ 画像が見つかりません")
    exit(1)

# ===== 3. CSV 初期化 =====
csv_path = "results.csv"
fieldnames = ["ファイル名", "氏名", "生年月日", "性別", "国籍", "住所", "在留資格", "在留期間", "許可年月日"]
with open(csv_path, mode='w', newline='', encoding='utf-8-sig') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

# ===== 4. 抽出関数 =====
def extract_info(text):
    result = {}
    json_match = re.search(r'\{.*\}', text, re.DOTALL)
    if json_match:
        try:
            data = json.loads(json_match.group(0))
            return data
        except:
            pass
    patterns = {
        "氏名": r'(氏名)[：:\s]*([^\n]+)',
        "生年月日": r'(生年月日)[：:\s]*([^\n]+)',
        "性別": r'(性別)[：:\s]*([^\n]+)',
        "国籍": r'(国籍)[：:\s]*([^\n]+)',
        "住所": r'(住所|住居地)[：:\s]*([^\n]+)',
        "在留資格": r'(在留資格)[：:\s]*([^\n]+)',
        "在留期間": r'(在留期間)[：:\s]*([^\n]+)',
        "許可年月日": r'(許可年月日)[：:\s]*([^\n]+)'
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, text)
        if match:
            value = match.group(2).strip()
            value = re.sub(r'\[\d+\.\d+,\s*\d+\.\d+,\s*\d+\.\d+,\s*\d+\.\d+\]', '', value)
            result[key] = value
    return result if result else text

def extract_bboxes(text):
    pattern = r'\[(\d+\.\d+),\s*(\d+\.\d+),\s*(\d+\.\d+),\s*(\d+\.\d+)\]'
    return [
        [float(m.group(i)) for i in range(1, 5)]
        for m in re.finditer(pattern, text)
        if all(0.0 <= float(m.group(i)) <= 1.0 for i in range(1, 5))
    ]

def normalize_to_pixels(bbox, w, h):
    return tuple(int(coord * dim) for coord, dim in zip(bbox, [w, h, w, h]))

# ===== 5. 各画像を処理 =====
for idx, filename in enumerate(image_files):
    print(f"\n📂 処理中: {filename} ({idx+1}/{len(image_files)})")
    image_path = os.path.join(image_folder, filename)
    image = Image.open(image_path).convert("RGB")
    original_image = image.copy()

    conversation = [
        {
            "role": "user",
            "content": "<image>\nこの日本の在留カードの情報を抽出してください。氏名、生年月日、性別、国籍、住所、在留資格、在留期間、許可年月日を JSON 形式で出力してください。"
        },
        {
            "role": "assistant",
            "content": ""
        }
    ]

    try:
        inputs = processor(
            conversations=conversation,
            images=[image],
            return_tensors="pt",
            force_batchify=True
        )
        inputs = inputs.to(model.device)
        inputs.images = inputs.images.to(dtype)

        model_inputs = {
            "input_ids": inputs.input_ids,
            "images": inputs.images,
        }
        if hasattr(inputs, 'attention_mask') and inputs.attention_mask is not None:
            model_inputs["attention_mask"] = inputs.attention_mask
        if hasattr(inputs, 'images_seq_mask') and inputs.images_seq_mask is not None:
            model_inputs["images_seq_mask"] = inputs.images_seq_mask
        if hasattr(inputs, 'images_spatial_crop') and inputs.images_spatial_crop is not None:
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
        start_token = "<|Assistant|>"
        start_idx = raw_output.find(start_token)
        decoded_text = raw_output[start_idx + len(start_token):].strip() if start_idx != -1 else raw_output
        cleaned_text = re.sub(r'<\|[^>]+\|>', '', decoded_text).strip()

        extracted_info = extract_info(cleaned_text)

        # ===== 描画 =====
        bboxes = extract_bboxes(raw_output)
        annotated_image = original_image.copy()
        draw = ImageDraw.Draw(annotated_image)
        width, height = annotated_image.size
        try:
            font = ImageFont.truetype("arial.ttf", 20)
        except:
            font = ImageFont.load_default()

        for i, bbox in enumerate(bboxes):
            pixel_box = normalize_to_pixels(bbox, width, height)
            draw.rectangle(pixel_box, outline="red", width=3)
            draw.text((pixel_box[0], pixel_box[1] - 25), f"領域{i+1}", fill="red", font=font)

        # 保存ファイル名
        base = os.path.splitext(filename)[0]
        annotated_path = os.path.join(output_folder, f"{base}_bbox.jpg")
        annotated_image.save(annotated_path)

        if isinstance(extracted_info, dict):
            info_image = annotated_image.copy()
            draw = ImageDraw.Draw(info_image)
            y = 10
            for k, v in extracted_info.items():
                draw.text((10, y), f"{k}: {v}", fill="blue", font=font)
                y += 30
            info_path = os.path.join(output_folder, f"{base}_info.jpg")
            info_image.save(info_path)

        # ===== 書き込み =====
        with open(csv_path, mode='a', newline='', encoding='utf-8-sig') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            if isinstance(extracted_info, dict):
                row = {"ファイル名": filename}
                for k in fieldnames[1:]:
                    row[k] = extracted_info.get(k, "")
                writer.writerow(row)

        print(f"✅ 抽出成功: {filename}")

    except Exception as e:
        print(f"❌ 処理失敗: {filename} -> {str(e)}")

print("\n🎉 すべての画像処理が完了しました")