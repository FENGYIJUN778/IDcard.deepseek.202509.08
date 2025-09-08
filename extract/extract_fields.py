import re
import torch
from PIL import Image, ImageDraw, ImageFont
from deepseek_vl2.models import DeepseekVLV2ForCausalLM, DeepseekVLV2Processor

class CardExtractor:
    def __init__(self, model_path="deepseek-ai/deepseek-vl2-tiny", device=None, dtype=None):
        """
        初始化卡片提取器
        
        参数:
            model_path: 模型路径 (默认使用 tiny 版本)
            device: 指定设备 (None 表示自动选择)
            dtype: 数据类型 (None 表示自动选择)
        """
        print("🔄 正在加载模型...")
        self.processor = DeepseekVLV2Processor.from_pretrained(model_path)
        
        # 设置EOS token
        if not hasattr(self.processor.tokenizer, 'eos_token') or self.processor.tokenizer.eos_token is None:
            self.processor.tokenizer.eos_token = "<|endoftext|>"
        if not hasattr(self.processor.tokenizer, 'eos_token_id') or self.processor.tokenizer.eos_token_id is None:
            self.processor.tokenizer.eos_token_id = 100010
        
        # 设备与精度
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if dtype is None:
            dtype = torch.float16 if device == "cuda" else torch.float32
            
        self.device = device
        self.dtype = dtype
        
        self.model = DeepseekVLV2ForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            device_map="auto"
        )
        print(f"✅ 模型加载成功，使用设备：{self.model.device}, 精度：{dtype}")

    def extract_fields(self, image, prompt=None):
        """
        从图像中提取字段信息
        
        参数:
            image: PIL.Image 对象
            prompt: 自定义提示词 (None 使用默认提示)
        
        返回:
            包含提取结果的字典
        """
        # 保存原始图像用于后续标注
        original_image = image.copy()
        
        # 默认提示词
        if prompt is None:
            prompt = "<image>\n请提取这张日本在留卡上的姓名、国籍、在留资格，并以JSON格式输出结果。"
        
        conversation = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": ""}
        ]
        
        # 处理输入
        inputs = self.processor(
            conversations=conversation,
            images=[image],
            return_tensors="pt",
            force_batchify=True
        )
        
        # 移动到设备
        inputs = inputs.to(self.model.device)
        
        # 转换图像数据类型
        if self.dtype == torch.float16:
            inputs.images = inputs.images.to(torch.float16)
        elif self.dtype == torch.float32:
            inputs.images = inputs.images.to(torch.float32)
            
        # 准备模型输入
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
        
        # 推理
        with torch.no_grad():
            output = self.model.generate(
                **model_inputs,
                max_new_tokens=300,
                temperature=0.1,
                do_sample=False,
                pad_token_id=self.processor.tokenizer.eos_token_id,
                eos_token_id=self.processor.tokenizer.eos_token_id
            )
        
        # 解码输出
        raw_output = self.processor.tokenizer.decode(output[0], skip_special_tokens=False)
        
        # 提取模型生成的内容
        start_token = "<|Assistant|>"
        start_idx = raw_output.find(start_token)
        if start_idx != -1:
            decoded_text = raw_output[start_idx + len(start_token):].strip()
        else:
            decoded_text = raw_output
        
        # 清理文本
        cleaned_text = re.sub(r'<\|[^>]+\|>', '', decoded_text).strip()
        
        # 提取信息
        extracted_info = self.extract_info(cleaned_text)
        
        # 提取边界框
        bboxes = self.extract_bboxes(raw_output)
        
        return {
            "raw_output": raw_output,
            "cleaned_text": cleaned_text,
            "extracted_info": extracted_info,
            "bboxes": bboxes,
            "original_image": original_image
        }
    
    def extract_info(self, text):
        """
        从文本中提取结构化信息
        
        参数:
            text: 模型输出的文本
            
        返回:
            字典形式的结构化信息 或 原始文本
        """
        result = {}
        # 尝试匹配JSON
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        if json_match:
            try:
                import json
                json_str = json_match.group(0)
                data = json.loads(json_str)
                return data
            except:
                pass
        
        patterns = {
            "姓名": r'(姓名|名字)[：:\s]*([^\n]+)',
            "国籍": r'(国籍|国家)[：:\s]*([^\n]+)',
            "在留资格": r'(在留资格|资格)[：:\s]*([^\n]+)'
        }
        
        for key, pattern in patterns.items():
            match = re.search(pattern, text)
            if match:
                value = match.group(2).strip()
                value = re.sub(r'\[\d+\.\d+,\s*\d+\.\d+,\s*\d+\.\d+,\s*\d+\.\d+\]', '', value)
                result[key] = value
        
        return result if result else text
    
    def extract_bboxes(self, text):
        """
        从文本中提取边界框坐标
        
        参数:
            text: 包含坐标标记的文本
            
        返回:
            边界框坐标列表 [[x_min, y_min, x_max, y_max], ...]
        """
        bboxes = []
        pattern = r'\[(\d+\.\d+),\s*(\d+\.\d+),\s*(\d+\.\d+),\s*(\d+\.\d+)\]'
        
        for match in re.finditer(pattern, text):
            try:
                coords = [float(match.group(i)) for i in range(1, 5)]
                # 验证坐标值在0-1之间
                if all(0.0 <= coord <= 1.0 for coord in coords):
                    bboxes.append(coords)
            except:
                continue
        
        return bboxes
    
    def draw_annotations(self, image, bboxes, info=None):
        """
        在图像上绘制边界框和信息
        
        参数:
            image: PIL.Image 对象
            bboxes: 边界框坐标列表
            info: 要显示的信息字典
            
        返回:
            标注后的PIL.Image对象
        """
        annotated_image = image.copy()
        draw = ImageDraw.Draw(annotated_image)
        width, height = annotated_image.size
        
        # 加载字体
        try:
            font = ImageFont.truetype("arial.ttf", 20)
        except:
            font = ImageFont.load_default()
        
        # 绘制边界框
        for i, bbox in enumerate(bboxes):
            # 转换坐标
            pixel_bbox = self.normalize_to_pixels(bbox, width, height)
            
            # 绘制边界框
            draw.rectangle(pixel_bbox, outline="red", width=3)
            
            # 添加标签
            label = f"区域{i+1}"
            draw.text((pixel_bbox[0], pixel_bbox[1] - 25), label, fill="red", font=font)
        
        # 绘制信息
        if info and isinstance(info, dict):
            y_offset = 10
            for key, value in info.items():
                text = f"{key}: {value}"
                draw.text((10, y_offset), text, fill="blue", font=font)
                y_offset += 30
        
        return annotated_image
    
    def normalize_to_pixels(self, bbox, img_width, img_height):
        """
        将归一化坐标转换为像素坐标
        
        参数:
            bbox: 归一化坐标 [x_min, y_min, x_max, y_max]
            img_width: 图像宽度
            img_height: 图像高度
            
        返回:
            像素坐标元组 (x_min, y_min, x_max, y_max)
        """
        x_min, y_min, x_max, y_max = bbox
        return (
            int(x_min * img_width),
            int(y_min * img_height),
            int(x_max * img_width),
            int(y_max * img_height)
        )