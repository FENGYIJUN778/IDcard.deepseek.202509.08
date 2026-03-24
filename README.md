# 🪪 Zairyu Card OCR System

> ⚠️ 本项目依赖 DeepSeek-VL2 模型，运行前请先完成模型下载与路径配置

---

## 📌 项目简介

本项目是一个基于多模态模型的在留卡（Residence Card）信息识别系统，实现从图像输入到结构化数据输出（CSV）的完整流程。

系统利用 DeepSeek-VL2 对在留卡图像进行 OCR 识别，并通过正则表达式提取关键字段信息，最终生成结构化数据，适用于证件信息录入与数据整理场景。

---

## 🚀 功能特点

* 📷 在留卡图像 OCR 识别（DeepSeek-VL2）
* 🧾 关键信息字段提取（正则表达式）
* 📊 自动生成 CSV 文件
* 🧪 单张图片推理耗时测试（time.py）
* ⚙️ 支持基础批量处理流程

---

## 🧠 技术流程

```text
输入图像
   ↓
DeepSeek-VL2（OCR识别）
   ↓
文本结果
   ↓
正则表达式解析字段
   ↓
CSV结构化输出
```

---

## 📂 项目结构

```text
zairyu-card-ocr/
│
├── extract/                # 字段提取模块（正则解析）
├── images/                 # 测试图片
│
├── main.py                 # 主程序入口
├── time.py                 # 单张图片推理时间测试脚本
├── requirements.txt        # 依赖环境
├── README.md               # 项目说明
│
├── card_info.json          # 示例识别结果（JSON格式）
├── results.csv             # 示例输出（CSV格式）
```

---

## 📥 模型下载说明（重要）

本项目未包含 DeepSeek-VL2 模型文件（体积较大，未上传至 GitHub）。

请用户自行下载模型，并在代码中配置对应路径后再运行项目。

### 使用步骤：

1. 下载 DeepSeek-VL2 模型
2. 将模型存放至本地目录
3. 修改代码中的模型路径（如 main.py / 相关推理文件）

示例（伪路径）：

```python
model_path = "your/path/to/deepseek-vl2"
```

⚠️ 若未正确配置模型路径，程序将无法正常运行

---

## ⚙️ 环境配置

### 1️⃣ 创建虚拟环境

```bash
python -m venv .venv
```

### 2️⃣ 激活环境

```bash
# Windows
.venv\Scripts\activate

# Mac/Linux
source .venv/bin/activate
```

### 3️⃣ 安装依赖

```bash
pip install -r requirements.txt
```

---

## ▶️ 使用方法

### 1️⃣ 运行主程序

```bash
python main.py
```

功能：

* 读取 images 文件夹中的图片
* 执行 OCR 识别
* 提取字段信息
* 生成 CSV 文件

---

### 2️⃣ 测试推理时间

```bash
python time.py
```

功能：

* 对单张图片进行推理
* 输出模型处理耗时
* 用于评估系统性能

---

## 📊 示例输出

### CSV（results.csv）

| 姓名 | 国籍 | 在留资格    | 有效期限       |
| -- | -- | ------- | ---------- |
| 张三 | 中国 | 技术·人文知识 | 2026-12-01 |

---

### JSON（card_info.json）

```json
{
  "name": "张三",
  "nationality": "中国",
  "status": "技术·人文知识",
  "expiry_date": "2026-12-01"
}
```

---

## ⚠️ 注意事项

* 本项目未包含 DeepSeek-VL2 模型文件，请自行下载并配置
* 输入图片需保证清晰度，否则可能影响识别结果
* 当前对日文平假名/片假名识别仍有优化空间

---

## 📈 项目亮点

* ✅ 基于多模态模型的 OCR 信息提取方案
* ✅ 使用正则表达式实现结构化字段解析
* ✅ 自动化生成 CSV 数据
* ✅ 包含推理时间测试模块（性能评估）

---

## 🧩 可扩展方向

* 引入目标检测模型（如 YOLO）提升区域定位精度
* 优化日文字符识别能力（平假名 / 片假名）
* 构建 Web 可视化界面（上传图片→返回结果）
* 支持批量处理与数据库集成

---

## 👨‍💻 作者

* FENG YIJUN

---

## ⭐ 欢迎 Star！
