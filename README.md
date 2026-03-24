# 🪪 Zairyu Card OCR System

## 📌 项目简介

本项目是一个基于多模态模型的在留卡（Residence Card）信息识别系统，实现从图像输入到结构化数据输出（CSV）的完整流程。

系统利用视觉语言模型（DeepSeek-VL2）对在留卡图像进行OCR识别，并通过正则表达式提取关键字段，实现自动化信息整理。

---

## 🚀 功能特点

* 📷 在留卡图像OCR识别（DeepSeek-VL2）
* 🧾 关键信息字段提取（正则表达式）
* 📊 自动生成 CSV 文件
* 🔁 支持多张图片批量处理
* ⚙️ 结构化 Prompt 提高识别稳定性

---

## 🧠 技术流程

```
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

```
zairyu-card-ocr/
│
├── main.py                 # 主程序入口
├── deepseek1127.py         # 模型推理逻辑
├── extract/                # 字段提取模块
├── images/                 # 测试图片
├── results.csv             # 输出示例
├── annotated_card.jpg      # 示例图片
├── requirements.txt        # 依赖环境
└── README.md
```

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

```bash
python main.py
```

运行后将自动完成：

* OCR识别
* 字段提取
* CSV文件生成

---

## 📊 示例输出

| 姓名 | 国籍 | 在留资格    | 有效期限       |
| -- | -- | ------- | ---------- |
| 张三 | 中国 | 技术·人文知识 | 2026-12-01 |

---

## ⚠️ 注意事项

* 本项目未包含模型文件（DeepSeek-VL2）
* 请自行下载并配置模型路径
* 输入图片需保证清晰度，否则可能影响识别结果

---

## 📈 项目亮点

* ✅ 基于多模态模型的OCR信息提取方案
* ✅ 使用结构化 Prompt 提高识别准确率
* ✅ 自动化结构化数据输出（CSV）
* ✅ 具备实际工程应用价值

---

## 🧩 可扩展方向

* 增加目标检测（YOLO）提高定位精度
* 优化日语字符识别（平假名/片假名）
* 构建 Web 可视化界面
* 支持更多证件类型识别

---

## 👨‍💻 作者

* FYJ

---

## ⭐ 欢迎 Star！
