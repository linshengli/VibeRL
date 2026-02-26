# RAG 接入与资料导入指南

本项目已接入本地 RAG（检索增强），支持：
- 文档上传：`PDF / PPT(PPTX) / TXT / JSON`
- 聊天记录导入：`Telegram / WhatsApp / Discord / Slack / 通用第三方聊天`
- 查询时自动检索并注入上下文

实现位置：
- 存储与检索：`src/rag/store.py`
- 文件解析：`src/rag/ingest.py`
- Web API：`src/web/app.py`

## 1. 依赖

```bash
pip install -r requirements.txt
```

关键依赖：
- `pypdf`：PDF 解析
- `python-pptx`：PPT/PPTX 解析

## 2. 启动 Web

```bash
python src/web/app.py --host 0.0.0.0 --port 5000 --model rule-based
```

可选 RAG 参数：
- `--rag-file`：RAG 存储路径（默认 `data/rag_store.json`）
- `--rag-max-docs`：最多文档数
- `--rag-max-chunks`：最多 chunk 数

## 3. 前端使用

页面左栏有 `RAG 资料库`：
1. 选择导入类型（文档 / Telegram / 第三方聊天 / WhatsApp / Discord / Slack）
2. 选择文件并上传
3. 资料会显示在 RAG 列表中

聊天输入区可勾选/取消 `RAG` 开关，控制本轮是否启用检索增强。

## 4. API

### 4.1 上传资料

`POST /api/rag/upload`（multipart/form-data）
- `file`: 文件
- `import_type`: `document | telegram | chat | whatsapp | discord | slack`

### 4.2 列出文档

`GET /api/rag/docs?limit=100`

### 4.3 检索测试

`POST /api/rag/query`

请求示例：
```json
{"query":"茅台 风险", "top_k":5}
```

### 4.4 聊天时启用 RAG

`POST /api/chat` / `POST /api/chat/stream`

请求体新增字段：
```json
{"query":"...", "model":"rule-based", "use_rag":true}
```

返回记录中会带：
- `rag.enabled`
- `rag.hits`（命中文档片段）

## 5. Telegram 导入建议

推荐使用 Telegram 官方导出 JSON，再用 `import_type=telegram` 上传。
解析字段：
- `messages[].from`
- `messages[].date`
- `messages[].text`（支持字符串和富文本数组）

## 6. WhatsApp / Discord / Slack 适配说明

- WhatsApp
  - 文本导出：`[date] name: message` 或 `date - name: message`
  - JSON 导出：`messages[]` 结构
- Discord
  - JSON：`messages[].author` + `messages[].content`
- Slack
  - JSON：`text` + `user/user_profile/username` + `ts`
