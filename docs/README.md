# AI Town Lite 文档

## 1. 项目概览
本项目是一个轻量 AI Town 原型：
- 前端：Vite + React（现代/像素主题切换）
- 后端：FastAPI + SQLite
- LLM：OpenAI API（支持自定义 Base URL）
- 事件溯源：世界状态更新写事件，快照版本化

目录结构：
- `apps/api`：后端服务
- `apps/web`：前端页面
- `data/app.db`：SQLite 数据文件
- `scripts/k6.js`：压力测试脚本

## 2. 快速运行
### 2.1 后端
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r apps/api/requirements.txt
export OPENAI_API_KEY=YOUR_KEY
export OPENAI_BASE_URL=YOUR_BASE_URL  # optional
export SQLITE_PATH=./data/app.db
uvicorn apps.api.main:app --reload
```

### 2.2 前端
```bash
cd apps/web
npm install
npm run dev
```

### 2.3 环境变量
- `.env.example`：后端示例
- `apps/web/.env.example`：前端 API Base
 - LLM 可按角色拆分配置（NPC/User/Classifier）。若未填写，则自动回退到通用 `OPENAI_API_KEY / OPENAI_BASE_URL`。

## 3. 核心功能与数据流
### 3.1 World Model
- 可一键生成世界模型
- 可加载指定 `world_id`
- 更新世界模型时：
  - 事务内写入 `world_events`
  - 同步更新 `world_models.snapshot`
  - 校验 `expected_version`，避免并发覆盖

### 3.2 NPC / 用户模拟
- NPC 知晓 world model 全部内容
- 用户只基于自己的设定与上下文
- 前端保存的 NPC/用户设定会进入 LLM context
- 自动聊天模式：一句 / 一步 / 直到欲望 < 5
- NPC 新增 `volcengine` 模式（SP 结构）：
  - 在 NPC 设定 JSON 中设置：
    - `mode`: `"volcengine"`
    - `role_name`: 角色名称
     - `user_name`: 用户称呼
    - `init_role_sp`: 角色初始设定
    - `user_info`: 用户画像信息
    - `golden_sp`: 角色行为准则
   - 生成的 SP 格式：
     1. `{init_role_sp}`
     2. `{user_info}`
     3. `{golden_sp}`
     4. `现在请扮演{role_name}，{role_name}正在和{user_name}对话。`
   - 会转成系统提示词并驱动 NPC 回复

### 3.3 情绪标签
- 规则优先（关键词）
- 置信度 < 0.6 时调用 LLM 分类
- 记录 `emotion_label + confidence + method`

### 3.4 多样性指标
- Lexical: TTR / MTLD
- Topic: unique count + entropy
- 面板展示：情绪分布、欲望趋势、词汇多样性

## 4. 数据库表
- `world_models`: id, snapshot_json, version
- `world_events`: event_type, payload_json, version
- `chats`: chat id, world id
- `messages`: content, emotion, desire, topic
- `actor_profiles`: NPC/用户设定
- `memory_states`: 最近 60 条记忆

## 5. API 说明
### 5.1 World
- `POST /worlds/generate`
- `POST /worlds`
- `GET /worlds/{id}`
- `PATCH /worlds/{id}`（必须带 `expected_version`）
- `GET /worlds/{id}/events`

### 5.2 Chat
- `POST /chats`
- `GET /chats/{id}`
- `POST /chats/{id}/message`
- `POST /chats/{id}/auto/one`
- `POST /chats/{id}/auto/step`
- `POST /chats/{id}/auto/until`
- `GET /chats/{id}/stats`

### 5.3 Actor
- `POST /actor`
- `PATCH /actor/{id}`

### 5.4 Config
- `GET /config`

## 6. 压力测试
使用 k6（轻量）
```bash
k6 run scripts/k6.js
```
指标关注：P95 latency、错误率、吞吐量。

## 7. 需求复盘
已实现：
- World model 状态存储 + 事件溯源
- 一键生成 + 手动更新
- NPC/用户 LLM 对话与自动聊天模式
- 情绪标签自动生成（规则 + LLM）
- 统计面板（欲望趋势、情绪分布、多样性）
- 版本控制 + 事务保证一致性
- 响应式图表（SVG）

未实现/可增强：
- 更高级的记忆摘要与长期记忆检索
- 多人/多角色并发对话
- 复杂剧情引擎（显式 plot hook 触发器）

## 8. 可扩展性设计
### 8.1 记忆扩展
- 在 `memory.py` 中增加新策略（如摘要/embedding）
- 在 `update_memory` 中切换 mode

### 8.2 模型扩展
- `llm.py` 中增加新模型适配或批量代理
- `.env` 中新增配置即可切换

### 8.3 事件扩展
- 为 `world_events` 引入更多 `event_type`
- 可添加回溯接口（按版本重建快照）

### 8.4 指标扩展
- 在 `stats.py` 增加新指标（例如一致性评分、角色偏差）
- 前端面板新增可视化组件

### 8.5 存储扩展
- SQLite 可替换为 Postgres（修改 `db.py` 连接字符串）

## 9. 用户视角注意点
- 更新 World 需保持 JSON 合法
- 先生成/加载 World，再创建 Chat
- 输入内容即可自动标注情绪与话题
