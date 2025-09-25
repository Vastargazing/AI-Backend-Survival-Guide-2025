# 🤖 AI Backend Survival Guide 2025

<img width="624" height="624" alt="Image" src="https://github.com/user-attachments/assets/df83e7cf-8f98-41a3-9076-68a795b25a9d" />

Это практическая шпаргалка для инженеров, работающих с продакшн-системами на базе LLM, RAG и ML.  
Документ охватывает архитектуру, инструменты, безопасность, мониторинг и реальные паттерны, которые применяются в банках, финтехе, e-commerce и AI-продуктах.


## 📚 Часть 1: AI Backend Fundamentals
- 🔍 Vector Databases (pgvector, Pinecone, Weaviate)
- 📊 Embeddings: что это и как работает
- 🎯 Similarity Search и Distance Metrics
- ⚡ Vector Indexing (HNSW, IVF)

## 🌐 LLM Integration
- 🤖 LLM APIs: OpenAI, Anthropic, Local Models
- 💰 Token Management и Cost Optimization
- 🔄 Prompt Engineering для Production
- ⚡ Streaming Responses
- 🛡️ Rate Limiting для LLM APIs

## 🏗️ RAG Architecture
- 🔍 RAG Pipeline: Ingestion → Retrieval → Generation
- 📝 Document Processing (PDF, DOC, HTML)
- 🧹 Text Chunking Strategies
- 🎯 Hybrid Search (Vector + BM25)
- 📊 RAG Evaluation Metrics

## ⚙️ Часть 2: Production AI Systems
- ⚡ AI Model Caching Strategies
- 🔄 Async Processing для AI Tasks
- 📊 GPU Resource Management
- ⚖️ Load Balancing для AI Workloads
- 🎯 Model Serving (ONNX, TensorRT)

## 🗄️ AI Data Architecture
- 📊 Feature Stores (Feast, Tecton)
- ⚡ Real-time vs Batch ML Pipelines
- 🔄 Data Versioning для ML
- 📈 Vector Database Scaling

## 🔐 AI Security & Ethics
- 🛡️ Prompt Injection Protection
- 🔒 PII Detection в AI Responses
- ⚖️ AI Content Moderation
- 📋 AI Compliance (GDPR, AI Act)

## 🛠️ Часть 3: AI Backend Tools & Stack

### 🔧 Essential Libraries
- 🐍 LangChain vs LlamaIndex vs Custom
- 📊 Transformers, Sentence-Transformers
- ⚡ FastAPI + WebSockets для AI
- 📈 MLflow, Weights & Biases

### 🏗️ Infrastructure
- 🐳 Docker для ML Models
- ☸️ Kubernetes + GPU Scheduling
- ☁️ Cloud AI Services (AWS Bedrock, Azure OpenAI)
- 📊 Model Registry (MLflow, DVC)

## 📈 Monitoring & Observability
- 📊 AI Model Drift Detection
- 💰 Cost Monitoring для LLM APIs
- 🎯 Quality Metrics (Hallucination Detection)
- 🔍 AI Request Tracing

## 🏢 Часть 4: Real-World AI Patterns

### 🎯 Production AI Patterns
- 🔄 Circuit Breaker для AI Services
- ⚡ AI Response Caching
- 🎯 Fallback Strategies
- 📊 A/B Testing AI Models

## 🏗️ System Design для AI
- 🤖 Спроектируй ChatGPT-like сервис
- 🔍 AI-powered поиск для e-commerce
- 📊 ML Platform для 40+ команд

## 💼 Часть 5: AI Backend Interview Prep

### 🎯 AI Technical Questions
- 🧠 Объясни как работает RAG
- 💰 Как оптимизировать затраты на LLM
- 🔍 Vector search vs SQL search
- ⚡ Bottlenecks в AI системах

---

## 🔍 Vector Databases (pgvector, Pinecone, Weaviate) ⭐⭐⭐

**Что это:**  
Базы данных, заточенные под хранение и быстрый поиск векторов (embedding'ов).  
Позволяют делать similarity search по миллионам объектов.

**Примеры:**
- **pgvector** — расширение для PostgreSQL, удобно для гибридных систем  
- **Pinecone** — облачный сервис, масштабируется, используется в проде  
- **Weaviate** — open-source, поддерживает GraphQL, metadata + vector search

🏢 **Используют:**  
- **OpenAI** — pgvector для RAG  
- **Notion AI** — Pinecone для поиска по заметкам  
- **Aleph Alpha** — Weaviate для семантического поиска

---

## 📊 Embeddings: что это и как работает ⭐⭐⭐

**Что это:**  
Числовое представление смысла текста, изображения, аудио и т.д.  
Например: `"банковский перевод"` → `[0.12, -0.87, ..., 0.44]`

**Где берём:**
- `text-embedding-3` от OpenAI  
- `sentence-transformers` от HuggingFace  
- `cohere.embed()` — быстрые и дешёвые

**Примеры использования:**
- Поиск похожих документов  
- Кластеризация пользователей  
- Ранжирование ответов в чатах

🏢 **Используют:**  
- **You.com** — для поиска по embedding'ам  
- **Tinkoff AI** — для семантического поиска по транзакциям  
- **GitHub Copilot** — embeddings кода для подсказок

---

## 🎯 Similarity Search и Distance Metrics ⭐⭐

**Что это:**  
Поиск ближайших векторов по метрике расстояния.

**Метрики:**
- **Cosine similarity** — угол между векторами  
- **Euclidean** — обычное расстояние  
- **Dot product** — скалярное произведение

**Примеры:**
- Поиск похожих товаров  
- Рекомендации фильмов  
- Семантический поиск по базе знаний

🏢 **Используют:**  
- **Amazon** — рекомендации  
- **Netflix** — похожие фильмы  
- **VK AI** — поиск по embedding'ам постов

---

## ⚡ Vector Indexing (HNSW, IVF) ⭐⭐

**Что это:**  
Алгоритмы для быстрого поиска ближайших векторов.

**Типы:**
- **HNSW (Hierarchical Navigable Small World):** граф, быстрый и точный  
- **IVF (Inverted File Index):** кластеризация + поиск внутри кластера  
- **FAISS** — библиотека от Meta, поддерживает оба

**Примеры:**
- Поиск по миллионам embedding'ов  
- Ранжирование ответов в чатах  
- Семантический поиск в документах

🏢 **Используют:**  
- **Meta** — FAISS для поиска по embedding'ам  
- **DeepL** — HNSW для перевода  
- **Tinkoff AI** — IVF для быстрого поиска по транзакциям

---

## 🤖 LLM APIs: OpenAI, Anthropic, Local Models ⭐⭐⭐

**Что это:**  
Интерфейсы для общения с языковыми моделями — через HTTP-запросы.

**Примеры:**
- **OpenAI** — `gpt-4`, `text-embedding-3`, `function calling`  
- **Anthropic** — `Claude 2`, `Claude 3` — безопасный и длинный контекст  
- **Local Models** — `LLaMA`, `Mistral`, `GGUF` — для on-prem решений

🏢 **Используют:**  
- **Notion AI** — OpenAI  
- **Slack** — Anthropic  
- **Tinkoff AI** — локальные модели для приватных данных

---

## 💰 Token Management и Cost Optimization ⭐⭐⭐

**Что это:**  
LLM считает токены → чем больше токенов, тем дороже.  
Важно контролировать длину промпта, контекста и частоту вызовов.

**Практики:**
- ✅ Сжатие промптов (`summarize`, `strip`)  
- ✅ Кэширование ответов  
- ✅ Использование `embedding` вместо LLM, где можно  
- ✅ Мониторинг через `OpenAI Usage API`

🏢 **Используют:**  
- **Zapier AI** — лимитирует токены на пользователя  
- **Tinkoff AI** — кэширует ответы и режет контекст  
- **Miro AI** — использует embeddings вместо LLM для поиска

---

## 🔄 Prompt Engineering для Production ⭐⭐⭐

**Что это:**  
Искусство писать промпты, которые стабильно работают в проде.

**Практики:**
- ✅ Chain-of-thought → пошаговое мышление  
- ✅ Few-shot → примеры в промпте  
- ✅ Role-based → "Ты — банковский аналитик…"  
- ✅ Промпт как код → версионирование, тесты, A/B

🏢 **Используют:**  
- **GitHub Copilot** — few-shot + role  
- **Tinkoff AI** — шаблоны промптов для разных сценариев  
- **Replit AI** — chain-of-thought для генерации кода

---

## ⚡ Streaming Responses ⭐⭐

**Что это:**  
Ответ приходит по кускам (`stream`), а не весь сразу.  
Улучшает UX — пользователь видит, как "печатает" модель.

**Техники:**
- ✅ SSE (`Server-Sent Events`)  
- ✅ WebSocket  
- ✅ `stream=True` в OpenAI API

🏢 **Используют:**  
- **ChatGPT** — стримит ответ  
- **Tinkoff AI** — стримит подсказки в чатах  
- **Miro AI** — стримит генерацию текста

---

## 🛡️ Rate Limiting для LLM APIs ⭐⭐

**Что это:**  
Ограничение количества запросов → защита от перегрузки и злоупотреблений.

**Практики:**
- ✅ Лимит по IP / токену / пользователю  
- ✅ `429 Too Many Requests` + `Retry-After`  
- ✅ Тарифы: free, pro, enterprise

🏢 **Используют:**  
- **OpenAI** — лимиты по API-ключу  
- **Anthropic** — rate limit по модели и аккаунту  
- **Tinkoff AI** — лимитирует токены и RPS на пользователя

---

## 🏗️ RAG Architecture

---

## 🔍 RAG Pipeline: Ingestion → Retrieval → Generation ⭐⭐⭐

**Что это:**  
Классическая схема для Retrieval-Augmented Generation:  
1. **Ingestion** — загрузка и обработка документов  
2. **Retrieval** — поиск релевантных фрагментов  
3. **Generation** — генерация ответа с учётом найденного контекста

🏢 **Используют:**  
- **OpenAI ChatGPT + Browsing**  
- **Tinkoff AI** — поиск по базе знаний  
- **You.com** — поиск + генерация ответа

---

## 📝 Document Processing (PDF, DOC, HTML) ⭐⭐

**Что это:**  
Извлечение текста из разных форматов для последующего chunk'инга и индексации.

**Инструменты:**
- `pdfminer`, `PyMuPDF` — для PDF  
- `python-docx` — для DOC/DOCX  
- `BeautifulSoup` — для HTML

🏢 **Используют:**  
- **Notion AI** — парсинг заметок  
- **Tinkoff AI** — обработка договоров  
- **Aleph Alpha** — ingestion корпоративных документов

---

## 🧹 Text Chunking Strategies ⭐⭐⭐

**Что это:**  
Разделение текста на куски для embedding'а и поиска.

**Стратегии:**
- ✅ Fixed-size chunks (например, 500 токенов)  
- ✅ Semantic splitting (по заголовкам, абзацам)  
- ✅ Overlap (например, 20% пересечения)

🏢 **Используют:**  
- **LangChain** — `RecursiveCharacterTextSplitter`  
- **Tinkoff AI** — chunk'инг по смыслу  
- **ChatGPT RAG** — с overlap для контекста

---

## 🎯 Hybrid Search (Vector + BM25) ⭐⭐

**Что это:**  
Комбинируем семантический поиск (векторы) и keyword-поиск (BM25).

**Зачем:**
- ✅ Векторы — смысл  
- ✅ BM25 — точные ключевые слова  
- ✅ Гибкость и точность

🏢 **Используют:**  
- **You.com** — гибридный поиск  
- **Tinkoff AI** — BM25 + pgvector  
- **Weaviate** — встроенная поддержка hybrid search

---

## 📊 RAG Evaluation Metrics ⭐⭐

**Что это:**  
Метрики для оценки качества RAG-систем.

**Примеры:**
- **Precision@k / Recall@k** — насколько релевантны найденные фрагменты  
- **Faithfulness** — модель не "придумывает"  
- **Answer correctness** — совпадение с ground truth  
- **Latency** — скорость ответа

🏢 **Используют:**  
- **OpenAI evals** — для тестов моделей  
- **Tinkoff AI** — ручная и автоматическая оценка  
- **Meta** — eval pipeline для RAG моделей

---

## ⚙️ Часть 2: Production AI Systems  
🚀 Performance & Scalability

---

## ⚡ AI Model Caching Strategies ⭐⭐⭐

**Что это:**  
Кэшируем ответы LLM или embedding'и, чтобы не вызывать модель повторно.

**Стратегии:**
- ✅ Кэш по промпту (`prompt_hash`)  
- ✅ Кэш embedding'ов (`text → vector`)  
- ✅ Кэш на уровне retrieval (RAG)

**Инструменты:**
- `Redis`, `Memcached`, `DuckDB`, `SQLite`  
- `LangChain` — встроенный кэш

🏢 **Используют:**  
- **Tinkoff AI** — кэширует ответы и embedding'и  
- **Notion AI** — кэш по заметкам  
- **OpenAI** — кэширует retrieval в RAG

---

## 🔄 Async Processing для AI Tasks ⭐⭐⭐

**Что это:**  
Асинхронная обработка тяжёлых задач — генерация, embedding, анализ.

**Техники:**
- ✅ `Celery`, `FastAPI + asyncio`, `Kafka`  
- ✅ Очереди задач → worker'ы  
- ✅ Webhook / polling для ответа

🏢 **Используют:**  
- **Tinkoff AI** — Celery + Redis  
- **Slack AI** — async генерация сообщений  
- **GitHub Copilot** — фоновая генерация кода

---

## 📊 GPU Resource Management ⭐⭐

**Что это:**  
Управление доступом к GPU — чтобы не перегружать и не простаивали.

**Техники:**
- ✅ `NVIDIA MIG` — делим GPU на части  
- ✅ `Kubernetes + GPU scheduler`  
- ✅ Мониторинг: `nvidia-smi`, `Prometheus`, `Grafana`

🏢 **Используют:**  
- **Tinkoff AI** — MIG + мониторинг  
- **OpenAI** — кластеризация GPU  
- **HuggingFace Spaces** — ограничение по GPU usage

---

## ⚖️ Load Balancing для AI Workloads ⭐⭐

**Что это:**  
Распределение запросов между моделями/нодами.

**Техники:**
- ✅ `Round Robin`, `Least Loaded`, `Token-aware`  
- ✅ Балансировка по типу задачи (LLM, embedding, RAG)

🏢 **Используют:**  
- **Tinkoff AI** — балансировка по типу модели  
- **Anthropic** — распределение по регионам  
- **OpenRouter** — маршрутизация между LLM-провайдерами

---

## 🎯 Model Serving (ONNX, TensorRT) ⭐

**Что это:**  
Оптимизация и деплой моделей для inference.

**Инструменты:**
- ✅ `ONNX` — универсальный формат  
- ✅ `TensorRT` — ускорение на NVIDIA GPU  
- ✅ `TorchServe`, `Triton`, `vLLM`

🏢 **Используют:**  
- **Tinkoff AI** — ONNX для inference  
- **Meta** — TensorRT для LLaMA  
- **NVIDIA** — Triton для прод-сервинга

---

## 🗄️ AI Data Architecture

---

## 📊 Feature Stores (Feast, Tecton) ⭐⭐

**Что это:**  
Централизованное хранилище признаков для ML-моделей — с версионированием, доступом и обновлением.

**Инструменты:**
- **Feast** — open-source, интеграция с Redis, BigQuery  
- **Tecton** — enterprise-решение, real-time + batch

**Зачем:**
- ✅ Повторяемость экспериментов  
- ✅ Единый источник признаков  
- ✅ Онлайн/оффлайн синхронизация

🏢 **Используют:**  
- **Robinhood** — Tecton для real-time фич  
- **Tinkoff AI** — Feast + PostgreSQL  
- **Airbnb** — собственный feature store

---

## ⚡ Real-time vs Batch ML Pipelines ⭐⭐⭐

**Что это:**  
ML-пайплайны бывают двух типов:  
- **Batch** — обучение и инференс по расписанию  
- **Real-time** — инференс на лету, при событии

**Инструменты:**
- `Airflow`, `Spark`, `dbt` — для batch  
- `Kafka`, `Flink`, `FastAPI` — для real-time

🏢 **Используют:**  
- **Tinkoff AI** — real-time скоринг транзакций  
- **Netflix** — batch для рекомендаций  
- **Uber** — real-time ML для ETA и surge pricing

---

## 🔄 Data Versioning для ML ⭐⭐

**Что это:**  
Храним версии данных, признаков и моделей для воспроизводимости.

**Инструменты:**
- `DVC`, `LakeFS`, `MLflow`, `Weights & Biases`

**Зачем:**
- ✅ Повторяемость экспериментов  
- ✅ Аудит и откат  
- ✅ Сравнение моделей

🏢 **Используют:**  
- **Tinkoff AI** — DVC + Git  
- **DeepMind** — MLflow для трекинга  
- **Yandex Cloud** — LakeFS для версионирования данных

---

## 📈 Vector Database Scaling ⭐⭐

**Что это:**  
Масштабирование векторных БД для поиска по embedding'ам.

**Техники:**
- ✅ Шардирование по namespace  
- ✅ Индексация (HNSW, IVF)  
- ✅ Кэширование популярных запросов  
- ✅ Параллельный retrieval

**Инструменты:**
- `Pinecone`, `Weaviate`, `pgvector`, `FAISS`

🏢 **Используют:**  
- **Tinkoff AI** — pgvector + шардирование  
- **Notion AI** — Pinecone с миллионами embedding'ов  
- **You.com** — Weaviate + hybrid search

---
## 🔐 AI Security & Ethics

---

## 🛡️ Prompt Injection Protection ⭐⭐⭐

**Что это:**  
Атака, при которой злоумышленник внедряет инструкции в промпт, чтобы изменить поведение модели.

**Защита:**
- ✅ Экранирование пользовательского ввода  
- ✅ Разделение системного и пользовательского промпта  
- ✅ Валидация output'а  
- ✅ Использование Guardrails / Rebuff / LlamaGuard

🏢 **Используют:**  
- **OpenAI** — фильтрует ввод и output  
- **Tinkoff AI** — защищает промпты в чатах  
- **Anthropic** — обучает модели на безопасное поведение

---

## 🔒 PII Detection в AI Responses ⭐⭐⭐

**Что это:**  
Обнаружение персональных данных (PII) в ответах модели — чтобы не утекли email, телефоны, карты и т.д.

**Инструменты:**
- ✅ `Presidio`, `PII Scanner`, `Regex + ML`  
- ✅ Post-processing перед выводом  
- ✅ Маскирование (`****`, `last 4 digits`)

🏢 **Используют:**  
- **Tinkoff AI** — маскирует карты и телефоны  
- **Meta** — фильтрует PII в LLaMA  
- **AWS Comprehend** — PII detection в текстах

---

## ⚖️ AI Content Moderation ⭐⭐

**Что это:**  
Фильтрация токсичного, вредного или запрещённого контента, сгенерированного моделью.

**Методы:**
- ✅ Классификаторы (`toxicity`, `hate`, `violence`)  
- ✅ Модерация output'а  
- ✅ RLHF — обучение модели на безопасное поведение

🏢 **Используют:**  
- **YouTube AI** — фильтрация комментариев  
- **OpenAI** — moderation endpoint  
- **Tinkoff AI** — фильтрация запросов и ответов

---

## 📋 AI Compliance (GDPR, AI Act) ⭐

**Что это:**  
Соблюдение законов по защите данных и прозрачности AI.

**Требования:**
- ✅ Объяснимость решений  
- ✅ Удаление данных по запросу  
- ✅ Логирование и аудит  
- ✅ Согласие пользователя

🏢 **Используют:**  
- **Tinkoff AI** — соблюдение ФЗ-152 и GDPR  
- **HuggingFace** — AI Act compliance  
- **Microsoft Copilot** — прозрачность и контроль данных

---

## 🛠️ Часть 3: AI Backend Tools & Stack  
🔧 Essential Libraries

---

## 🐍 LangChain vs LlamaIndex vs Custom ⭐⭐⭐

**Что это:**  
Фреймворки для построения LLM-приложений — с цепочками, памятью, RAG и агентами.

**Сравнение:**
- **LangChain** — цепочки, агенты, memory, tools  
- **LlamaIndex** — ingestion, retrieval, RAG  
- **Custom** — максимум гибкости, минимум зависимости

🏢 **Используют:**  
- **Tinkoff AI** — кастомный стек  
- **Replit AI** — LangChain  
- **Notion AI** — LlamaIndex для поиска по заметкам

---

## 📊 Transformers, Sentence-Transformers ⭐⭐

**Что это:**  
Библиотеки для работы с LLM и embedding'ами.

**Инструменты:**
- `transformers` — HuggingFace, LLM, tokenizer  
- `sentence-transformers` — быстрые embedding'и, semantic search

🏢 **Используют:**  
- **Tinkoff AI** — sentence-transformers для поиска  
- **HuggingFace Spaces** — transformers для inference  
- **You.com** — semantic search по embedding'ам

---

## ⚡ FastAPI + WebSockets для AI ⭐⭐⭐

**Что это:**  
FastAPI — быстрый backend-фреймворк, WebSocket — для стриминга ответов.

**Зачем:**
- ✅ Быстрый REST API  
- ✅ SSE / WebSocket для real-time  
- ✅ Async обработка AI задач

🏢 **Используют:**  
- **Tinkoff AI** — FastAPI + WebSocket для чатов  
- **Replit AI** — FastAPI для LLM-интеграции  
- **Mistral** — FastAPI для inference

---

## 📈 MLflow, Weights & Biases ⭐⭐

**Что это:**  
Инструменты для трекинга экспериментов, метрик и моделей.

**Функции:**
- ✅ Логирование метрик  
- ✅ Визуализация  
- ✅ Model registry  
- ✅ Автоматизация экспериментов

🏢 **Используют:**  
- **Tinkoff AI** — MLflow + DVC  
- **DeepMind** — Weights & Biases  
- **OpenAI evals** — логирование качества моделей

---

## 🏗️ Infrastructure

---

## 🐳 Docker для ML Models ⭐⭐⭐

**Что это:**  
Контейнеризация моделей — для reproducibility и деплоя.

**Зачем:**
- ✅ Изоляция зависимостей  
- ✅ Быстрый rollout  
- ✅ Совместимость с CI/CD

🏢 **Используют:**  
- **Tinkoff AI** — Docker + TorchServe  
- **Meta** — Docker + Triton  
- **HuggingFace** — Docker образы моделей

---

## ☸️ Kubernetes + GPU Scheduling ⭐⭐

**Что это:**  
Оркестрация моделей и задач на кластере с GPU.

**Техники:**
- ✅ `nvidia.com/gpu` в pod'ах  
- ✅ `MIG` для деления GPU  
- ✅ `KubeFlow`, `Ray`, `Argo` — для ML pipeline'ов

🏢 **Используют:**  
- **Tinkoff AI** — K8s + GPU scheduler  
- **Anthropic** — кластеризация inference  
- **OpenAI** — распределение по регионам

---

## ☁️ Cloud AI Services (AWS Bedrock, Azure OpenAI) ⭐⭐

**Что это:**  
Облачные сервисы для доступа к LLM, embedding'ам и inference.

**Примеры:**
- **AWS Bedrock** — Anthropic, Mistral, Cohere  
- **Azure OpenAI** — GPT-4, embeddings, DALL·E  
- **GCP Vertex AI** — PaLM, Gemini

🏢 **Используют:**  
- **Tinkoff AI** — Azure OpenAI  
- **Miro AI** — AWS Bedrock  
- **Kaspersky AI** — GCP Vertex

---

## 📊 Model Registry (MLflow, DVC) ⭐⭐

**Что это:**  
Хранилище моделей с версионированием, метаданными и rollout'ом.

**Инструменты:**
- ✅ `MLflow` — registry + tracking  
- ✅ `DVC` — Git-подход к моделям  
- ✅ `HuggingFace Hub` — публичные модели

🏢 **Используют:**  
- **Tinkoff AI** — MLflow + DVC  
- **DeepMind** — MLflow  
- **VK AI** — DVC для моделей и данных

---
## 📈 Monitoring & Observability

---

## 📊 AI Model Drift Detection ⭐⭐⭐

**Что это:**  
Отслеживание изменений в данных или поведении модели — чтобы не деградировала со временем.

**Типы дрейфа:**
- ✅ Data drift — изменились входные данные  
- ✅ Concept drift — изменилась логика задачи  
- ✅ Prediction drift — изменилась структура output'а

**Инструменты:**
- `Evidently`, `WhyLabs`, `Fiddler`, `Arize`

🏢 **Используют:**  
- **Tinkoff AI** — Evidently + ручной аудит  
- **Uber** — автоматическое отслеживание дрейфа  
- **Meta** — мониторинг output'ов LLaMA

---

## 💰 Cost Monitoring для LLM APIs ⭐⭐⭐

**Что это:**  
Контроль затрат на вызовы LLM — токены, RPS, пользователи.

**Метрики:**
- ✅ Токены per request  
- ✅ Стоимость per user/session  
- ✅ Частота вызовов  
- ✅ Кэш-хитрейт

**Инструменты:**
- `OpenAI Usage API`, `Prometheus`, `Grafana`, `FinOps Dashboards`

🏢 **Используют:**  
- **Tinkoff AI** — токен-лимиты и отчёты  
- **Notion AI** — мониторинг по workspace  
- **Replit AI** — billing per user

---

## 🎯 Quality Metrics (Hallucination Detection) ⭐⭐

**Что это:**  
Оценка качества ответов модели — чтобы не "придумывала".

**Метрики:**
- ✅ Faithfulness — ответ основан на источнике  
- ✅ Groundedness — есть ссылка на источник  
- ✅ Toxicity / Bias — нет вредного контента

**Инструменты:**
- `OpenAI evals`, `TruthfulQA`, `RAGAS`, `LlamaGuard`

🏢 **Используют:**  
- **Tinkoff AI** — ручная и автоматическая проверка  
- **OpenAI** — evals + human review  
- **Anthropic** — RLHF на безопасность

---

## 🔍 AI Request Tracing ⭐⭐

**Что это:**  
Трассировка запроса от клиента до модели — для дебага и мониторинга.

**Техники:**
- ✅ `trace_id` в каждом запросе  
- ✅ Логирование промпта, контекста, output'а  
- ✅ Интеграция с `OpenTelemetry`, `Jaeger`, `Sentry`

🏢 **Используют:**  
- **Tinkoff AI** — trace_id + логирование  
- **Slack AI** — трассировка генерации  
- **GitHub Copilot** — логирование промптов

---

## 🏢 Часть 4: Real-World AI Patterns  
🎯 Production AI Patterns

---

## 🔄 Circuit Breaker для AI Services ⭐⭐⭐

**Что это:**  
Механизм защиты от перегрузки — временно отключает сервис при ошибках.

**Зачем:**
- ✅ Защита от каскадных фейлов  
- ✅ Быстрое восстановление  
- ✅ Graceful degradation

**Инструменты:**
- `Resilience4j`, `Envoy`, `Istio`, `FastAPI middleware`

🏢 **Используют:**  
- **Tinkoff AI** — circuit breaker на LLM API  
- **OpenAI** — fallback при перегрузке  
- **Slack AI** — защита от rate limit

---

## ⚡ AI Response Caching ⭐⭐⭐

**Что это:**  
Кэшируем ответы модели — чтобы не вызывать повторно.

**Стратегии:**
- ✅ Кэш по промпту  
- ✅ Кэш retrieval в RAG  
- ✅ Кэш популярных запросов

**Инструменты:**
- `Redis`, `DuckDB`, `LangChain Cache`, `SQLite`

🏢 **Используют:**  
- **Tinkoff AI** — кэш на уровне retrieval  
- **Notion AI** — кэш по workspace  
- **You.com** — кэш похожих запросов

---

## 🎯 Fallback Strategies ⭐⭐

**Что это:**  
Резервные сценарии при ошибке модели или API.

**Примеры:**
- ✅ Возврат шаблонного ответа  
- ✅ Переключение на другую модель  
- ✅ Уведомление пользователя

🏢 **Используют:**  
- **Tinkoff AI** — fallback на GPT-3.5  
- **Slack AI** — шаблонный ответ при ошибке  
- **OpenRouter** — переключение между провайдерами

---

## 📊 A/B Testing AI Models ⭐⭐

**Что это:**  
Сравнение моделей или промптов — чтобы выбрать лучший вариант.

**Метрики:**
- ✅ CTR / Conversion  
- ✅ User feedback  
- ✅ Token usage  
- ✅ Latency

**Инструменты:**
- `Optimizely`, `Split.io`, `LangSmith`, `MLflow`

🏢 **Используют:**  
- **Tinkoff AI** — A/B моделей в чате  
- **GitHub Copilot** — тестирование промптов  
- **Notion AI** — сравнение генераторов

---
## 🏗️ AI System Design

---

## 🤖 Chatbot Architecture ⭐⭐⭐

**Что это:**  
Архитектура для построения продакшн-чатбота с LLM.

**Компоненты:**
- ✅ Input → Preprocessing → Prompt Builder  
- ✅ LLM API / Local Model  
- ✅ Postprocessing → Moderation → Output  
- ✅ Logging, tracing, caching

**Инструменты:**
- `FastAPI`, `LangChain`, `Redis`, `OpenAI`, `Anthropic`

🏢 **Используют:**  
- **Tinkoff AI** — чат с LLM + retrieval  
- **Replika** — кастомная архитектура  
- **Slack AI** — чат + async генерация

---

## 🔍 Semantic Search System ⭐⭐⭐

**Что это:**  
Поиск по embedding'ам вместо ключевых слов.

**Компоненты:**
- ✅ Ingestion → Chunking → Embedding  
- ✅ Vector DB (pgvector, Pinecone)  
- ✅ Retrieval → Ranking → Output

**Инструменты:**
- `sentence-transformers`, `FAISS`, `Weaviate`, `LangChain`

🏢 **Используют:**  
- **You.com** — гибридный поиск  
- **Tinkoff AI** — поиск по базе знаний  
- **Notion AI** — поиск по заметкам

---

## 📝 Content Generation Pipeline ⭐⭐

**Что это:**  
Автоматическая генерация текста, изображений, кода и т.д.

**Этапы:**
- ✅ Input → Prompt → LLM  
- ✅ Postprocessing → Moderation  
- ✅ Logging → Feedback → Retraining

**Инструменты:**
- `OpenAI`, `Anthropic`, `LangChain`, `Guardrails`

🏢 **Используют:**  
- **Notion AI** — генерация заметок  
- **Tinkoff AI** — генерация email/ответов  
- **Replit AI** — генерация кода

---

## 🎯 Recommendation Engine ⭐⭐

**Что это:**  
Система рекомендаций на основе поведения, embedding'ов, графов.

**Методы:**
- ✅ Collaborative filtering  
- ✅ Content-based  
- ✅ Embedding similarity  
- ✅ Hybrid

**Инструменты:**
- `LightFM`, `RecBole`, `FAISS`, `Neo4j`

🏢 **Используют:**  
- **Netflix** — hybrid engine  
- **Amazon** — real-time рекомендации  
- **Tinkoff AI** — рекомендации продуктов

---

## 💼 Часть 5: AI Backend Interview Prep  
🎯 AI Technical Questions

---

## 🧠 "Объясни как работает RAG" ⭐⭐⭐

**Ответ:**  
RAG = Retrieval-Augmented Generation  
1. **Retrieval:** ищем релевантные фрагменты из базы  
2. **Augmentation:** добавляем их в промпт  
3. **Generation:** LLM генерирует ответ с учётом контекста

🏢 **Пример:**  
В **Tinkoff AI** — поиск по базе знаний → вставка в промпт → ответ

---

## 💰 "Как оптимизировать затраты на LLM?" ⭐⭐⭐

**Ответ:**
- ✅ Кэшировать ответы  
- ✅ Использовать embeddings вместо LLM  
- ✅ Ограничивать токены  
- ✅ Использовать GPT-3.5 вместо GPT-4, где можно  
- ✅ Мониторить usage per user/session

🏢 **Пример:**  
**Tinkoff AI** — кэш + токен-лимиты + fallback на GPT-3.5

---

## 🔍 "Vector search vs SQL search" ⭐⭐⭐

**Ответ:**
- **SQL search** — точный, по ключевым словам  
- **Vector search** — по смыслу, embedding'ам  
- ✅ SQL — `WHERE name = 'Valeriy'`  
- ✅ Vector — `similarity(text, query) > 0.8`

🏢 **Пример:**  
**Tinkoff AI** — pgvector + BM25  
**You.com** — hybrid search

---

## ⚡ "Bottlenecks в AI системах" ⭐⭐

**Ответ:**
- ✅ LLM latency  
- ✅ GPU перегрузка  
- ✅ Токенизация больших текстов  
- ✅ Сетевые задержки  
- ✅ Кэш-мисс при retrieval

🏢 **Пример:**  
**Tinkoff AI** — async обработка + кэш + GPU scheduler

---

## 🏗️ System Design для AI

---

## 🤖 "Спроектируй ChatGPT-like сервис" ⭐⭐⭐

**Компоненты:**
- ✅ Frontend: Web + Mobile (React, Flutter)  
- ✅ Backend: FastAPI / Node.js → LLM API  
- ✅ Prompt Builder: системный + пользовательский  
- ✅ Streaming: WebSocket / SSE  
- ✅ Moderation: фильтрация input/output  
- ✅ Logging + Tracing: OpenTelemetry  
- ✅ Кэш: Redis / DuckDB  
- ✅ Rate Limiting + Auth: JWT, OAuth  
- ✅ Model Layer: OpenAI / Anthropic / vLLM

🏢 **Пример:**  
**Tinkoff AI** — FastAPI + GPT + pgvector  
**Slack AI** — Claude + async генерация  
**Replika** — кастомная модель + WebSocket

---

## 🔍 "AI-powered поиск для e-commerce" ⭐⭐

**Компоненты:**
- ✅ Ingestion: товары, описания, отзывы  
- ✅ Embedding: `sentence-transformers`, `text-embedding-3`  
- ✅ Vector DB: pgvector / Pinecone  
- ✅ Hybrid Search: BM25 + cosine similarity  
- ✅ Ranking: по релевантности и CTR  
- ✅ UI: фильтры, подсказки, семантический поиск

🏢 **Пример:**  
**Ozon AI** — поиск по embedding'ам  
**Amazon** — hybrid search + рекомендации  
**Tinkoff AI** — поиск по транзакциям и продуктам

---

## 📊 "ML Platform для 40+ команд" ⭐⭐

**Компоненты:**
- ✅ Feature Store: Feast / Tecton  
- ✅ Model Registry: MLflow / DVC  
- ✅ CI/CD: GitLab + ArgoCD  
- ✅ GPU Orchestration: Kubernetes + MIG  
- ✅ Monitoring: Prometheus + Grafana  
- ✅ Access Control: RBAC, audit logs  
- ✅ Experiment Tracking: Weights & Biases / MLflow  
- ✅ Data Versioning: LakeFS / DVC

🏢 **Пример:**  
**Tinkoff AI** — MLflow + Feast + K8s  
**Uber Michelangelo** — платформа для всех ML-команд  
**Yandex Cloud** — ML платформы с GPU и трекингом

---
