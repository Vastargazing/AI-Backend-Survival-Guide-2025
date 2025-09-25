# 🤖 AI Backend Survival Guide 2025

<img width="624" height="624" alt="Image" src="https://github.com/user-attachments/assets/df83e7cf-8f98-41a3-9076-68a795b25a9d" />

Это практическая шпаргалка для инженеров, работающих с продакшн-системами на базе LLM, RAG и ML.  
Документ охватывает архитектуру, инструменты, безопасность, мониторинг и реальные паттерны, которые применяются в банках, финтехе, e-commerce и AI-продуктах.

---



## 📚 Оглавление

### **Часть 1: AI Backend Fundamentals**

#### 🧠 **Vector & Embeddings**
- [🔍 Vector Databases (pgvector, Pinecone, Weaviate)](#-vector-databases-pgvector-pinecone-weaviate-) ⭐⭐⭐
- [📊 Embeddings: что это и как работает](#-embeddings-что-это-и-как-работает-) ⭐⭐⭐
- [🎯 Similarity Search и Distance Metrics](#-similarity-search-и-distance-metrics-) ⭐⭐
- [⚡ Vector Indexing (HNSW, IVF)](#-vector-indexing-hnsw-ivf-) ⭐⭐

#### 🌐 **LLM Integration**
- [🤖 LLM APIs: OpenAI, Anthropic, Local Models](#-llm-apis-openai-anthropic-local-models-) ⭐⭐⭐
- [💰 Token Management и Cost Optimization](#-token-management-и-cost-optimization-) ⭐⭐⭐
- [🔄 Prompt Engineering для Production](#-prompt-engineering-для-production-) ⭐⭐⭐
- [⚡ Streaming Responses](#-streaming-responses-) ⭐⭐
- [🛡️ Rate Limiting для LLM APIs](#️-rate-limiting-для-llm-apis-) ⭐⭐

#### 🏗️ **RAG Architecture**
- [🔍 RAG Pipeline: Ingestion → Retrieval → Generation](#-rag-pipeline-ingestion--retrieval--generation-) ⭐⭐⭐
- [📝 Document Processing (PDF, DOC, HTML)](#-document-processing-pdf-doc-html-) ⭐⭐
- [🧹 Text Chunking Strategies](#-text-chunking-strategies-) ⭐⭐⭐
- [🎯 Hybrid Search (Vector + BM25)](#-hybrid-search-vector--bm25-) ⭐⭐
- [📊 RAG Evaluation Metrics](#-rag-evaluation-metrics-) ⭐⭐

---

### **Часть 2: Production AI Systems**

#### ⚡ **Performance & Scalability**
- [⚡ AI Model Caching Strategies](#-ai-model-caching-strategies-) ⭐⭐⭐
- [🔄 Async Processing для AI Tasks](#-async-processing-для-ai-tasks-) ⭐⭐⭐
- [📊 GPU Resource Management](#-gpu-resource-management-) ⭐⭐
- [⚖️ Load Balancing для AI Workloads](#️-load-balancing-для-ai-workloads-) ⭐⭐
- [🎯 Model Serving (ONNX, TensorRT)](#-model-serving-onnx-tensorrt-) ⭐

#### 🗄️ **AI Data Architecture**
- [📊 Feature Stores (Feast, Tecton)](#-feature-stores-feast-tecton-) ⭐⭐
- [⚡ Real-time vs Batch ML Pipelines](#-real-time-vs-batch-ml-pipelines-) ⭐⭐⭐
- [🔄 Data Versioning для ML](#-data-versioning-для-ml-) ⭐⭐
- [📈 Vector Database Scaling](#-vector-database-scaling-) ⭐⭐

#### 🔐 **AI Security & Ethics**
- [🛡️ Prompt Injection Protection](#️-prompt-injection-protection-) ⭐⭐⭐
- [🔒 PII Detection в AI Responses](#-pii-detection-в-ai-responses-) ⭐⭐⭐
- [⚖️ AI Content Moderation](#️-ai-content-moderation-) ⭐⭐
- [📋 AI Compliance (GDPR, AI Act)](#-ai-compliance-gdpr-ai-act-) ⭐

---

### **Часть 3: AI Backend Tools & Stack**

#### 🔧 **Essential Libraries**
- [🐍 LangChain vs LlamaIndex vs Custom](#-langchain-vs-llamaindex-vs-custom-) ⭐⭐⭐
- [📊 Transformers, Sentence-Transformers](#-transformers-sentence-transformers-) ⭐⭐
- [⚡ FastAPI + WebSockets для AI](#-fastapi--websockets-для-ai-) ⭐⭐⭐
- [📈 MLflow, Weights & Biases](#-mlflow-weights--biases-) ⭐⭐

#### 🏗️ **Infrastructure**
- [🐳 Docker для ML Models](#-docker-для-ml-models-) ⭐⭐⭐
- [☸️ Kubernetes + GPU Scheduling](#️-kubernetes--gpu-scheduling-) ⭐⭐
- [☁️ Cloud AI Services (AWS Bedrock, Azure OpenAI)](#️-cloud-ai-services-aws-bedrock-azure-openai-) ⭐⭐
- [📊 Model Registry (MLflow, DVC)](#-model-registry-mlflow-dvc-) ⭐⭐

#### 📈 **Monitoring & Observability**
- [📊 AI Model Drift Detection](#-ai-model-drift-detection-) ⭐⭐⭐
- [💰 Cost Monitoring для LLM APIs](#-cost-monitoring-для-llm-apis-) ⭐⭐⭐
- [🎯 Quality Metrics (Hallucination Detection)](#-quality-metrics-hallucination-detection-) ⭐⭐
- [🔍 AI Request Tracing](#-ai-request-tracing-) ⭐⭐
- [📊 AI Monitoring Dashboards (Grafana, DataDog)](#-ai-monitoring-dashboards-grafana-datadog-) ⭐⭐⭐

---

### **Часть 4: Real-World AI Patterns**

#### 🎯 **Production AI Patterns**
- [🔄 Circuit Breaker для AI Services](#-circuit-breaker-для-ai-services-) ⭐⭐⭐
- [⚡ AI Response Caching](#-ai-response-caching-) ⭐⭐⭐
- [🎯 Fallback Strategies](#-fallback-strategies-) ⭐⭐
- [📊 A/B Testing AI Models](#-ab-testing-ai-models-) ⭐⭐

#### 🏗️ **AI System Design**
- [🤖 Chatbot Architecture](#-chatbot-architecture-) ⭐⭐⭐
- [🔍 Semantic Search System](#-semantic-search-system-) ⭐⭐⭐
- [📝 Content Generation Pipeline](#-content-generation-pipeline-) ⭐⭐
- [🎯 Recommendation Engine](#-recommendation-engine-) ⭐⭐

---

### **Часть 5: AI Backend Interview Prep**

#### 🎯 **AI Technical Questions**
- [🧠 "Объясни как работает RAG"](#-объясни-как-работает-rag-) ⭐⭐⭐
- [💰 "Как оптимизировать затраты на LLM?"](#-как-оптимизировать-затраты-на-llm-) ⭐⭐⭐
- [🔍 "Vector search vs SQL search"](#-vector-search-vs-sql-search-) ⭐⭐⭐
- [⚡ "Bottlenecks в AI системах"](#-bottlenecks-в-ai-системах-) ⭐⭐

#### 🏗️ **System Design для AI**
- [🤖 "Спроектируй ChatGPT-like сервис"](#-спроектируй-chatgpt-like-сервис-) ⭐⭐⭐
- [🔍 "AI-powered поиск для e-commerce"](#-ai-powered-поиск-для-e-commerce-) ⭐⭐
- [📊 "ML Platform для 40+ команд"](#-ml-platform-для-40-команд-) ⭐⭐

---

## 📖 **Легенда важности**
- ⭐⭐⭐ **Must Know** - критично для AI Backend Engineer
- ⭐⭐ **Should Know** - желательно для продвинутого уровня  
- ⭐ **Nice to Know** - для экспертного уровня

---

## 🔥 **Быстрый старт**
**Новичок в AI Backend?** Начни с разделов ⭐⭐⭐  
**Готовишься к собесу?** Изучи "Часть 5: AI Backend Interview Prep"  
**Хочешь system design?** Переходи к "Часть 4: Real-World AI Patterns"  
**Production опыт?** Сосредоточься на "Часть 2: Production AI Systems"

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
- **OpenAI ChatGPT** — pgvector для RAG в custom GPTs
- **Notion AI** — Pinecone для поиска по заметкам  
- **Alibaba Cloud** — Weaviate для e-commerce поиска
- **Yandex Cloud** — ClickHouse + vector extensions

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
- **You.com** — embeddings для semantic search
- **Baidu AI** — ERNIE embeddings для поиска по транзакциям  
- **GitHub Copilot** — code embeddings для подсказок
- **Alibaba Qwen** — multilingual embeddings

---

## 🎯 Similarity Search и Distance Metrics ⭐⭐

**Что это:**  
Поиск ближайших векторов по метрике расстояния.

### **Сравнение Distance Metrics:**

| Метрика | Формула | Лучше для | Недостатки |
|---------|---------|-----------|------------|
| **Cosine** | `1 - (A·B)/(||A||·||B||)` | Текст, семантика | Игнорирует magnitude |
| **Euclidean** | `√Σ(ai-bi)²` | Изображения, embeddings | Чувствителен к размерности |
| **Dot Product** | `Σ(ai·bi)` | Рекомендации | Зависит от длины вектора |
| **Manhattan** | `Σ|ai-bi|` | Sparse features | Медленнее на высоких размерностях |

### **Performance Comparison:**
```
Cosine:     ~0.1ms per 1K vectors (768dim)
Euclidean:  ~0.08ms per 1K vectors  
Dot Product: ~0.05ms per 1K vectors
Manhattan:  ~0.12ms per 1K vectors
```

🏢 **Используют:**  
- **Amazon** — cosine для продуктовых рекомендаций  
- **Netflix** — dot product для user-item similarity  
- **Google Search** — euclidean для image embeddings
- **Alibaba** — cosine для e-commerce поиска

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
- **Notion AI** — OpenAI GPT-4 для генерации контента
- **Claude by Anthropic** — используется в Slack, Notion
- **Google Bard** — PaLM/Gemini для поиска и чатов  
- **Local deployments** — Alibaba Qwen, Yandex GPT для приватных данных

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

### **Production Prompt Patterns:**

**1. Chain-of-Thought (CoT):**
```python
def create_cot_prompt(question: str) -> str:
    return f"""
Реши задачу пошагово:

Вопрос: {question}

Давай думать пошагово:
1. Сначала определю, что нужно найти
2. Затем проанализирую исходные данные  
3. Применю нужную логику/формулу
4. Проверю результат на здравый смысл

Ответ:
"""

# Пример использования
prompt = create_cot_prompt("Если в корзине 12 яблок и я съел треть, сколько осталось?")
```

**2. Few-Shot Learning:**
```python
def build_few_shot_prompt(examples: list, new_input: str) -> str:
    prompt = "Анализируй тональность отзывов:\n\n"
    
    for example in examples:
        prompt += f"Отзыв: {example['text']}\n"
        prompt += f"Тональность: {example['sentiment']}\n\n"
    
    prompt += f"Отзыв: {new_input}\nТональность:"
    return prompt

# Production example
examples = [
    {"text": "Отличный продукт, всем рекомендую!", "sentiment": "положительная"},
    {"text": "Ужасное качество, деньги на ветер", "sentiment": "отрицательная"},
    {"text": "Нормально, но есть недочёты", "sentiment": "нейтральная"}
]
```

**3. Role-Based Prompting:**
```python
def create_role_prompt(role: str, task: str, context: str = "") -> str:
    roles = {
        "banker": "Ты — опытный банковский аналитик с 15-летним стажем.",
        "developer": "Ты — senior Python разработчик, эксперт по чистому коду.",
        "lawyer": "Ты — корпоративный юрист, специализируешься на IT-праве."
    }
    
    return f"""
{roles[role]}

{context}

Задача: {task}

Отвечай профессионально, основываясь на своём опыте:
"""
```

**4. Prompt Versioning & A/B Testing:**
```python
class PromptTemplate:
    def __init__(self, name: str, version: str, template: str):
        self.name = name
        self.version = version  
        self.template = template
        self.metrics = {"accuracy": 0, "latency": 0, "cost": 0}
    
    def render(self, **kwargs) -> str:
        return self.template.format(**kwargs)

# Version management
PROMPTS = {
    "summarization_v1": PromptTemplate(
        "summarization", "v1",
        "Кратко перескажи текст в 3 предложениях:\n{text}"
    ),
    "summarization_v2": PromptTemplate(
        "summarization", "v2", 
        "Создай структурированный summary:\n\nОсновные тезисы:\n{text}\n\nВывод в 2-3 предложениях:"
    )
}
```

🏢 **Используют:**  
- **GitHub Copilot** — few-shot + role-based для code generation
- **Google Bard** — CoT для математических задач  
- **Anthropic Claude** — constitutional AI prompts для безопасности
- **Alibaba Qwen** — role-based промпты для разных экспертных доменов
- **Yandex GPT** — few-shot learning для русскоязычных задач

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

### **Сравнение Chunking Strategies:**

| Стратегия | Размер | Overlap | Плюсы | Минусы | Лучше для |
|-----------|--------|---------|--------|--------|-----------|
| **Fixed Size** | 500-1000 токенов | 10-20% | Простота, стабильность | Может резать предложения | Общие документы |
| **Semantic** | Переменный | По смыслу | Сохраняет контекст | Сложнее реализовать | Структурированный текст |
| **Recursive** | Адаптивный | 20% | Баланс размера/смысла | Может быть медленным | Смешанный контент |
| **Document-based** | По разделам | Заголовки | Естественные границы | Неравномерные размеры | Техническая документация |

### **Code Example - Semantic Chunking:**
```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

def smart_chunk_text(text: str, chunk_size: int = 1000):
    """Умное разбиение с overlap и сохранением контекста"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=200,  # 20% overlap
        separators=["\n\n", "\n", ".", "!", "?", " "],
        length_function=len
    )
    
    chunks = splitter.split_text(text)
    
    # Добавляем метаданные для каждого chunk'а
    enhanced_chunks = []
    for i, chunk in enumerate(chunks):
        enhanced_chunks.append({
            "text": chunk,
            "chunk_id": i,
            "char_count": len(chunk),
            "word_count": len(chunk.split())
        })
    
    return enhanced_chunks
```

🏢 **Используют:**  
- **LangChain** — RecursiveCharacterTextSplitter  
- **OpenAI** — semantic chunking для ChatGPT plugins
- **Anthropic** — document-based для Claude  
- **Yandex GPT** — adaptive chunking для YaGPT

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

## 📊 AI Monitoring Dashboards (Grafana, DataDog) ⭐⭐⭐

**Что мониторим в AI системах:**

### **LLM Performance Metrics**
- **Latency:** P50, P95, P99 времени ответа
- **Throughput:** RPS (requests per second)  
- **Token Usage:** input/output tokens per request
- **Error Rate:** 4xx/5xx ошибки, timeouts

### **Business Metrics**
- **Cost per Request:** $ на запрос/пользователя/день
- **User Engagement:** session length, repeat usage
- **Quality Score:** thumbs up/down, CSAT
- **Cache Hit Rate:** % кэшированных ответов

### **Infrastructure Metrics**
- **GPU Utilization:** % загрузки, memory usage
- **Queue Depth:** pending AI tasks
- **Model Load Time:** время загрузки модели
- **Database Performance:** vector search latency

### **Grafana Dashboard Example:**
```json
{
  "dashboard": {
    "title": "AI Backend Monitoring",
    "panels": [
      {
        "title": "LLM Request Latency",
        "type": "stat",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, llm_request_duration_seconds)",
            "legendFormat": "P95 Latency"
          }
        ]
      },
      {
        "title": "Token Usage by Model",
        "type": "bargauge", 
        "targets": [
          {
            "expr": "rate(llm_tokens_total[5m]) by (model)",
            "legendFormat": "{{model}}"
          }
        ]
      },
      {
        "title": "Cost Tracking",
        "type": "singlestat",
        "targets": [
          {
            "expr": "sum(llm_cost_usd_total)",
            "legendFormat": "Total Cost ($)"
          }
        ]
      }
    ]
  }
}
```

### **Alert Rules:**
```yaml
groups:
  - name: ai_backend_alerts
    rules:
      - alert: HighLLMLatency
        expr: histogram_quantile(0.95, llm_request_duration_seconds) > 5
        for: 2m
        labels:
          severity: warning
        annotations:
          summary: "LLM latency is high ({{ $value }}s)"
          
      - alert: LLMCostSpike
        expr: rate(llm_cost_usd_total[1h]) > 100
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "LLM costs spiking: ${{ $value }}/hour"
```

🏢 **Используют:**
- **Anthropic** — Grafana + Prometheus для мониторинга Claude API
- **Google AI** — DataDog для Bard infrastructure monitoring  
- **Alibaba Cloud** — кастомные дашборды для Qwen models
- **Yandex GPT** — внутренние дашборды + YandexCloud monitoring

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
- ✅ User feedback (thumbs up/down)
- ✅ Token usage efficiency
- ✅ Response latency
- ✅ Task completion rate

### **A/B Testing Framework:**
```python
import random
from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class ABTestConfig:
    experiment_id: str
    variants: Dict[str, Any]  # model configs
    traffic_split: Dict[str, float]  # percentage per variant
    
def route_to_variant(user_id: str, config: ABTestConfig) -> str:
    """Consistent user routing to A/B variants"""
    hash_val = hash(f"{config.experiment_id}_{user_id}") % 100
    
    cumulative = 0
    for variant, percentage in config.traffic_split.items():
        cumulative += percentage * 100
        if hash_val < cumulative:
            return variant
    
    return list(config.variants.keys())[0]  # fallback

# Example A/B test config
experiment = ABTestConfig(
    experiment_id="gpt4_vs_claude",
    variants={
        "gpt4": {"model": "gpt-4", "temperature": 0.7},
        "claude": {"model": "claude-3", "temperature": 0.7}
    },
    traffic_split={"gpt4": 0.5, "claude": 0.5}
)
```

**Инструменты:**
- ✅ **LangSmith** — A/B testing промптов с автоматическими метриками
- ✅ **Optimizely** — feature flags для model routing
- ✅ **Split.io** — real-time experiment management  
- ✅ **MLflow** — experiment tracking и comparison
- ✅ **Weights & Biases** — model performance comparison

### **LangSmith Integration:**
```python
from langsmith import Client
from langsmith.evaluation import evaluate

# Автоматическое A/B тестирование промптов
client = Client()

def evaluate_prompts():
    dataset = client.create_dataset("customer_support_qa")
    
    # Тестируем разные промпты
    results = evaluate(
        lambda inputs: run_chain_variant_a(inputs),
        data=dataset,
        evaluators=[
            "qa_correctness",
            "helpfulness", 
            "response_time"
        ],
        experiment_prefix="prompt_variant_a"
    )
    
    return results
```

🏢 **Используют:**  
- **Anthropic** — A/B тестирование safety prompts в Claude
- **Google Bard** — comparing response quality across model versions
- **GitHub Copilot** — testing code generation prompts
- **Alibaba Qwen** — A/B testing для разных доменов (код, математика, текст)

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
