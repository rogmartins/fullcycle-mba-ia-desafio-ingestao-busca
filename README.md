# Desafio MBA Engenharia de Software com IA - Full Cycle

# 🧠 Chat RAG com PGVector e OpenAI

Este projeto permite **consultar o conteúdo de um documento PDF** por meio de perguntas em linguagem natural, usando **busca vetorial (PGVector)** e um **modelo de linguagem (LLM)** da OpenAI.

---

## 📁 Estrutura do projeto

```
src/
 ├─ ingest.py   → Faz a ingestão do PDF (vetorização e gravação no banco)
 ├─ search.py   → Implementa a lógica de recuperação e geração de respostas
 └─ chat.py     → Interface de chat no terminal
```

---

## ⚙️ Requisitos

- Python 3.10+  
- PostgreSQL com extensão **pgvector** instalada  
- Pacotes Python:

```bash
pip install langchain langchain-openai langchain-community langchain-postgres psycopg[binary] python-dotenv
```

---

## 🔧 Configuração

Crie um arquivo `.env` na raiz do projeto com as variáveis necessárias:

```env
OPENAI_API_KEY=sk-...
OPENAI_CHAT_MODEL=gpt-4o-mini
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
OPENAI_TEMPERATURE=0

DATABASE_URL=postgresql+psycopg://usuario:senha@localhost:5432/sua_base
PGVECTOR_COLLECTION=colecao_pdf
PDF_PATH=/caminho/para/seu/documento.pdf

TOPK=10
```

> 💡 **Importante:**  
> - O modelo de embeddings deve ser o mesmo na ingestão e nas consultas.  
> - A base de dados deve ter a extensão `pgvector` habilitada (`CREATE EXTENSION IF NOT EXISTS vector;`).

---

## 📥 1. Ingestão do PDF

Antes de fazer perguntas, é necessário **indexar o conteúdo do PDF** no banco vetorial.

Execute:

```bash
python3 -m src/ingest.py
```

Isso:
- Carrega o PDF definido em `PDF_PATH`;
- Divide em trechos (`chunks`);
- Gera embeddings com o modelo da OpenAI;
- Armazena tudo na coleção `PGVECTOR_COLLECTION`.

---

## 💬 2. Rodando o chat

Após a ingestão, execute:

```bash
python -m src.chat
```

Você verá:

```
======== CHAT PDF (RAG com PGVector) ========
Digite sua pergunta e pressione Enter.
Comandos: :sair  (ou :q / :quit / :exit) para encerrar
```

Agora basta digitar perguntas relacionadas ao conteúdo do PDF.

Exemplo:

```
> Qual é o objetivo principal do documento?
--- Resposta ---
O documento descreve ...
----------------
```

---

## 📚 Funcionamento interno (resumo)

1. **Vetoriza a pergunta** com o mesmo modelo de embeddings.  
2. **Busca os 10 trechos mais similares** (k=10) no banco PGVector.  
3. **Monta o prompt** com esses trechos e a pergunta do usuário.  
4. **Chama o modelo de chat** da OpenAI.  
5. **Retorna a resposta** diretamente no terminal.

---

## 🛠️ Personalizações

- Edite `TOPK` no `.env` para mudar o número de trechos recuperados.  
- Altere `PROMPT_TEMPLATE` em `src/search.py` para modificar o estilo e regras de resposta.  
- Substitua o modelo `OPENAI_CHAT_MODEL` por outro compatível (ex: `gpt-4o`).

---

## 🚪 Encerrando o chat

Durante o uso, digite qualquer um dos comandos abaixo para sair:

```
:sair   |   :q   |   :quit   |   :exit
```

---

## ✅ Conclusão

Com essa aplicação, é possível criar um **chat de perguntas e respostas baseado em documentos PDF**, combinando:
- Vetorização e busca semântica via **PGVector**;
- Geração de respostas precisas via **OpenAI Chat Models**;
- Execução simples em terminal com Python.

---