# ========================================================================
#  Script........: search.py
#  Autor.........: Rogério Martins
#  Descrição.....: Módulo de busca e recuperação com RAG usando PGVector
#  Criado em.....: 25/10/2025
#  Última alteração: 25/10/2025
# ========================================================================

import os
from typing import List, Tuple, Callable
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_postgres import PGVector

PROMPT_TEMPLATE = """
CONTEXTO:
{contexto}

REGRAS:
- Responda somente com base no CONTEXTO.
- Se a informação não estiver explicitamente no CONTEXTO, responda:
  "Não tenho informações necessárias para responder sua pergunta."
- Nunca invente ou use conhecimento externo.
- Nunca produza opiniões ou interpretações além do que está escrito.

EXEMPLOS DE PERGUNTAS FORA DO CONTEXTO:
Pergunta: "Qual é a capital da França?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Quantos clientes temos em 2024?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Você acha isso bom ou ruim?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

PERGUNTA DO USUÁRIO:
{pergunta}

RESPONDA A "PERGUNTA DO USUÁRIO"
"""

# Resposta padrão quando não há evidência suficiente no contexto
DEFAULT_NOINFO = 'Não tenho informações necessárias para responder sua pergunta.'


def _format_context(docs_with_scores: List[Tuple[Document, float]]) -> str:
    """
    Concatena os trechos recuperados (Document + score) para compor o CONTEXTO do prompt.

    Parâmetros:
        docs_with_scores: lista de tuplas (Document, score) retornadas pela busca vetorial.

    Retorno:
        Uma string com cabeçalho de cada trecho (índice, score, metadados úteis) e o conteúdo do chunk.
    """
    if not docs_with_scores:
        return ""
    parts = []
    for i, (doc, score) in enumerate(docs_with_scores, start=1):
        meta = doc.metadata or {}
        page = meta.get("page")
        source = meta.get("source") or meta.get("file_path") or ""
        header = f"[Trecho {i} | score={score:.4f}"
        if page is not None:
            header += f" | page={page}"
        if source:
            header += f" | src={source}"
        header += "]"
        parts.append(f"{header}\n{doc.page_content.strip()}\n")
    return "\n".join(parts)


def _load_clients():
    """
    Carrega variáveis de ambiente e instancia clientes necessários ao pipeline RAG.

    Fluxo:
        - Garante presença das variáveis críticas (.env).
        - Cria o cliente de embeddings (OpenAIEmbeddings).
        - Cria o vetor store PGVector (conexão Postgres informada em DATABASE_URL).
        - Cria o cliente de chat LLM (ChatOpenAI).

    Retorno:
        (embeddings, store, llm)
    """
    load_dotenv()

    for k in ("OPENAI_API_KEY", "DATABASE_URL", "PGVECTOR_COLLECTION"):
        if not os.getenv(k):
            raise RuntimeError(f"Environment variable {k} is not set")

    embedding_model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    chat_model = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
    temperature = float(os.getenv("OPENAI_TEMPERATURE", "0"))

    embeddings = OpenAIEmbeddings(model=embedding_model)
    store = PGVector(
        embeddings=embeddings,
        collection_name=os.environ["PGVECTOR_COLLECTION"],
        connection=os.environ["DATABASE_URL"],
        use_jsonb=True,
    )
    llm = ChatOpenAI(model=chat_model, temperature=temperature)

    return embeddings, store, llm


def search_prompt(question: str | None = None) -> Callable[[str, int], str] | str:
    """
    Constrói a cadeia de pesquisa e resposta (RAG) ou executa-a imediatamente.

    Comportamentos:
        - Se 'question' for None:
            retorna uma função `chain(q: str, k: int = 10) -> str` que:
                1) vetoriza q e busca top-k no PGVector,
                2) formata o contexto,
                3) injeta no PROMPT_TEMPLATE,
                4) chama a LLM e retorna a resposta (ou DEFAULT_NOINFO).
        - Se 'question' for str:
            executa imediatamente a mesma cadeia e retorna a resposta.

    Parâmetros:
        question: pergunta opcional para execução imediata.

    Retorno:
        Função `chain(...)` pronta para uso interativo OU a string de resposta.
    """
    embeddings, store, llm = _load_clients()

    def chain(q: str, k: int = 10) -> str:
        """
        Executa o pipeline RAG completo para a pergunta 'q'.

        Etapas:
            1) Busca vetorial (top-k) no PGVector com base na pergunta.
            2) Monta o CONTEXTO a partir dos trechos retornados.
            3) Formata o PROMPT_TEMPLATE com (contexto, pergunta).
            4) Invoca a LLM e retorna o conteúdo textual.

        Parâmetros:
            q: pergunta do usuário.
            k: quantidade de resultados mais relevantes a recuperar (default=10).

        Retorno:
            Resposta da LLM, ou DEFAULT_NOINFO se não houver contexto suficiente.
        """
        # 1) Recuperação por similaridade (PGVector)
        docs_with_scores = store.similarity_search_with_score(q, k=k)

        # 2) Formatação do contexto
        contexto = _format_context(docs_with_scores)

        # Short-circuit: sem contexto, respeite as regras e não invente
        if not contexto.strip():
            return DEFAULT_NOINFO

        # 3) Prompt a partir do template fornecido
        prompt = PROMPT_TEMPLATE.format(contexto=contexto, pergunta=q).strip()

        # 4) Chamada à LLM
        resp = llm.invoke(prompt)
        answer = (resp.content or "").strip()

        # Fallback defensivo
        return answer if answer else DEFAULT_NOINFO

    if question is None:
        return chain
    else:
        # Execução imediata com k definido via env (TOPK) ou padrão 10
        return chain(question, k=int(os.getenv("TOPK", "10")))

