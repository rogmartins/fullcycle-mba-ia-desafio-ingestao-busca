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

DEFAULT_NOINFO = 'Não tenho informações necessárias para responder sua pergunta.'


def _format_context(docs_with_scores: List[Tuple[Document, float]]) -> str:
    """Formata os chunks recuperados para uso no CONTEXTO do prompt."""
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
    """Inicializa e retorna (embeddings, store, llm) prontos para uso."""
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
    Se 'question' for None, retorna uma função chamável:
        chain(q: str, k: int = 10) -> str

    Se 'question' for uma string, executa o pipeline imediatamente e retorna a resposta.
    """
    embeddings, store, llm = _load_clients()

    def chain(q: str, k: int = 10) -> str:
        # 1+2) Recuperação com similaridade
        docs_with_scores = store.similarity_search_with_score(q, k=k)

        # 3) Montagem do prompt a partir do template fornecido
        contexto = _format_context(docs_with_scores)

        # Caso não haja contexto, respeite a regra e responda diretamente
        if not contexto.strip():
            return DEFAULT_NOINFO

        prompt = PROMPT_TEMPLATE.format(contexto=contexto, pergunta=q).strip()

        # 4) Chamada à LLM
        resp = llm.invoke(prompt)
        answer = (resp.content or "").strip()

        # Fallback defensivo: se por algum motivo vier vazio, devolve a mensagem padrão
        return answer if answer else DEFAULT_NOINFO

    if question is None:
        return chain
    else:
        # Execução imediata
        return chain(question, k=int(os.getenv("TOPK", "10")))

