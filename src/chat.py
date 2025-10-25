import os
from dotenv import load_dotenv

from search import search_prompt

# Comandos de saída do chat
EXIT_COMMANDS = {":q", ":quit", ":exit", ":sair"}


def main():
    load_dotenv()

    try:
        chain = search_prompt()
    except Exception as e:
        print(f"Não foi possível iniciar o chat. Verifique os erros de inicialização.\nDetalhes: {e}")
        return

    if not chain:
        print("Não foi possível iniciar o chat. Verifique os erros de inicialização.")
        return
    
    # Define o número de resultados mais relevantes a serem retornados
    topk = int(os.getenv("TOPK", "10"))

    print("======== CHAT PDF (RAG com PGVector) ========")
    print("Digite sua pergunta e pressione Enter.")
    print("Comandos: :sair  (ou :q / :quit / :exit) para encerrar\n")

    while True:
        try:
            # Captura a entrada do usuário (pergunta ou comando)
            q = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nEncerrando...")
            break
        
        # Ignora entradas vazias
        if not q:
            continue
        
        # Encerra o loop se o usuário digitar um comando de saída
        if q.lower() in EXIT_COMMANDS:
            print("Tchau!")
            break

        try:
            #  Executa a busca / recuperação na base de dados da questão formulada pelo usuário,
            # limitando os resultados aos top-k mais relevantes (onde topk=10).
            #  Armazena a resposta na variável 'answer'.
            answer = chain(q, k=topk)
            print("\n--- Resposta ---")
            print(answer)
            print("----------------\n")
        except Exception as e:
            print(f"[erro] {e}\n")


if __name__ == "__main__":
    main()