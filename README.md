# BootcampSantander

# 🔧 [Bloco de Modificação / Personalização]
# Este bloco simula a entrada de dados que originalmente viriam de uma API externa.
# A URL base da API fornecida no projeto original pode estar offline ou indisponível.
# Por isso, os dados abaixo servem como exemplo genérico.

# 💡 Sugestão: Este ponto do código pode ser adaptado para buscar dados de outras fontes como:
# - Webhooks personalizados
# - Serviços em nuvem (Firebase, Supabase, etc)
# - Bancos de dados locais (SQLite, PostgreSQL)
# - Planilhas online (Google Sheets via API)
# - Ou qualquer outro backend de sua preferência

# Dados de exemplo usados temporariamente:
fake_client_data = {
    "id": 101,
    "name": "Maria Teste",
    "account": {
        "number": "98765432",
        "agency": "0001",
        "balance": 1200.00
    }
}

client = fake_client_data
print("📌 Dados mock carregados. Fonte pode ser alterada conforme necessidade.")


# 🧠 [Bloco de Modificação Opcional - Análise de Sentimentos nas Transações]
# Este bloco aplica uma análise de sentimentos usando o modelo VADER (pré-treinado do NLTK).
# A ideia é associar emoções positivas ou negativas às descrições das transações financeiras.
# Isso pode ser útil para detectar padrões, comportamentos incomuns ou melhorar a experiência do cliente.

# 💡 Possíveis usos práticos:
# - Transações com sentimentos negativos frequentes podem indicar problemas financeiros ou fraudes.
# - Agrupar clientes por padrão emocional de consumo.
# - Oferecer produtos bancários com base no perfil comportamental.

# Requisitos: nltk (e a primeira vez rodando: nltk.download('vader_lexicon'))

import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd

nltk.download('vader_lexicon')

# Exemplo de dados de transações com descrição
df = pd.DataFrame({
    'descricao_transacao': [
        'comprei um presente incrível',
        'paguei multa atrasada',
        'ganhei cashback da Netflix',
        'empréstimo caiu na conta',
        'tive que pagar uma taxa absurda'
    ]
})

# Inicializa o analisador de sentimentos
analyzer = SentimentIntensityAnalyzer()

# Aplica o modelo em cada descrição e adiciona o score de sentimento
df['sentimento'] = df['descricao_transacao'].apply(lambda x: analyzer.polarity_scores(x)['compound'])

# Exibe a tabela com os scores
print("📊 Análise de Sentimentos aplicada nas transações:")
display(df)


⚙️ O que o .compound retorna?
    Valor entre -1 (negativo extremo) e +1 (positivo extremo).
    Ex:
        "ganhei cashback" → positivo.
        "paguei multa" → negativo.
