# Santander Dev Week 2023 (ETL com Python + IA + Análise de Sentimentos)

import pandas as pd
import requests
import json
import openai
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# --------------------------------------
# Configurações Iniciais
# --------------------------------------
sdw2023_api_url = 'https://sdw-2023-prd.up.railway.app'
openai_api_key = 'TODO'  # Substitua pela sua chave
openai.api_key = openai_api_key
nltk.download('vader_lexicon')

# --------------------------------------
# [ETAPA EXTRA] - Fonte alternativa se API estiver offline
# --------------------------------------
# Esse bloco simula dados caso a API esteja inativa.
# Pode ser substituído por banco de dados, webhook, Excel etc.
fake_client_data = {
    "id": 101,
    "name": "Maria Teste",
    "account": {
        "number": "98765432",
        "agency": "0001",
        "balance": 1200.00
    },
    "transactions": [
        {"descricao": "comprei um presente incrível"},
        {"descricao": "paguei multa atrasada"},
        {"descricao": "ganhei cashback da Netflix"},
        {"descricao": "empréstimo caiu na conta"},
        {"descricao": "tive que pagar uma taxa absurda"}
    ],
    "news": []
}

# --------------------------------------
# Extract
# --------------------------------------
df = pd.read_csv('SDW2023.csv')
user_ids = df['UserID'].tolist()


def get_user(id):
    try:
        response = requests.get(f'{sdw2023_api_url}/users/{id}')
        return response.json() if response.status_code == 200 else None
    except:
        return None

users = [user for id in user_ids if (user := get_user(id)) is not None]
if not users:
    users = [fake_client_data]  # fallback se a API estiver offline

# --------------------------------------
# Transform - IA Generativa + Análise de Sentimento
# --------------------------------------

def generate_ai_news(user):
    completion = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "Você é um especialista em marketing bancário."},
            {"role": "user", "content": f"Crie uma mensagem para {user['name']} sobre a importância dos investimentos (100 caracteres)"}
        ]
    )
    return completion.choices[0].message.content.strip('"')


analyzer = SentimentIntensityAnalyzer()

def classificar_sentimento(score):
    if score >= 0.05:
        return 'positivo'
    elif score <= -0.05:
        return 'negativo'
    else:
        return 'neutro'

for user in users:
    news = generate_ai_news(user)
    user['news'].append({
        "icon": "https://digitalinnovationone.github.io/santander-dev-week-2023-api/icons/credit.svg",
        "description": news
    })

    # Análise de sentimento nas descrições de transações
    transacoes = [t['descricao'] for t in user['transactions']]
    df_sentimento = pd.DataFrame({'descricao_transacao': transacoes})
    df_sentimento['sentimento'] = df_sentimento['descricao_transacao'].apply(lambda x: analyzer.polarity_scores(x)['compound'])
    df_sentimento['classificacao'] = df_sentimento['sentimento'].apply(classificar_sentimento)

    print(f"\n📊 Sentimentos detectados para {user['name']}")
    print(df_sentimento[['descricao_transacao', 'classificacao']])

# --------------------------------------
# Load
# --------------------------------------

def update_user(user):
    try:
        response = requests.put(f"{sdw2023_api_url}/users/{user['id']}", json=user)
        return response.status_code == 200
    except:
        return False

for user in users:
    success = update_user(user)
    print(f"✅ User {user['name']} atualizado? {success}!")
