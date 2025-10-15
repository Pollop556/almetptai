import os
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, util
import csv
import re
import logging
from flask import Flask, request, jsonify, render_template_string

# Настройка логирования
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

class FAQBot:
    def __init__(self, model_dir="./faq_model", csv_path="data_conter.csv", template_path="shablon.csv"):
        self.model_dir = model_dir
        self.csv_path = csv_path
        self.template_path = template_path
        self.model = None
        self.df = None
        self.templates = {}
        self.corpus_embeddings = None
        
        os.makedirs(model_dir, exist_ok=True)
        self.load_or_download_model()
        self.load_data()
        self.load_templates()
        self.encode_corpus()
    
    def load_or_download_model(self):
        model_name = 'cointegrated/rubert-tiny2'
        local_model_path = os.path.join(self.model_dir, 'rubert-tiny2')
        try:
            logger.info("Пытаюсь загрузить модель из локальной папки...")
            self.model = SentenceTransformer(local_model_path)
            logger.info("✅ Модель успешно загружена из локальной папки!")
        except Exception as e:
            logger.info(f"❌ Локальная модель не найдена ({e}). Скачиваю модель...")
            self.model = SentenceTransformer(model_name)
            self.model.save(local_model_path)
            logger.info(f"✅ Модель сохранена в: {local_model_path}")
    
    def load_data(self):
        try:
            logger.info(f"Пытаюсь загрузить данные из {self.csv_path}...")
            try:
                self.df = pd.read_csv(self.csv_path, header=None, names=['question', 'answer'])
            except:
                self.df = pd.read_csv(
                    self.csv_path, 
                    header=None, 
                    names=['question', 'answer'],
                    quoting=csv.QUOTE_ALL,
                    escapechar='\\'
                )
            logger.info(f"📊 Загружено {len(self.df)} вопросов-ответов")
        except Exception as e:
            logger.error(f"❌ Ошибка при загрузке CSV: {e}")
            self.load_data_manual()
    
    def load_data_manual(self):
        data = []
        with open(self.csv_path, 'r', encoding='utf-8') as file:
            reader = csv.reader(file)
            for i, row in enumerate(reader):
                if len(row) >= 2:
                    question = row[0].strip()
                    answer = ' '.join(row[1:]).strip()
                    data.append([question, answer])
                else:
                    logger.warning(f"⚠️ Пропущена строка {i+1}: {row}")
        self.df = pd.DataFrame(data, columns=['question', 'answer'])
        if len(self.df) == 0:
            raise Exception("Не удалось загрузить данные")
    
    def load_templates(self):
        try:
            with open(self.template_path, 'r', encoding='utf-8') as file:
                for line in file:
                    line = line.strip()
                    if ',' in line:
                        key, value = line.split(',', 1)
                        key = key.strip().strip('[]')
                        value = value.strip().strip('"')
                        self.templates[key] = value
        except Exception as e:
            logger.warning(f"Шаблоны не загружены: {e}")
    
    def replace_templates(self, text):
        if not self.templates:
            return text
        result = text
        for key, val in self.templates.items():
            result = result.replace(f"[{key}]", val)
        return result
    
    def encode_corpus(self):
        logger.info("Кодирую базу вопросов...")
        questions = self.df['question'].tolist()
        self.corpus_embeddings = self.model.encode(
            questions,
            convert_to_tensor=True,
            show_progress_bar=False,
            normalize_embeddings=True
        )
        logger.info("✅ База вопросов закодирована!")
    
    def find_best_answer(self, query, threshold=0.3):
        query_embedding = self.model.encode(query, convert_to_tensor=True, normalize_embeddings=True)
        cos_scores = util.cos_sim(query_embedding, self.corpus_embeddings)[0]
        top_score, top_idx = torch.max(cos_scores, dim=0)
        score = top_score.item()
        if score < threshold:
            return {"answer": "К сожалению, я не нашёл подходящего ответа. Попробуйте переформулировать вопрос.", "score": score}
        original_answer = self.df.iloc[top_idx.item()]['answer']
        processed_answer = self.replace_templates(original_answer)
        return {"answer": processed_answer, "score": score}

# === Flask App ===
app = Flask(__name__)
faq_bot = None

HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
    <title>AlmetPT AI</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, sans-serif;
        }
        body {
            background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 12px;
            color: #fff;
        }
        .chat-container {
            width: 100%;
            max-width: 800px;
            background: rgba(20, 20, 35, 0.75);
            backdrop-filter: blur(16px);
            border-radius: 24px;
            overflow: hidden;
            box-shadow: 0 12px 40px rgba(0, 0, 0, 0.5);
            display: flex;
            flex-direction: column;
            height: 92vh;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }
        .header {
            padding: 18px 20px;
            text-align: center;
            background: rgba(0, 0, 0, 0.2);
        }
        .header h1 {
            font-size: 1.6rem;
            font-weight: 700;
            background: linear-gradient(90deg, #a18cd1, #fbc2eb);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            letter-spacing: -0.5px;
        }
        .header p {
            font-size: 0.9rem;
            opacity: 0.85;
            margin-top: 6px;
            color: #e0e0ff;
        }
        .chat-messages {
            flex: 1;
            padding: 16px;
            overflow-y: auto;
            display: flex;
            flex-direction: column;
            gap: 14px;
            scroll-behavior: smooth;
        }
        .message {
            max-width: 85%;
            padding: 14px 18px;
            border-radius: 20px;
            line-height: 1.5;
            word-wrap: break-word;
            white-space: pre-wrap;
            position: relative;
            animation: fadeIn 0.3s ease;
            font-size: 1rem;
        }
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(8px); }
            to { opacity: 1; transform: translateY(0); }
        }
        .user-message {
            background: linear-gradient(135deg, #6a11cb, #2575fc);
            color: white;
            align-self: flex-end;
            border-bottom-right-radius: 6px;
        }
        .bot-message {
            background: rgba(30, 30, 50, 0.8);
            color: #f0f4ff;
            align-self: flex-start;
            border-bottom-left-radius: 6px;
        }
        .typing-indicator {
            align-self: flex-start;
            background: rgba(30, 30, 50, 0.8);
            color: #d0d8ff;
            padding: 14px 18px;
            border-radius: 20px;
            border-bottom-left-radius: 6px;
            font-size: 1rem;
            display: flex;
            align-items: center;
            gap: 6px;
        }
        .typing-indicator span {
            display: inline-block;
            width: 8px;
            height: 8px;
            background: #a18cd1;
            border-radius: 50%;
            animation: bounce 1.4s infinite ease-in-out;
        }
        .typing-indicator span:nth-child(2) { animation-delay: 0.2s; }
        .typing-indicator span:nth-child(3) { animation-delay: 0.4s; }
        @keyframes bounce {
            0%, 100% { transform: translateY(0); }
            50% { transform: translateY(-6px); }
        }
        .input-area {
            display: flex;
            padding: 14px;
            background: rgba(0, 0, 0, 0.25);
            gap: 10px;
        }
        .input-area input {
            flex: 1;
            padding: 14px 20px;
            border: none;
            border-radius: 30px;
            outline: none;
            font-size: 1rem;
            background: rgba(255, 255, 255, 0.92);
            color: #222;
            min-width: 0;
        }
        .input-area button {
            background: linear-gradient(135deg, #6a11cb, #2575fc);
            color: white;
            border: none;
            border-radius: 30px;
            padding: 0 26px;
            font-weight: 600;
            cursor: pointer;
            transition: opacity 0.2s;
            white-space: nowrap;
        }
        .input-area button:hover:not(:disabled) {
            opacity: 0.9;
        }
        .input-area button:disabled {
            opacity: 0.6;
            cursor: not-allowed;
        }

        /* Mobile optimization */
        @media (max-width: 600px) {
            .chat-container {
                height: 88vh;
                border-radius: 20px;
            }
            .header h1 {
                font-size: 1.4rem;
            }
            .header p {
                font-size: 0.85rem;
            }
            .message {
                max-width: 92%;
                font-size: 0.95rem;
                padding: 12px 16px;
            }
            .typing-indicator {
                font-size: 0.95rem;
                padding: 12px 16px;
            }
            .input-area {
                padding: 12px;
            }
            .input-area input {
                font-size: 0.95rem;
                padding: 12px 16px;
            }
            .input-area button {
                padding: 0 20px;
                font-size: 0.95rem;
            }
        }

        /* Prevent overflow on long words */
        .message {
            overflow-wrap: break-word;
            word-break: break-word;
        }
    </style>
</head>
<body>
    <div class="chat-container">
        <div class="header">
            <h1>🧠 AlmetPT AI</h1>
            <p>Спросите — я отвечу из базы знаний</p>
        </div>
        <div class="chat-messages" id="chatMessages">
            <div class="message bot-message">Привет! Задайте ваш вопрос, и я постараюсь помочь.</div>
        </div>
        <div class="input-area">
            <input type="text" id="userInput" placeholder="Введите ваш вопрос..." autocomplete="off" maxlength="300">
            <button id="sendBtn" onclick="sendMessage()">Отправить</button>
        </div>
    </div>

    <script>
        const chatMessages = document.getElementById('chatMessages');
        const userInput = document.getElementById('userInput');
        const sendBtn = document.getElementById('sendBtn');

        function addMessage(text, isUser) {
            const messageDiv = document.createElement('div');
            messageDiv.classList.add('message');
            messageDiv.classList.add(isUser ? 'user-message' : 'bot-message');
            messageDiv.textContent = text;
            chatMessages.appendChild(messageDiv);
            chatMessages.scrollTop = chatMessages.scrollHeight;
        }

        function showThinking() {
            const typingDiv = document.createElement('div');
            typingDiv.classList.add('typing-indicator');
            typingDiv.id = 'thinkingIndicator';
            typingDiv.innerHTML = 'Думаю... <span></span><span></span><span></span>';
            chatMessages.appendChild(typingDiv);
            chatMessages.scrollTop = chatMessages.scrollHeight;
        }

        function hideThinking() {
            const indicator = document.getElementById('thinkingIndicator');
            if (indicator) indicator.remove();
        }

        function typeMessage(text, callback) {
            const messageDiv = document.createElement('div');
            messageDiv.classList.add('message', 'bot-message');
            messageDiv.textContent = '';
            chatMessages.appendChild(messageDiv);
            chatMessages.scrollTop = chatMessages.scrollHeight;

            let i = 0;
            const speed = 25 + Math.random() * 25; // 25–50 ms per char

            function type() {
                if (i < text.length) {
                    messageDiv.textContent += text.charAt(i);
                    i++;
                    chatMessages.scrollTop = chatMessages.scrollHeight;
                    setTimeout(type, speed);
                } else {
                    if (callback) callback();
                }
            }
            type();
        }

        async function sendMessage() {
            const text = userInput.value.trim();
            if (!text) return;

            addMessage(text, true);
            userInput.value = '';
            sendBtn.disabled = true;
            userInput.blur();
            showThinking();

            try {
                const response = await fetch('/ask', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ question: text })
                });
                const data = await response.json();
                hideThinking();
                typeMessage(data.answer);
            } catch (error) {
                hideThinking();
                addMessage("❌ Ошибка: не удалось получить ответ. Проверьте соединение.", false);
            }

            sendBtn.disabled = false;
            userInput.focus();
        }

        userInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                e.preventDefault();
                sendMessage();
            }
        });

        userInput.focus();
    </script>
</body>
</html>
'''

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/ask', methods=['POST'])
def ask():
    try:
        data = request.get_json()
        question = data.get('question', '').strip()
        if not question:
            return jsonify({"answer": "Пожалуйста, введите вопрос."})
        result = faq_bot.find_best_answer(question)
        return jsonify({"answer": result["answer"]})
    except Exception as e:
        logger.error(f"Ошибка в /ask: {e}")
        return jsonify({"answer": "Произошла внутренняя ошибка. Попробуйте позже."}), 500

def main():
    global faq_bot
    print("🚀 Инициализация AlmetPT AI...")
    faq_bot = FAQBot(
        model_dir="./faq_model",
        csv_path="data_conter.csv",
        template_path="shablon.csv"
    )
    print("✅ Модель и данные загружены!")
    print("\n🌐 Запуск веб-сервера...")
    print("   💻 Локальный доступ: http://127.0.0.1:5000")
    
    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
    except Exception:
        ip = "127.0.0.1"
    finally:
        s.close()
    
    print(f"   📱 Доступ в локальной сети: http://{ip}:5000")
    print("\nОстановка: Ctrl+C\n")

    app.run(host='0.0.0.0', port=5000, debug=False)

if __name__ == "__main__":
    main()