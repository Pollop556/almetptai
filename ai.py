import os
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, util
import csv
import re
import logging
import telebot
from telebot import types

# Настройка логирования
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

class FAQTelegramBot:
    def __init__(self, model_dir="./model", csv_path="data_new.csv", template_path="shablon.csv"):
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
        model_name = 'sentence-transformers/all-MiniLM-L6-v2'
        local_model_path = os.path.join(self.model_dir, 'all-MiniLM-L6-v2')
        
        try:
            logger.info("Пытаюсь загрузить модель из локальной папки...")
            self.model = SentenceTransformer(local_model_path)
            logger.info("✅ Модель успешно загружена из локальной папки!")
        except:
            logger.info("❌ Локальная модель не найдена. Скачиваю модель...")
            try:
                self.model = SentenceTransformer(model_name)
                self.model.save(local_model_path)
                logger.info(f"✅ Модель сохранена в: {local_model_path}")
            except Exception as e:
                logger.error(f"❌ Ошибка при скачивании модели: {e}")
                raise
    
    def load_data(self):
        try:
            logger.info(f"Пытаюсь загрузить данные из {self.csv_path}...")
            try:
                self.df = pd.read_csv(self.csv_path, header=None, names=['question', 'answer'])
                logger.info("✅ Данные загружены стандартным способом")
            except:
                self.df = pd.read_csv(
                    self.csv_path, 
                    header=None, 
                    names=['question', 'answer'],
                    quoting=csv.QUOTE_ALL,
                    escapechar='\\'
                )
                logger.info("✅ Данные загружены с обработкой кавычек")
            logger.info(f"📊 Загружено {len(self.df)} вопросов-ответов")
        except Exception as e:
            logger.error(f"❌ Ошибка при загрузке CSV файла: {e}")
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
                    logger.warning(f"⚠️ Пропущена строка {i+1}: неверное количество полей - {row}")
        self.df = pd.DataFrame(data, columns=['question', 'answer'])
        logger.info(f"✅ Данные загружены вручную. Загружено {len(self.df)} записей")
        if len(self.df) == 0:
            raise Exception("Не удалось загрузить данные")
    
    def load_templates(self):
        try:
            logger.info(f"Загружаю шаблоны из {self.template_path}...")
            with open(self.template_path, 'r', encoding='utf-8') as file:
                content = file.read().splitlines()
                for line in content:
                    if ',' in line:
                        parts = line.split(',', 1)
                        if len(parts) >= 2:
                            key = parts[0].strip()
                            value = parts[1].strip()
                            if key.startswith('[') and key.endswith(']'):
                                key = key[1:-1]
                            if value.startswith('"') and value.endswith('"'):
                                value = value[1:-1]
                            self.templates[key] = value
            logger.info(f"✅ Загружено {len(self.templates)} шаблонов")
        except Exception as e:
            logger.error(f"❌ Ошибка при загрузке шаблонов: {e}")
    
    def replace_templates(self, text):
        if not self.templates:
            return text
        templates_found = re.findall(r'\[(.*?)\]', text)
        if not templates_found:
            return text
        result = text
        for template in templates_found:
            if template in self.templates:
                replacement = self.templates[template]
                result = result.replace(f"[{template}]", replacement)
                logger.debug(f"Заменен шаблон '[{template}]' -> '{replacement}'")
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
    
    def find_best_answer(self, query, top_k=1):
        query_embedding = self.model.encode(query, convert_to_tensor=True, normalize_embeddings=True)
        cos_scores = util.cos_sim(query_embedding, self.corpus_embeddings)[0]
        top_score, top_idx = torch.max(cos_scores, dim=0)
        original_answer = self.df.iloc[top_idx.item()]['answer']
        processed_answer = self.replace_templates(original_answer)
        return {
            'question': self.df.iloc[top_idx.item()]['question'],
            'answer': processed_answer,
            'score': top_score.item()
        }

# === TELEBOT ИНТЕГРАЦИЯ ===

# Токен вашего Telegram-бота
BOT_TOKEN = "8450391232:AAGtconwAu_Lig4gre6k05NJgXWukh6NIHU"

if BOT_TOKEN == "YOUR_BOT_TOKEN_HERE":
    raise ValueError("❌ Пожалуйста, установите ваш токен бота в переменной BOT_TOKEN")

# Создаём бота
bot = telebot.TeleBot(BOT_TOKEN)

# Глобальный экземпляр FAQ бота
faq_bot = None

@bot.message_handler(commands=['start'])
def send_welcome(message):
    welcome_text = """
🤖 Добро пожаловать в FAQ бот!

Просто напишите ваш вопрос, и я найду на него ответ в базе знаний.

Примеры вопросов:
• Как узнать расписание?
• Где найти контакты?
• Как поступить в университет?
"""
    bot.reply_to(message, welcome_text)

@bot.message_handler(commands=['help'])
def send_help(message):
    help_text = """
📖 Помощь по боту:

• Просто напишите ваш вопрос в чат
• Бот найдет самый подходящий ответ из базы знаний
• Ответы автоматически форматируются с актуальными ссылками

Если ответ не точный, попробуйте перефразировать вопрос.
"""
    bot.reply_to(message, help_text)

@bot.message_handler(func=lambda message: True)
def handle_message(message):
    try:
        user_message = message.text
        logger.info(f"User {message.from_user.id}: '{user_message}'")

        # Имитация "печатания" (необязательно, но приятно)
        bot.send_chat_action(message.chat.id, 'typing')

        result = faq_bot.find_best_answer(user_message)
        
        if result['score'] > 0.3:
            response = f"💡 {result['answer']}"
            if result['score'] < 0.5:
                response += "\n\n⚠️ Если этот ответ не подходит, попробуйте перефразировать вопрос."
        else:
            response = "❌ К сожалению, я не нашел подходящего ответа в базе знаний. Попробуйте перефразировать вопрос или обратитесь в поддержку."

        bot.reply_to(message, response)
        logger.info(f"Ответ отправлен. Score: {result['score']:.4f}")

    except Exception as e:
        logger.error(f"Ошибка при обработке сообщения: {e}")
        bot.reply_to(message, "❌ Произошла ошибка при обработке запроса. Попробуйте позже.")

def main():
    global faq_bot
    print("🚀 Инициализация FAQ бота...")
    faq_bot = FAQTelegramBot(
        model_dir="./faq_model",
        csv_path="data_new.csv",
        template_path="shablon.csv"
    )
    print("✅ FAQ бот инициализирован!")
    print("🤖 Запускаю Telegram бота...")
    bot.infinity_polling()

if __name__ == "__main__":
    main()