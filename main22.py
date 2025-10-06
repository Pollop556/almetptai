import os
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, util
import csv
import re
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
import logging

# Настройка логирования
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

class FAQTelegramBot:
    def __init__(self, model_dir="./model", csv_path="data_new.csv", template_path="shablon.csv"):
        """
        Инициализация чат-бота для Telegram
        
        Args:
            model_dir (str): Папка для сохранения модели
            csv_path (str): Путь к CSV файлу с вопросами-ответами
            template_path (str): Путь к CSV файлу с шаблонами
        """
        self.model_dir = model_dir
        self.csv_path = csv_path
        self.template_path = template_path
        self.model = None
        self.df = None
        self.templates = {}
        self.corpus_embeddings = None
        
        # Создаем папку для модели если её нет
        os.makedirs(model_dir, exist_ok=True)
        
        # Загружаем или скачиваем модель
        self.load_or_download_model()
        
        # Загружаем данные
        self.load_data()
        
        # Загружаем шаблоны
        self.load_templates()
        
        # Векторизуем вопросы из базы знаний
        self.encode_corpus()
    
    def load_or_download_model(self):
        """Загружает модель из локальной папки или скачивает если её нет"""
        model_name = 'sentence-transformers/all-MiniLM-L6-v2'
        local_model_path = os.path.join(self.model_dir, 'all-MiniLM-L6-v2')
        
        try:
            logger.info("Пытаюсь загрузить модель из локальной папки...")
            self.model = SentenceTransformer(local_model_path)
            logger.info("✅ Модель успешно загружена из локальной папки!")
        except:
            logger.info("❌ Локальная модель не найдена. Скачиваю модель...")
            
            try:
                # Скачиваем модель
                self.model = SentenceTransformer(model_name)
                
                # Сохраняем модель в локальную папку для будущего использования
                self.model.save(local_model_path)
                logger.info(f"✅ Модель сохранена в: {local_model_path}")
                
            except Exception as e:
                logger.error(f"❌ Ошибка при скачивании модели: {e}")
                raise
    
    def load_data(self):
        """Загружает данные из CSV файла с обработкой ошибок формата"""
        try:
            logger.info(f"Пытаюсь загрузить данные из {self.csv_path}...")
            
            try:
                # Способ 1: Стандартное чтение
                self.df = pd.read_csv(self.csv_path, header=None, names=['question', 'answer'])
                logger.info("✅ Данные загружены стандартным способом")
            except:
                # Способ 2: Чтение с обработкой кавычек
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
        """Альтернативный способ чтения CSV вручную"""
        try:
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
                
        except Exception as e:
            logger.error(f"❌ Критическая ошибка при чтении файла: {e}")
            raise
    
    def load_templates(self):
        """Загружает шаблоны из CSV файла"""
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
        """Заменяет шаблоны в квадратных скобках на реальные значения"""
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
        """Векторизует все вопросы из базы знаний"""
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
        """
        Находит самый релевантный ответ для заданного вопроса
        Возвращает только один лучший результат
        """
        # Кодируем входной вопрос
        query_embedding = self.model.encode(query, convert_to_tensor=True, normalize_embeddings=True)
        
        # Вычисляем косинусное сходство
        cos_scores = util.cos_sim(query_embedding, self.corpus_embeddings)[0]
        
        # Находим самый похожий вопрос
        top_score, top_idx = torch.max(cos_scores, dim=0)
        
        original_answer = self.df.iloc[top_idx.item()]['answer']
        
        # Заменяем шаблоны в ответе
        processed_answer = self.replace_templates(original_answer)
        
        result = {
            'question': self.df.iloc[top_idx.item()]['question'],
            'answer': processed_answer,
            'score': top_score.item()
        }
        
        return result

# Глобальная переменная для бота
faq_bot = None

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик команды /start"""
    welcome_text = """
🤖 Добро пожаловать в FAQ бот!

Просто напишите ваш вопрос, и я найду на него ответ в базе знаний.

Примеры вопросов:
• Как узнать расписание?
• Где найти контакты?
• Как поступить в университет?
"""
    await update.message.reply_text(welcome_text)

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик команды /help"""
    help_text = """
📖 Помощь по боту:

• Просто напишите ваш вопрос в чат
• Бот найдет самый подходящий ответ из базы знаний
• Ответы автоматически форматируются с актуальными ссылками

Если ответ не точный, попробуйте перефразировать вопрос.
"""
    await update.message.reply_text(help_text)

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик текстовых сообщений"""
    try:
        user_message = update.message.text
        
        # Показываем что бот печатает
        await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")
        
        # Ищем лучший ответ
        result = faq_bot.find_best_answer(user_message)
        
        # Форматируем ответ
        if result['score'] > 0.3:  # Минимальный порог уверенности
            response = f"💡 {result['answer']}"
            
            # Добавляем подсказку если уверенность низкая
            if result['score'] < 0.5:
                response += "\n\n⚠️ Если этот ответ не подходит, попробуйте перефразировать вопрос."
        else:
            response = "❌ К сожалению, я не нашел подходящего ответа в базе знаний. Попробуйте перефразировать вопрос или обратитесь в поддержку."
        
        # Отправляем ответ
        await update.message.reply_text(response)
        
        # Логируем запрос
        logger.info(f"User {update.effective_user.id}: '{user_message}' -> Score: {result['score']:.4f}")
        
    except Exception as e:
        logger.error(f"Ошибка при обработке сообщения: {e}")
        await update.message.reply_text("❌ Произошла ошибка при обработке запроса. Попробуйте позже.")

async def error_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик ошибок"""
    logger.error(f"Ошибка: {context.error}")
    if update and update.message:
        await update.message.reply_text("❌ Произошла непредвиденная ошибка. Попробуйте позже.")

def main():
    """Основная функция для запуска Telegram бота"""
    global faq_bot
    
    # Токен бота (замените на ваш)
    BOT_TOKEN = "8450391232:AAGtconwAu_Lig4gre6k05NJgXWukh6NIHU"
    
    if BOT_TOKEN == "YOUR_BOT_TOKEN_HERE":
        print("❌ Пожалуйста, установите ваш токен бота в переменной BOT_TOKEN")
        return
    
    try:
        # Инициализируем FAQ бота
        print("🚀 Инициализация FAQ бота...")
        faq_bot = FAQTelegramBot(
            model_dir="./faq_model",
            csv_path="data_new.csv",
            template_path="shablon.csv"
        )
        print("✅ FAQ бот инициализирован!")
        
        # Создаем приложение Telegram
        application = Application.builder().token(BOT_TOKEN).build()
        
        # Добавляем обработчики
        application.add_handler(CommandHandler("start", start_command))
        application.add_handler(CommandHandler("help", help_command))
        application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
        application.add_error_handler(error_handler)
        
        # Запускаем бота
        print("🤖 Запускаю Telegram бота...")
        application.run_polling(allowed_updates=Update.ALL_TYPES)
        
    except Exception as e:
        logger.error(f"Критическая ошибка при запуске: {e}")
        print(f"❌ Не удалось запустить бота: {e}")

if __name__ == "__main__":
    main()