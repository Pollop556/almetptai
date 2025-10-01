# bot.py
import telebot
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import re
import csv
import os
from datetime import datetime
import logging

# ==================== КОНФИГУРАЦИЯ ====================
BOT_TOKEN = "8450391232:AAGtconwAu_Lig4gre6k05NJgXWukh6NIHU"  # Замени на свой токен
MODEL_PATH = "./my_rugpt3_finetuned"  # Путь к обученной модели
SHABLON_FILE = "shablon.csv"

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("bot.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Загрузка шаблонов из CSV
def load_shablons():
    shablons = {}
    try:
        with open(SHABLON_FILE, 'r', encoding='utf-8') as file:
            reader = csv.reader(file)
            for row in reader:
                if len(row) >= 2:
                    # Убираем кавычки если есть
                    key = row[0].strip().strip('"')
                    value = row[1].strip().strip('"')
                    shablons[key] = value
        logger.info(f"✅ Загружено {len(shablons)} шаблонов")
        return shablons
    except Exception as e:
        logger.error(f"❌ Ошибка загрузки шаблонов: {e}")
        return {}

# Загрузка модели
def load_model():
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        
        logger.info(f"✅ Модель загружена на устройство: {device}")
        return tokenizer, model, device
    except Exception as e:
        logger.error(f"❌ Ошибка загрузки модели: {e}")
        return None, None, None

# Функция замены шаблонов
def replace_shablons(text, shablons):
    """Заменяет шаблоны в тексте на значения из словаря"""
    if not text:
        return text
    
    # Ищем все шаблоны в формате [название шаблона]
    pattern = r'\[(.*?)\]'
    matches = re.findall(pattern, text)
    
    if not matches:
        return text
    
    result = text
    for match in matches:
        template_key = f"[{match}]"
        if template_key in shablons:
            # Заменяем шаблон на значение
            result = result.replace(template_key, shablons[template_key])
        else:
            # Удаляем шаблон если он не найден
            result = result.replace(template_key, "").strip()
    
    # Убираем лишние пробелы после удаления шаблонов
    result = re.sub(r'\s+', ' ', result).strip()
    return result

# Генерация ответа моделью
def generate_response(question, tokenizer, model, device, max_length=200):
    try:
        prompt = f"В: {question}\nО:"
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                max_length=max_length,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.2,
                early_stopping=True
            )
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Извлекаем только ответ (часть после "О:")
        answer = generated_text[len(prompt):].strip()
        
        # Обрезаем ответ если он слишком длинный
        if len(answer) > 400:
            answer = answer[:400] + "..."
            
        return answer
    except Exception as e:
        logger.error(f"❌ Ошибка генерации: {e}")
        return "Извините, произошла ошибка при генерации ответа."

# Инициализация бота
bot = telebot.TeleBot(BOT_TOKEN)

# Загрузка данных при старте
SHABLONS = load_shablons()
TOKENIZER, MODEL, DEVICE = load_model()

# Статистика использования
usage_stats = {
    "total_requests": 0,
    "successful_responses": 0,
    "users": set()
}

# ==================== ОБРАБОТЧИКИ СООБЩЕНИЙ ====================
@bot.message_handler(commands=['start', 'help'])
def send_welcome(message):
    welcome_text = """
🤖 Добро пожаловать в чат-бот Альметьевского политехнического техникума!

Я могу помочь вам с информацией о:
• Расписании занятий
• Поступлении и документах
• Отделениях и специальностях
• IT-Кубе и кружках
• Общежитии и стипендиях
• Контактах преподавателей

Просто задайте ваш вопрос, и я постараюсь помочь!

📊 Статистика: /stats
🆘 Помощь: /help
    """
    bot.reply_to(message, welcome_text)
    logger.info(f"👋 Пользователь {message.from_user.id} начал диалог")

@bot.message_handler(commands=['stats'])
def send_stats(message):
    stats_text = f"""
📊 Статистика бота:
• Всего запросов: {usage_stats['total_requests']}
• Успешных ответов: {usage_stats['successful_responses']}
• Уникальных пользователей: {len(usage_stats['users'])}
• Последнее обновление: {datetime.now().strftime('%H:%M:%S')}
    """
    bot.reply_to(message, stats_text)

@bot.message_handler(func=lambda message: True)
def handle_message(message):
    user_id = message.from_user.id
    question = message.text.strip()
    
    # Обновляем статистику
    usage_stats["total_requests"] += 1
    usage_stats["users"].add(user_id)
    
    logger.info(f"❓ Вопрос от {user_id}: {question}")
    
    # Проверяем загружена ли модель
    if TOKENIZER is None or MODEL is None:
        bot.reply_to(message, "⚠️ Модель временно недоступна. Попробуйте позже.")
        return
    
    # Показываем что бот печатает
    bot.send_chat_action(message.chat.id, 'typing')
    
    try:
        # Генерируем ответ моделью
        raw_answer = generate_response(question, TOKENIZER, MODEL, DEVICE)
        
        # Заменяем шаблоны
        final_answer = replace_shablons(raw_answer, SHABLONS)
        
        # Если ответ пустой после замены шаблонов
        if not final_answer or len(final_answer) < 5:
            final_answer = "Извините, не удалось сгенерировать подходящий ответ. Попробуйте переформулировать вопрос."
        
        # Отправляем ответ
        bot.reply_to(message, final_answer)
        usage_stats["successful_responses"] += 1
        
        logger.info(f"✅ Ответ для {user_id}: {final_answer[:100]}...")
        
    except Exception as e:
        error_msg = "❌ Произошла ошибка при обработке вашего запроса. Попробуйте еще раз."
        bot.reply_to(message, error_msg)
        logger.error(f"❌ Ошибка обработки для {user_id}: {e}")

# ==================== ЗАПУСК БОТА ====================
if __name__ == "__main__":
    logger.info("🚀 Запуск телеграм бота...")
    
    if TOKENIZER is None or MODEL is None:
        logger.error("❌ Не удалось загрузить модель! Бот не может быть запущен.")
        exit(1)
        
    if not SHABLONS:
        logger.warning("⚠️ Шаблоны не загружены, ответы будут без замены!")
    
    print("🤖 Бот запущен и готов к работе!")
    print("⏹️ Для остановки нажмите Ctrl+C")
    
    try:
        bot.polling(none_stop=True, interval=0)
    except KeyboardInterrupt:
        logger.info("🛑 Бот остановлен пользователем")
    except Exception as e:
        logger.error(f"❌ Критическая ошибка бота: {e}")