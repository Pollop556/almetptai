import re
import csv
import sys

def clean_text(text):
    """Очистка текста: удаление лишних пробелов по краям и экранирование кавычек для CSV."""
    text = text.strip()
    # Экранируем двойные кавычки как ""
    text = text.replace('"', '""')
    return text

def parse_line(line):
    """
    Парсит строку вида:
    (1549, '2012-02-29 15:39:21', 'вопрос...', 'автор', 'ответ...', ...)
    Возвращает (question, answer) или None, если не удалось распарсить.
    """
    # Убираем возможные переносы и лишние пробелы
    line = line.strip()
    if not line.startswith('(') or not line.endswith('),'):
        # Попробуем без запятой в конце (последняя запись)
        if line.endswith(');'):
            line = line[:-2] + ')'
        elif line.endswith(')'):
            pass
        else:
            return None

    # Убираем скобки
    content = line[1:-1]  # убираем '(' и ')' или '),'

    # Разделяем по запятым, но только если они НЕ внутри кавычек
    # Используем регулярное выражение для безопасного сплита
    try:
        # Ищем все значения в одинарных кавычках или числа
        parts = re.findall(r"'([^'\\]*(?:\\.[^'\\]*)*)'|(\d+\.?\d*|\d+)", content)
        # parts — список кортежей: либо ('текст', ''), либо ('', 'число')
        values = []
        for quoted, number in parts:
            if quoted != '':
                # Обрабатываем escape-последовательности, если есть (редко в таких данных)
                # В нашем случае, скорее всего, их нет, но на всякий случай:
                quoted = quoted.replace("''", "'")  # если экранирование через ''
                values.append(quoted)
            elif number != '':
                values.append(number)
        # Нам нужны 3-й и 5-й элементы (индексы 2 и 4), если их достаточно
        if len(values) >= 5:
            question = values[2]
            answer = values[4]
            return clean_text(question), clean_text(answer)
        else:
            return None
    except Exception:
        return None

def main(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as f_in, \
         open(output_file, 'w', encoding='utf-8', newline='') as f_out:

        writer = csv.writer(f_out, quoting=csv.QUOTE_ALL)
        # Не пишем заголовок, если не требуется. Если нужно — раскомментируйте:
        # writer.writerow(['Question', 'Answer'])

        buffer = ''
        for line in f_in:
            stripped = line.strip()
            if not stripped:
                continue

            # Пропускаем строки с INSERT
            if stripped.startswith('INSERT INTO'):
                continue

            # Добавляем строку в буфер
            buffer += line

            # Проверяем, завершена ли запись (заканчивается на '),', ');', или ')')
            if stripped.endswith('),') or stripped.endswith(');') or (stripped.endswith(')') and ',' not in stripped):
                # Пытаемся распарсить буфер как одну запись
                result = parse_line(buffer)
                if result:
                    writer.writerow(result)
                buffer = ''  # сбрасываем буфер

        # На случай, если последняя строка не обработалась
        if buffer.strip():
            result = parse_line(buffer)
            if result:
                writer.writerow(result)

    print(f"Готово! Результат сохранён в {output_file}")

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Использование: python extract_qa_to_csv.py <входной_файл.txt> <выходной_файл.csv>")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2]
    main(input_path, output_path)