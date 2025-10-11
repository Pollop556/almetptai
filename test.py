import os
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer


MODEL_NAME = "cointegrated/rut5-base-paraphraser"
MODEL_DIR = "./rut5-base-paraphaser"


def load_model():
    if not os.path.exists(MODEL_DIR):
        print("Модель не найдена, загружаем...")
        model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)
        tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
        # Сохраняем модель локально
        model.save_pretrained(MODEL_DIR)
        tokenizer.save_pretrained(MODEL_DIR)
        print("Модель загружена и сохранена.")
    else:
        print("Загружаем модель из локальной папки...")
        model = T5ForConditionalGeneration.from_pretrained(MODEL_DIR)
        tokenizer = T5Tokenizer.from_pretrained(MODEL_DIR)
    return model, tokenizer


def paraphrase(
    text,
    model,
    tokenizer,
    num_beams=6,
    max_length=None,
    min_length=10,
    do_sample=True,
    top_k=50,
    top_p=0.9,
    temperature=1.0,
    repetition_penalty=1.2,
    encoder_no_repeat_ngram_size=4,
    length_penalty=1.0,
    early_stopping=True
):
    inputs = tokenizer(text, return_tensors="pt", padding=True)
    max_len = max_length or (int(inputs.input_ids.shape[1] * 1.5) + 10)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_len,
            min_length=10,
            num_beams=8,
            do_sample=False,
            repetition_penalty=1.2,
            no_repeat_ngram_size=4,
            length_penalty=1.0,
            early_stopping=True
        )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def chat():
    model, tokenizer = load_model()
    model.eval()
    print("Введите текст для перефразирования (выход - пустая строка):")
    while True:
        text = input("Вы: ")
        if not text.strip():
            break
        paraphrased = paraphrase(text, model, tokenizer)
        print("Нейросеть:", paraphrased)


if __name__ == "__main__":
    chat()
