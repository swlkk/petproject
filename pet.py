import os
import json
import wave
import logging
import pyaudio
import fire
import requests
from vosk import Model, KaldiRecognizer
from gtts import gTTS
from pydub import AudioSegment
from pydub.playback import play
from llama_cpp import Llama
from telegram import Update, ForceReply, Audio
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters, CallbackContext

# ТГ токен
TELEGRAM_TOKEN = 'токен вставить сюда'

SYSTEM_PROMPT = "Ты помощник, помоги пожалуйста человеку с ответом на его вопрос"
SYSTEM_TOKEN = 1788
USER_TOKEN = 1404
BOT_TOKEN = 9225
LINEBREAK_TOKEN = 13

ROLE_TOKENS = {
    "user": USER_TOKEN,
    "bot": BOT_TOKEN,
    "system": SYSTEM_TOKEN
}

# Пути
VOSK_MODEL_URL = "https://alphacephei.com/vosk/models/vosk-model-small-ru-0.22.zip"
VOSK_MODEL_DIR = "model"
VOSK_MODEL_PATH = os.path.join(VOSK_MODEL_DIR, "vosk-model-small-ru-0.22")
LLAMA_MODEL_URL = "https://huggingface.co/IlyaGusev/saiga2_7b_gguf/resolve/main/model-q4_K.gguf"
LLAMA_MODEL_PATH = "model-q4_K.gguf"

def download_file(url, dest_path):
    if not os.path.exists(dest_path):
        print(f"Загрузка модели пум-бдум-тррмс {url}...")
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(dest_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Модель скачана {dest_path}")

def download_and_extract_vosk_model(url, extract_to):
    if not os.path.exists(extract_to):
        os.makedirs(extract_to, exist_ok=True)
        zip_path = os.path.join(extract_to, os.path.basename(url))
        print(f"Downloading model from {url}...")
        r = requests.get(url, allow_redirects=True)
        with open(zip_path, 'wb') as f:
            f.write(r.content)
        print("Модель загружаена извлечение")
        import zipfile
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        print("Модель извлечена")

def get_message_tokens(model, role, content):
    message_tokens = model.tokenize(content.encode("utf-8"))
    message_tokens.insert(1, ROLE_TOKENS[role])
    message_tokens.insert(2, LINEBREAK_TOKEN)
    message_tokens.append(model.token_eos())
    return message_tokens

def get_system_tokens(model):
    system_message = {
        "role": "system",
        "content": SYSTEM_PROMPT
    }
    return get_message_tokens(model, **system_message)

def generate_response(user_message, model):
    message_tokens = get_message_tokens(model, role="user", content=user_message)
    role_tokens = [model.token_bos(), BOT_TOKEN, LINEBREAK_TOKEN]
    tokens = get_system_tokens(model) + message_tokens + role_tokens
    generator = model.generate(
        tokens,
        top_k=30,
        top_p=0.9,
        temp=0.2,
        repeat_penalty=1.1
    )
    response = ""
    for token in generator:
        token_str = model.detokenize([token]).decode("utf-8", errors="ignore")
        tokens.append(token)
        if token == model.token_eos():
            break
        response += token_str
    return response.strip()

def handle_message(update: Update, context: CallbackContext):
    user_message = update.message.text
    context.bot.send_chat_action(chat_id=update.effective_chat.id, action='typing')
    bot_response = generate_response(user_message, context.bot_data['model'])
    audio_response = text_to_speech(bot_response)
    context.bot.send_audio(chat_id=update.effective_chat.id, audio=audio_response)

def handle_audio(update: Update, context: CallbackContext):
    audio_file = update.message.voice.get_file()
    audio_file.download('voice_message.ogg')

    # Конвертация ogg в wav
    try:
        audio = AudioSegment.from_ogg('voice_message.ogg')
        audio = audio.set_channels(1).set_frame_rate(16000).set_sample_width(2)
        audio.export('voice_message.wav', format='wav')
        logging.info("Audio converted to wav format")
    except Exception as e:
        logging.error(f"Error converting audio: {e}")
        update.message.reply_text("Ошибка конвертации аудио")
        return

    text = speech_to_text('voice_message.wav')
    if text:
        context.bot.send_message(chat_id=update.effective_chat.id, text=f'Вы сказали: {text}')
        bot_response = generate_response(text, context.bot_data['model'])
        audio_response = text_to_speech(bot_response)
        context.bot.send_audio(chat_id=update.effective_chat.id, audio=audio_response)
    else:
        update.message.reply_text("Не удалось распознать аудио")

def speech_to_text(audio_file):
    try:
        model = Model(VOSK_MODEL_PATH)
        recognizer = KaldiRecognizer(model, 16000)

        with wave.open(audio_file, "rb") as wf:
            if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getframerate() != 16000:
                logging.error(f"Invalid audio format: channels={wf.getnchannels()}, sample width={wf.getsампwidth()}, framerate={wf.getфрamerate()}")
                return ""

            results = []
            while True:
                data = wf.readframes(4000)
                if len(data) == 0:
                    break
                if recognizer.AcceptWaveform(data):
                    result = recognizer.Result()
                    results.append(json.loads(result).get('text', ''))
            final_result = recognizer.FinalResult()
            results.append(json.loads(final_result).get('text', ''))
            text = " ".join(results)
            logging.info(f"Recognized text: {text}")
            return text
    except Exception as e:
        logging.error(f"Error recognizing speech: {e}")
        return ""

def text_to_speech(text):
    tts = gTTS(text=text, lang='ru')
    tts.save("response.mp3")
    return open("response.mp3", "rb")

def start(update: Update, context: CallbackContext):
    update.message.reply_text("Привет, я Сайга, готова помочь с вашим вопросом")

def main():
    # Загрузка модели VOSK
    download_and_extract_vosk_model(VOSK_MODEL_URL, VOSK_MODEL_DIR)

    # Загрузка модели Llama
    download_file(LLAMA_MODEL_URL, LLAMA_MODEL_PATH)

    # Проверка целостности модели и повторная загрузка при необходимости при возможных перерываниях кода
    try:
        # Загрузка модели Llama
        model = Llama(
            model_path=LLAMA_MODEL_PATH,
            n_ctx=2000,
            n_parts=1,
        )
    except ValueError as e:
        print(f"Error loading model: {e}")
        print("Redownloading the model...")
        os.remove(LLAMA_MODEL_PATH)
        download_file(LLAMA_MODEL_URL, LLAMA_MODEL_PATH)
        model = Llama(
            model_path=LLAMA_MODEL_PATH,
            n_ctx=2000,
            n_parts=1,
        )

    # Логи
    logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Создание Updater и передача ему токена бота
    updater = Updater(TELEGRAM_TOKEN, use_context=True)
    dispatcher = updater.dispatcher
    dispatcher.bot_data['model'] = model

    # Обработчик для команды /start
    dispatcher.add_handler(CommandHandler('start', start))

    # Обработчик сообщений
    dispatcher.add_handler(MessageHandler(Filters.text & ~Filters.command, handle_message))

    # Обработчик голосовых сообщений
    dispatcher.add_handler(MessageHandler(Filters.voice, handle_audio))

    # Запуск бота
    updater.start_polling()
    updater.idle()

if __name__ == "__main__":
    fire.Fire(main)
