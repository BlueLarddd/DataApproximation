FROM python:3.9-slim

# Устанавливаем рабочую директорию
WORKDIR /app

# Устанавливаем зависимости для matplotlib
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libxext6 \
    libxrender1 \
    libxtst6 \
    libxi6 \
    && rm -rf /var/lib/apt/lists/*

# Копируем зависимости
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копируем программу
COPY A_14.py .
COPY data.csv .

# Создаем папку для сохранения графиков
RUN mkdir -p /app/output && chmod 777 /app/output

# Создаем пользователя
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Отключаем интерактивный режим matplotlib
ENV MPLBACKEND=Agg

CMD ["python", "Слинкина_Ю_В_БИС-24-2.py"]