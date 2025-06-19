FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app


# ---- Dependencies -------------------------------------------------
COPY requirements.txt* ./
RUN pip install --upgrade pip && \
    if [ -f requirements.txt ]; then \
        pip install --no-cache-dir -r requirements.txt; \
    fi

# ---- Source code --------------------------------------------------
COPY . .

# ---- Default command ---------------------------------------------
CMD ["python", "scripts/backtest_v1.py"]
