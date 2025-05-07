FROM python:3.13-slim

WORKDIR /app

RUN pip3 install --no-cache torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip3 install --no-cache -r requirements.txt

ENV WEIGHTS_PATH=efficientnet-b0_checkpoint.pth

COPY app.py efficientnet-b0_checkpoint.pth ./

EXPOSE 8081

CMD ["python3", "app.py"]
