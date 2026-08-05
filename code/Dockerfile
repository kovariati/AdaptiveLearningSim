FROM python:3.13.5-slim
WORKDIR /work
COPY requirements-lock.txt .
RUN pip install --no-cache-dir -r requirements-lock.txt
COPY . .
ENV MPLCONFIGDIR=/tmp/matplotlib
CMD ["python", "-m", "pytest", "-q", "test_taskenv_revised.py"]
