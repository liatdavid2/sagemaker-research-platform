FROM python:3.11-slim

ARG TERRAFORM_VERSION=1.9.5

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl unzip ca-certificates git docker.io \
    && rm -rf /var/lib/apt/lists/*

RUN curl -fsSL "https://releases.hashicorp.com/terraform/${TERRAFORM_VERSION}/terraform_${TERRAFORM_VERSION}_linux_amd64.zip" \
    -o /tmp/terraform.zip \
    && unzip /tmp/terraform.zip -d /usr/local/bin \
    && rm /tmp/terraform.zip

RUN pip install --no-cache-dir awscli

WORKDIR /workspace/ui/backend
COPY ui/backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY ui/backend/app.py /workspace/ui/backend/app.py

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
