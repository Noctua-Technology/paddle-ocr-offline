# Paddle OCR Microservice

A microservice for optical character recognition (OCR) powered by PaddleOCR.

## Features

- Fast and accurate text detection and recognition
- RESTful API interface
- Docker containerized deployment
- Support for multiple languages

## Getting Started

### Prerequisites

- Docker
- Docker Compose

### Usage

Build Docker Image:

```bash
docker build --platform linux/amd64 -t ocr-amd . --load
```

### Running Locally

```bash
uv run uvicorn src.app:app --reload
```