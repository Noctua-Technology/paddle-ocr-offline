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

Send an image to the OCR endpoint:

```bash
curl -X POST http://localhost:8000/ocr \
    -F "image=@path/to/image.png"
```