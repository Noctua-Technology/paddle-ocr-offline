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

### Sample API Output

```shell
{
  "filename": "french.jpg",
  "language": "fr",
  "extracted_text": "Monsieur Durand, Je vous écris afin de discuter de l'opportunité de collaborer sur un projet de marketing digital. Ayant suivi vos réalisations dans ce domaine, je suis convaincu que ma proposition pourrait répondre aux besoins de votre entreprise...",
  "text_boxes": [
    {
      "text": "Monsieur Durand,",
      "box": [
        [
          46,
          109
        ],
        [
          179,
          109
        ],
        [
          179,
          128
        ],
        [
          46,
          128
        ]
      ]
    }
  ]
}
```