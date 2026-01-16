# System Architecture: HAVEN

This document outlines the architectural principles, organizational structure, and design decisions for the HAVEN (Home Activity Vision & Event Notification) system.

## 1. Core Principles

### Separation of Concerns
The project is strictly divided into four main domains:
- **Backend**: Handles API management, business logic, and Computer Vision (CV) processing.
- **Frontend**: Presentation layer. Initially focused on a Streamlit dashboard for monitoring, with future expansion to Flutter mobile apps.
- **Models**: AI weights and configurations are kept separate from the source code to ensure portability and clean version control.
- **Data**: All runtime data (logs, videos, snapshots) is git-ignored and managed locally.

## 2. Backend Layered Architecture

The backend follows a modular, layered approach to ensure clean code and easy maintenance:

- **`api/` (Presentation Layer)**: Contains HTTP (FastAPI) and WebSocket endpoints. It handles request validation and response formatting.
- **`core/` (Business Logic Layer)**: The "engine" of the system.
    - `capture/`: Manages RTSP streams and frame queuing.
    - `cv/`: Object detection (YOLOv8-Pose) and tracking.
    - `adl/`: Activity recognition logic (Rule-based and Temporal models).
    - `alerts/`: Alert management and notification (Telegram).
    - `privacy/`: Privacy masking and anonymization.
- **`models/` (Data Access Layer)**: Defines SQLAlchemy ORM models for the database and Pydantic schemas for data transfer (DTOs).
- **`services/` (Orchestration Layer)**: Acts as the bridge between `core/` logic and `models/`. It orchestrates complex operations like the video processing pipeline.

## 3. Scalability & Performance

- **Module Independence**: Core modules are designed to be decoupled, allowing for potential future migration to a microservices architecture.
- **Queue System**: Implements a producer-consumer pattern for video capture and processing to prevent frame drops and handle processing spikes.
- **Externalized Configuration**: All system behaviors (camera URLs, ADL rules, alert thresholds) are managed via external YAML files in the `config/` directory.

## 4. Maintainability & Organization

- **Documentation**: Sub-divided into `design/`, `deployment/`, `research/`, and `user-guide/`.
- **Testing**: Explicit separation of `unit/` (logic), `integration/` (components), and `e2e/` (flow) tests.
- **Tooling**: Utility scripts for environment setup, database backups, and performance benchmarking are located in `scripts/`.

## 5. Reference & Standards

HAVEN draws inspiration from the **SMAC** repository structure:
- Maintains a clean `backend/` and `frontend/` split.
- Enhanced `core/` organization for complex CV tasks.
- Weights are stored in `models/weights/` to keep the repository size manageable.
- YAML-based configuration for maximum flexibility without code changes.
