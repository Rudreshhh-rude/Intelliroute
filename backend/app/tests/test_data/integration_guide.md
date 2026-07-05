# IntelliRoute Enterprise Integration Guide

## Chapter 1: Introduction and System Overview
This guide provides complete specifications for integrating the IntelliRoute Adaptive Knowledge Engine.
IntelliRoute is designed for enterprise grade assistants that require high speed, low cost, and adaptive accuracy.

### 1.1 Architecture Highlights
- Multi-model routing: dynamically chooses model based on query difficulty.
- Dual retrieval routing: switches between Vector DB search and PageIndex structural tree navigation.
- Semantic cache: skips duplicate LLM calls to minimize latency and token billing.

## Chapter 2: Security & Authentication Specification
All connections to IntelliRoute APIs must be signed and authorized.

### 2.1 API Key Management
Enterprise administrators can generate API keys from the admin dashboard. Keys must be passed in the `X-API-Key` HTTP request header.
To revoke a key, update the key status to `revoked` in the DB or click the revoke button.

### 2.2 Role Based Access Control (RBAC)
There are three standard roles supported:
1. `Reader`: Can query search indices and chat.
2. `Writer`: Can upload and parse manuals/documents.
3. `Admin`: Full permissions, including API key generation and system configuration edits.

## Chapter 3: Performance Tuning and Caching
To maintain high responsiveness, caching must be properly configured.

### 3.1 Semantic Caching
The semantic cache uses a similarity threshold of 0.85 (cosine distance). If a query matches a cached question with similarity >= 0.85, the cached response is returned.

### 3.2 Token Bucket Rate Limiter
The rate limiter enforces limits per API key:
- Burst limit: 60 tokens (requests)
- Refill rate: 1 token per second
