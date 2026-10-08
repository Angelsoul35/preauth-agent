# Guía de estudio — PreAuth Agent

Esta guía está en español y explica el proyecto para que lo estudies y lo sustentes en una
entrevista. El código y la documentación técnica están en inglés; aquí va el *por qué*.

## Orden de lectura

1. [Agentes y herramientas (MCP)](01-agentes-y-herramientas.md) — qué es un agente con
   herramientas y cómo se exponen por MCP.
2. [Decisiones seguras y humano en el ciclo](02-decisiones-seguras.md) — por qué la decisión
   es basada en reglas (no del LLM) y por qué siempre decide un humano.
3. [Preguntas de entrevista](03-preguntas-de-entrevista.md) — preguntas típicas con
   respuestas.

## Cómo correrlo mientras estudias

```bash
make install
make demo     # corre los casos de ejemplo (aprobar / pendiente / negar)
make test     # 27 pruebas
make mcp      # levanta el servidor MCP con las herramientas
```

El concepto central: **un agente que hace el primer filtro de una autorización médica con
herramientas, pero donde la decisión es por reglas (explicable) y siempre la confirma un
humano.**
