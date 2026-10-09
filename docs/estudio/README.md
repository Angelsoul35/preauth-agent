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
4. [Evaluación con datos reales](04-evaluacion-datos-reales.md) — cómo validamos el modelo
   de triaje aprendido sobre un dataset médico real: objetivo, variables, fuga de
   información, métricas con fórmulas (macro-F1 / ROC-AUC), validación cruzada y held-out,
   sobre/subajuste, y el control de etiquetas barajadas, con un ejemplo trabajado.

## Cómo correrlo mientras estudias

```bash
make install
make demo       # corre los casos de ejemplo (aprobar / pendiente / negar)
make test       # 46 pruebas
make mcp        # levanta el servidor MCP con las herramientas
make benchmark  # evalúa el modelo de triaje aprendido sobre datos reales
```

El concepto central: **un agente que hace el primer filtro de una autorización médica con
herramientas, donde la decisión es por reglas (explicable) y siempre la confirma un humano,
y junto a ellas un modelo aprendido de triaje validado sobre datos médicos reales.**
