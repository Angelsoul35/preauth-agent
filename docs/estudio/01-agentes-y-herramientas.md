# 1. Agentes y herramientas (MCP)

## Qué es un agente con herramientas

Un **agente** es un programa que, para lograr un objetivo, usa **herramientas**: funciones
que consultan datos o ejecutan acciones. En vez de que una sola función gigante lo haga
todo, el agente llama herramientas pequeñas y enfocadas y junta sus resultados.

En PreAuth Agent, las herramientas (`preauth/tools/`) son cuatro:

* `check_eligibility` — ¿el afiliado está activo? ¿qué plan tiene?
* `check_coverage` — ¿el plan cubre el procedimiento?
* `check_documents` — ¿están todos los documentos requeridos?
* `check_clinical_criteria` — ¿se cumplen los criterios clínicos?

El agente (`preauth/agent.py`) las llama en orden, junta los resultados y aplica una
política para recomendar `approve`, `deny` o `pend`.

> **Concepto — por qué herramientas pequeñas.** Son fáciles de probar, de explicar y de
> reusar. Cada una tiene una entrada y una salida claras. Si una regla cambia, cambias una
> herramienta, no todo el agente.

## MCP (Model Context Protocol)

MCP es un **estándar** para que un modelo o un agente use herramientas externas. En vez de
que cada cliente reimplemente las mismas funciones, un **servidor MCP** las expone y
cualquier cliente MCP (un IDE, otro agente, un asistente de escritorio) las puede llamar.

En `preauth/mcp_server.py` exponemos las mismas cuatro herramientas por MCP:

```bash
python -m preauth.mcp_server   # las sirve por stdio
```

Así hay **una sola fuente de verdad**: el agente y el servidor MCP usan exactamente las
mismas funciones. (Ver [ADR-0004](../adr/0004-tools-over-mcp.md).)

> **Concepto — una sola fuente de verdad.** Si las reglas vivieran en dos lugares (el
> agente y el cliente MCP), con el tiempo se desincronizarían y aparecerían bugs. Con una
> sola definición, eso no pasa.
