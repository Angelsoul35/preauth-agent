# 2. Decisiones seguras y humano en el ciclo

Este proyecto es, sobre todo, un ejemplo de **IA segura en un dominio sensible** (salud).
Dos reglas de diseño lo gobiernan.

## Regla 1: la decisión es por reglas, no del LLM

Una decisión de autorización afecta el cuidado de un paciente y el dinero de un afiliado.
Tiene que ser **explicable, reproducible y auditable**. Un LLM, por defecto, no es ninguna
de las tres: puede dar respuestas distintas a la misma entrada y no se puede explicar del
todo.

Por eso, elegibilidad, cobertura y criterios clínicos son **verificaciones deterministas**
(reglas) en `preauth/tools/`. El agente aplica una política fija sobre sus resultados. El
LLM **solo redacta el texto** de la justificación, nunca decide el resultado. (Ver
[ADR-0002](../adr/0002-rule-based-decisions-not-llm.md).)

> **La trampa a evitar.** Es tentador dejar que el LLM "decida" porque es fácil de
> programar. Pero en salud o finanzas, "lo dijo el modelo" no es una justificación válida.
> La parte que decide debe poder revisarla un clínico o cumplimiento, línea por línea.

## Regla 2: siempre decide un humano

Aunque las reglas sean correctas, un sistema automático no debería **finalizar** una
decisión médica solo. Hay casos límite, contexto que falta y el peso del resultado.

El agente solo **recomienda** (`approve`, `deny`, `pend`) y pone el estado en
`pending_human_approval` (o `needs_info` si faltan documentos). El estado final
(`approved` / `denied`) solo se fija en el paso de aprobación, con el nombre del revisor. El
servicio **se niega** a aprobar una solicitud que no esté esperando aprobación humana. (Ver
[ADR-0003](../adr/0003-human-in-the-loop.md).)

> **Concepto — human-in-the-loop.** El humano está "en el ciclo": el sistema prepara y
> acelera, pero la decisión final es de una persona, con nombre y registro. Es asistencia,
> no automatización total — y en este dominio, esa es la decisión correcta.

## La política de decisión

* no elegible, o no cubierto, o criterios no cumplidos → **deny**
* faltan documentos → **pend** (pedir los documentos)
* todo pasa → **approve** (solo recomendación; un humano confirma)
