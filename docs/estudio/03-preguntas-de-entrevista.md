# 3. Preguntas de entrevista

## Nivel junior / media

**¿Qué hace PreAuth Agent en una frase?**
Hace el primer filtro de una autorización médica con herramientas (elegibilidad, cobertura,
documentos, criterios), recomienda un resultado y deja que un humano decida.

**¿Qué es una "herramienta" en este contexto?**
Una función pequeña y enfocada que el agente llama para obtener un dato o hacer una
verificación. Aquí son cuatro, una por cada chequeo.

**¿Qué es MCP?**
Un estándar para exponer herramientas a modelos o agentes. Nuestro servidor MCP ofrece las
mismas cuatro herramientas, así cualquier cliente MCP las puede usar.

## Nivel media / senior

**¿Por qué la decisión es por reglas y no del LLM?**
Porque una decisión en salud debe ser explicable, reproducible y auditable. El LLM no da eso
por defecto. Las reglas sí, y un clínico puede revisarlas. El LLM solo redacta el texto.

**¿Cómo garantizas que el agente no aprueba solo?**
El agente solo recomienda y pone estado `pending_human_approval`. El estado final lo fija el
paso de aprobación con el nombre del revisor, y el servicio rechaza aprobar algo que no esté
esperando aprobación. Hay una prueba que lo verifica (`never_auto_approves`).

**¿Por qué exponer las herramientas por MCP y no solo usarlas internamente?**
Para tener una sola fuente de verdad y poder reusarlas desde otros clientes (un IDE, otro
agente) sin reimplementarlas. Evita que las reglas se desincronicen entre sistemas.

## Nivel senior / diseño

**¿Dónde está el límite de confianza en este diseño?**
Entre las verificaciones automáticas (deterministas, auditables) y la decisión humana. El
LLM queda fuera de la ruta de decisión: solo texto. Ese límite es el corazón del proyecto.

**¿Qué le falta para producción?**
Cargar los criterios desde fuentes de guías clínicas mantenidas, registrar cada decisión
para auditoría, autenticación y roles (quién puede aprobar), y conectores reales al sistema
de afiliados y de beneficios.

## Reto para ti

* Agrega una herramienta `check_prior_authorizations` que evite duplicados.
* Agrega registro de auditoría: cada decisión y cada aprobación con marca de tiempo y
  usuario.
* Agrega un nivel de "revisión por par" para casos de alto costo (doble aprobación humana).
