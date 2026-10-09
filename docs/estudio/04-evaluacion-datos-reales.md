# 4. Evaluación con datos reales

Este capítulo explica, en profundidad, cómo validamos el **modelo de triaje
aprendido** (`preauth/triage.py`) sobre un conjunto de datos **real y público**.
Es la parte más técnica de la guía: aquí viven las fórmulas, el método y un
ejemplo trabajado con los números reales del proyecto.

> **Idea central.** La decisión de autorización sigue siendo por reglas y la
> confirma un humano (capítulo 2). *Junto* a esas reglas añadimos un modelo
> aprendido que estima el riesgo de fraude de una reclamación a partir de la
> información que llega con ella. No reemplaza a las reglas: las complementa y
> permite enviar antes a revisión humana los casos de alto riesgo.

## 4.1 Los datos reales

Usamos el dataset
[`AniketAsla/adaption-medical-necessity-reviews`](https://huggingface.co/datasets/AniketAsla/adaption-medical-necessity-reviews)
del Hugging Face Hub: **35.976 registros reales** de reclamaciones médicas que un
pagador revisó por necesidad médica. Cada registro trae el texto de la
reclamación, algunos campos estructurados y una etiqueta de fraude.

Lo leemos por la **datasets-server rows API** (sin autenticación) y guardamos un
subconjunto **balanceado** bajo `datasets/`: **3.572 reclamaciones** (1.786
fraudulentas y 1.786 legítimas). Balancear 50/50 hace que las dos clases sean
aprendibles y que las métricas se interpreten de forma directa (el azar es 0,5 de
exactitud). El código del lector, el descargador y la derivación del objetivo
está en `preauth/real_data.py`.

## 4.2 El objetivo (target) y cómo se deriva

El objetivo que el modelo aprende a predecir es `fraud_label`, y lo traducimos al
vocabulario del proyecto:

* `fraud_label = 0` → reclamación **legítima** → vía *approve* → clase 0.
* `fraud_label = 1` → reclamación **fraudulenta** → vía *deny / flag* → clase 1.

Así, el "primer filtro" aprendido predice a qué vía pertenece una reclamación que
entra.

> **Por qué no usamos "aprobar si `claim_amount_approved_inr > 0`".** Al
> inspeccionar los datos reales, el monto aprobado es **positivo en el 100 % de
> ambas clases**. Ese objetivo sería degenerado (todo sería "approve") y no
> aportaría señal. Por eso `fraud_label` es el objetivo binario con contenido.

## 4.3 Las variables de entrada (features)

Usamos **solo información disponible al ingreso** de la reclamación, antes de
cualquier adjudicación:

| Variable | Tipo | Representación |
|---|---|---|
| `claim_description` | texto libre | TF-IDF (1–2 n-gramas, `sublinear_tf`) |
| `claim_amount_requested_inr` | entero | logaritmo y estandarización |
| `icd_code_matches_procedure` | booleano | 0/1 (chequeo de codificación al ingreso) |
| `provider_blacklist_flag` | booleano | 0/1 (chequeo de registro del prestador) |

**TF-IDF** (Term Frequency – Inverse Document Frequency) convierte el texto en
números: pondera cada término por su frecuencia en el documento y la rareza del
término en el corpus, de modo que las palabras muy comunes pesan poco y las
distintivas pesan más. Sobre esa matriz entrenamos una **regresión logística**
(regularizada con L2 y con `class_weight="balanced"`). Elegimos regresión
logística a propósito: igual que las reglas, su decisión es **inspeccionable**
(coeficientes por variable), lo que encaja en un dominio donde un revisor debe
poder ver *por qué* se marcó una reclamación.

## 4.4 La fuga de información (leakage) y cómo la evitamos

**Fuga de información** es cuando una variable de entrada contiene, directa o
indirectamente, la respuesta. Un modelo entrenado con fuga parece excelente y es
inútil: no generaliza a datos nuevos porque "hizo trampa".

Excluimos **todos** los campos decididos *durante o después* de la adjudicación:

* `fraud_type` vale `"legitimate"` exactamente cuando `fraud_label = 0`: **define
  la etiqueta**. Como prueba, si lo usáramos como texto, el modelo alcanza
  **ROC-AUC ≈ 1,000** (fuga perfecta). Por eso se excluye.
* `adjudication_reasoning` dice el fallo (`APPROVE` / `NOT SUBSTANTIATED`);
  `fraud_confidence` es la confianza del propio adjudicador; `enhanced_prompt`,
  `enhanced_completion`, `reasoning_trace` y `original_context` son artefactos de
  generación que describen el resultado; `claim_amount_approved_inr` es ya un
  resultado de la adjudicación; `is_phantom` es una marca posterior a la
  investigación.

**Verificación de que ninguna variable "se filtra" por sí sola.** Medimos la
exactitud de la mejor regla de un solo campo:

| Regla de un solo campo | Exactitud |
|---|---|
| `icd_code_matches_procedure = False` → fraude | 0,632 |
| `provider_blacklist_flag = True` → fraude | 0,551 |
| `claim_amount_requested_inr > mediana` → fraude | 0,567 |

La mejor es 0,632, muy por debajo del modelo aprendido (0,705 de exactitud en
CV). Es decir: **ninguna variable separa las clases de forma trivial**; el modelo
combina señales. Esa es la evidencia de que las variables que conservamos no
contienen fuga.

## 4.5 Las métricas, con fórmulas

Trabajamos con una clase **positiva = fraude** (clase 1). De la matriz de
confusión salen cuatro conteos: verdaderos positivos (TP), verdaderos negativos
(TN), falsos positivos (FP) y falsos negativos (FN).

**Exactitud (accuracy)** — fracción de aciertos:

$$\text{accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$

**Precisión y exhaustividad (recall) por clase $c$:**

$$\text{precision}_c = \frac{TP_c}{TP_c + FP_c}, \qquad \text{recall}_c = \frac{TP_c}{TP_c + FN_c}$$

La precisión responde "de lo que marqué como clase $c$, ¿cuánto acerté?"; el
recall, "de todo lo que era clase $c$, ¿cuánto capturé?".

**F1 por clase** — media armónica de precisión y recall (castiga el desequilibrio
entre ambas):

$$F1_c = 2 \cdot \frac{\text{precision}_c \cdot \text{recall}_c}{\text{precision}_c + \text{recall}_c}$$

**macro-F1** — media simple de los F1 de las dos clases (pondera igual ambas
clases, aunque estén desbalanceadas):

$$\text{macro-F1} = \frac{F1_0 + F1_1}{2}$$

**ROC-AUC** — área bajo la curva ROC. Interpretación probabilística:

$$\text{ROC-AUC} = P\big(\text{score}(\text{fraude al azar}) > \text{score}(\text{legítimo al azar})\big)$$

Es decir, la probabilidad de que el modelo dé mayor puntuación de riesgo a una
reclamación fraudulenta tomada al azar que a una legítima tomada al azar.
**0,5 = azar; 1,0 = ranking perfecto.** A diferencia de la exactitud, no depende
de un umbral fijo: mide la calidad del *ordenamiento* por riesgo.

### El baseline que hay que superar

En el subconjunto balanceado, el **baseline de clase mayoritaria** (predecir
siempre la clase más común) da:

* exactitud = 0,500 (mitad y mitad),
* ROC-AUC = 0,500 (azar),
* macro-F1 = 0,333.

El macro-F1 de 0,333 se obtiene así: si predecimos siempre "legítimo" (clase 0)
sobre 1.786 + 1.786 casos, entonces para la clase 0 tenemos precisión
$1786/3572 = 0{,}5$ y recall $1{,}0$, luego $F1_0 = 2\cdot\frac{0{,}5\cdot 1}{1{,}5} = 0{,}667$;
para la clase 1, recall $= 0$ y $F1_1 = 0$. La media es
$(0{,}667 + 0)/2 = 0{,}333$. Cualquier modelo útil debe superar estos tres
valores.

## 4.6 Ejemplo trabajado con números reales

Entrenamos en el 75 % y evaluamos en el **25 % held-out (893 reclamaciones**: 447
legítimas, 446 fraudulentas). La matriz de confusión real (positivo = fraude):

|  | pred. legítimo | pred. fraude |
|---|---|---|
| **real legítimo** | TN = 405 | FP = 42 |
| **real fraude** | FN = 210 | TP = 236 |

Calculemos a mano, para la **clase fraude (1)**:

$$\text{precision}_1 = \frac{236}{236 + 42} = \frac{236}{278} = 0{,}849$$

$$\text{recall}_1 = \frac{236}{236 + 210} = \frac{236}{446} = 0{,}529$$

$$F1_1 = 2 \cdot \frac{0{,}849 \cdot 0{,}529}{0{,}849 + 0{,}529} = \frac{0{,}898}{1{,}378} = 0{,}652$$

Para la **clase legítimo (0)**:

$$\text{precision}_0 = \frac{405}{405 + 210} = 0{,}659, \qquad \text{recall}_0 = \frac{405}{405 + 42} = 0{,}906, \qquad F1_0 = 0{,}763$$

Entonces:

$$\text{macro-F1} = \frac{0{,}763 + 0{,}652}{2} = 0{,}707$$

$$\text{accuracy} = \frac{TP + TN}{n} = \frac{236 + 405}{893} = \frac{641}{893} = 0{,}718$$

Y la **ROC-AUC = 0,786**. Lectura de negocio: el modelo captura algo más de la
mitad del fraude (recall 0,529) con alta precisión al marcarlo (0,849), y ordena
el riesgo muy por encima del azar (AUC 0,786). Para un primer filtro que solo
*prioriza* casos para el revisor humano, es un resultado fuerte y **verificable**,
sin fuga: lejos del sospechoso 1,000 que delataría una filtración de la respuesta.

## 4.7 El método: validación cruzada y held-out

Dos formas complementarias de estimar cómo se comporta el modelo con datos que
**no** vio al entrenar:

* **Partición held-out.** Se reserva un 25 % de los datos, se entrena en el 75 %
  restante y se mide en el 25 % reservado. Es una sola estimación, rápida y
  realista.
* **Validación cruzada estratificada de $k$ pliegues (k-fold).** Se parte el
  conjunto en $k$ pliegues del mismo tamaño manteniendo la proporción de clases
  ("estratificada"). Se entrena $k$ veces, cada vez dejando un pliegue distinto
  para medir, y se promedian los $k$ resultados. Con $k = 5$, cada registro se
  usa 4 veces para entrenar y 1 para medir. Ventaja: la estimación es más
  **estable** y trae una **desviación estándar** entre pliegues, que nos dice si
  el resultado es consistente.

Resultados reales (regresión logística, $C = 0{,}3$):

| Métrica | Baseline mayoritario | Validación cruzada (5-fold) | Held-out (test) |
|---|---|---|---|
| macro-F1 | 0,333 | **0,695** ± 0,010 | **0,707** |
| exactitud | 0,500 | **0,705** | **0,718** |
| ROC-AUC | 0,500 | **0,760** ± 0,015 | **0,786** |

Lo importante no es solo que supere el baseline, sino que **CV y held-out
coinciden** (0,695 vs 0,707 de macro-F1): dos métodos independientes cuentan la
misma historia, señal de que el modelo **generaliza** y no depende de una
partición afortunada. Las desviaciones estándar pequeñas (±0,010, ±0,015)
confirman que es estable entre pliegues.

## 4.8 Sobreajuste y subajuste: la brecha (gap)

* **Sobreajuste (overfitting).** El modelo memoriza el entrenamiento y falla con
  datos nuevos: la métrica de entrenamiento es **muy** superior a la de
  validación. Señal: **brecha grande** entre train y test.
* **Subajuste (underfitting).** El modelo es demasiado simple y ni siquiera
  aprende el entrenamiento: apenas supera el baseline. Señal: métricas bajas en
  *todo*, train incluido.
* **Bien ajustado (well-fit).** Supera el baseline con holgura **y** las métricas
  de train, CV y held-out están cerca entre sí.

La **brecha** se define como:

$$\text{gap} = \text{macro-F1}_{\text{train}} - \text{macro-F1}_{\text{validación}}$$

En nuestro held-out: $\text{gap} = 0{,}744 - 0{,}707 = 0{,}037$. En la CV:
$0{,}750 - 0{,}695 = 0{,}054$. Son brechas **pequeñas**. Sumado a que la CV (0,695)
y el held-out (0,707) concuerdan y a que superamos el baseline (0,333) por mucho,
el veredicto es **bien ajustado**: ni sobreajuste ni subajuste.

> **Nota sobre el 1,000.** Si un modelo satura el entrenamiento a ~1,0 pero su
> held-out también es alto y estable, no hay que "mutilarlo": se juzga por el
> acuerdo CV/held-out y por el control de abajo. Lo inaceptable es un 1,0 que
> *no* generaliza, o uno producto de una fuga (sección 4.4).

### El camino de regularización

La constante $C$ de la regresión logística controla la regularización: $C$ chico
= más regularización = modelo más simple. Al bajar $C$, la brecha se encoge:

| $C$ | macro-F1 (CV) | ROC-AUC (CV) | brecha train/CV |
|---|---|---|---|
| 0,1 | 0,682 | 0,754 | 0,028 |
| 0,3 | 0,695 | 0,760 | 0,054 |
| 1,0 | 0,706 | 0,765 | 0,108 |

Se ve el compromiso clásico: más capacidad ($C = 1{,}0$) sube un poco el
rendimiento pero agranda la brecha; más regularización ($C = 0{,}1$) cierra la
brecha a costa de algo de rendimiento. Elegimos $C = 0{,}3$ como punto equilibrado
y bien ajustado.

## 4.9 El control de etiquetas barajadas

Es la prueba decisiva de que el modelo aprende una **relación real** y no un
artefacto. Barajamos aleatoriamente las etiquetas (rompiendo el vínculo entre
features y objetivo) y reentrenamos con el mismo procedimiento. Si el modelo
"aprendía" algo genuino, ahora **debe colapsar al azar**:

| | ROC-AUC | macro-F1 |
|---|---|---|
| modelo real | 0,760 | 0,695 |
| etiquetas barajadas | **0,511** | **0,515** |

Con las etiquetas barajadas la ROC-AUC cae a 0,511 (prácticamente 0,5 = azar).
Esto demuestra que el 0,760 del modelo real proviene de una relación verdadera
entre la descripción/señales de ingreso y el fraude, no de memorizar ruido ni de
una fuga oculta.

## 4.10 Cómo correrlo

```bash
pip install 'preauth-agent[ml]'
python -m preauth.real_benchmark
```

Imprime el informe completo y reproducible: baseline, validación cruzada,
held-out, separabilidad por campo único, camino de regularización, control de
etiquetas barajadas y el veredicto (bien ajustado / sobreajustado / subajustado).

## Preguntas de entrevista sobre esta parte

**¿Cómo evitaste la fuga de información?**
Usando solo variables de ingreso (texto de la reclamación, monto solicitado, dos
chequeos de ingreso) y excluyendo todo campo decidido en la adjudicación. Lo
verifiqué de dos formas: `fraud_type` usado como feature da AUC 1,000 (por eso se
excluye) y ninguna variable conservada separa las clases sola (mejor regla de un
campo: 0,632 vs 0,705 del modelo).

**¿Cómo sabes que generaliza y no está sobreajustado?**
Porque la validación cruzada (0,695) y el held-out (0,707) coinciden, la brecha
train/test es pequeña (0,037) y un control de etiquetas barajadas colapsa al azar
(AUC 0,511). Los tres hechos juntos descartan sobreajuste, subajuste y fuga.

**¿Por qué macro-F1 y ROC-AUC, y no solo exactitud?**
Porque macro-F1 pondera igual ambas clases (no premia ignorar la minoría) y
ROC-AUC mide la calidad del ordenamiento por riesgo sin depender de un umbral,
que es justo lo que necesita un primer filtro que prioriza casos.
