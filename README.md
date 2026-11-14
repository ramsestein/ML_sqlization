# ML_sqlization (`fnxsql`) — de tu modelo a una consulta SQL que se ejecuta en FakeNex

**Tienes un modelo de machine learning entrenado con datos de DataNex y quieres comprobar que se puede ejecutar dentro
del data warehouse, como una consulta SQL, sin servidores de inferencia.** `fnxsql` lo convierte en reglas if-then
exactas, genera a partir de ellas **una única consulta SQL** (Trino / Athena) que calcula las variables desde las tablas
crudas y puntúa a los pacientes, la ejecuta en [FakeNex](https://fakenex.sliplane.app/guide) (réplica sintética de
DataNex, de solo lectura) y **verifica automáticamente** que el resultado coincide con el modelo original (el error que
añade la conversión, frente al modelo tal como se entrenó: ~10⁻⁶ o menos con float32, 0 con `--precision float64`).

```
tu modelo (.json XGBoost / .joblib scikit-learn)
   │  1. nombres  → cada variable es un código de DataNex (FC_mean, LAB2467_max, ...): de ahí sale cómo calcularla
   ▼
documento if-then exacto (reglas, árboles if-then anidados, capas con ReLU = "si z > 0 entonces z si no 0")
   │  2. compilar → una consulta SQL ≤ 262.144 bytes (AWS Athena): variables desde rc / labs / demographics + el modelo
   │               (parámetros en float32, arrays binarios, sin bucles)
   ▼
FakeNex (Trino) ──► probabilidad por ingreso
   │  3. validar  → variables del servidor = variables recalculadas en local; SQL = modelo = if-then
   ▼
informe (PASA / FALLA) + consulta.sql + if_then.json + comparacion.csv + tiempos
```

No se mide el rendimiento clínico de tu modelo (los datos de FakeNex son sintéticos): se comprueba que **la conversión es
exacta y viable** en el servidor (tamaño, planificación, memoria, tiempos).

## Instalación

Python 3.10 o superior y salida a internet por HTTPS hacia `fakenex.sliplane.app`. Nada más.

```bash
git clone https://github.com/ramsestein/ML_sqlization.git && cd ML_sqlization
python -m venv .venv
.venv/Scripts/activate        # Windows (Linux / macOS: source .venv/bin/activate)
pip install -r requirements.txt
python -m pytest              # tests sin red (~5 s)
```

(`pip install -e .[xgboost]` instala además la orden `fnxsql`; `xgboost` solo hace falta para comparar con la librería
original al evaluar modelos XGBoost.)

## Inicio rápido

**Todo solo, de principio a fin** (entrena modelos de práctica con datos de FakeNex y los evalúa uno a uno):

```bash
python -m fnxsql demo --lotes 1,10,50
```

**Con tu modelo:**

```bash
python -m fnxsql nombres mi_modelo.json                        # ¿siguen sus variables la gramática de DataNex?
python -m fnxsql normalizar mi_modelo.json --contrato mi_contrato.json --salida modelo_fakenex.json   # si no
python -m fnxsql evaluar modelo_fakenex.json --contrato mi_contrato.json --muestra 5 --lotes 1,10,50
```

El informe queda en `salidas/<modelo>/informe.md` y el proceso termina con código 1 si alguna comprobación falla (apto
para CI).

## La condición: variables con nombres de DataNex

El nombre de cada variable dice qué dato es y cómo se agrega. `<est>` = `last`, `first`, `mean`, `std` (muestral),
`min`, `max` o `count` sobre las medidas de la ventana:

| Nombre | Dato | Ejemplo |
|---|---|---|
| `<RC>_<est>` | `rc.result_num` del código `rc_sap_ref` | `FC_mean`, `TAS_last`, `PULSIOX_min` |
| `<LAB>_<est>` | `labs.result_num` del código `lab_sap_ref` (empieza por `LAB`; debe existir en `dic_lab_loinc`) | `LAB2467_max` (creatinina) |
| `<RC>_<RC>_<k>_<est>` | ítem codificado: indicador 0/1 de `rc.result_txt = '<RC>_<k>'` | `ESTADO_CONCIENC_ESTADO_CONCIENC_1_last` |
| `age`, `age_<est>` | edad en años en el momento de evaluación (`demographics.birth_date`); `age_std` = 0 | `age` |
| `sex`, `sex_<k>` | `demographics.sex` o su indicador | `sex_2` |

Lo que no tenga equivalente en DataNex se declara constante en el contrato. **Unidad**: cada ingreso `HOSP` / `HOSP_IQ`.
**Momento de evaluación**: inicio del ingreso + 24 h (o el alta si es antes). **Ventana**: las 48 h anteriores. Se
cambian en el contrato:

```json
{"momento_horas": 24, "ventana_horas": 48,
 "si_falta": null,
 "constantes": {"index_last": 0},
 "renombres": {"P_ART_S": "TAS", "Creatinine": "LAB2467"}}
```

`si_falta`: `null` deja los ausentes como nulos (el modelo decide, p. ej. la rama por defecto de XGBoost); un número los
rellena (como hace muchas veces el código de producción). `renombres` solo lo usa `normalizar`. Ver
`ejemplos/contrato_deterioro.json`, el contrato de un XGBoost real de deterioro clínico (el modelo no se incluye).

## Modelos admitidos y representación SQL

| Modelo | Documento if-then | SQL (`--sql`; el primero es `auto`) |
|---|---|---|
| XGBoost `.json` (gbtree, `binary:logistic`, cortes numéricos) | árboles if-then en float32 | `arrays`, `case` |
| `LogisticRegression` | tarjeta de puntuación | `terminos` |
| `DecisionTreeClassifier`, `RandomForestClassifier` | árboles if-then | `arrays`, `case` |
| `HistGradientBoostingClassifier` (sin categóricas) | árboles if-then | `arrays`, `case` |
| `MLPClassifier` (ReLU) | capas densas; ReLU = si z > 0 entonces z si no 0 | `arrays`, `terminos` |
| `KNeighborsClassifier` (euclídea, manhattan o minkowski p = 1 / 2 con pesos `w`) | núcleo `vecinos`: preproceso, k, ponderación y pesos; **la bolsa es la propia base** | `arrays` (float64) |

Cualquiera de los de scikit-learn puede ir en un `Pipeline` con `SimpleImputer` y / o `StandardScaler` delante: se
traducen como reglas de preproceso (el KNN necesita un `SimpleImputer` o un `si_falta` numérico: un ausente daría
distancias nulas). **No admitidos**: KNN con su bolsa de entrenamiento incrustada (ver abajo), multiclase, regresión,
variables categóricas de HistGradientBoosting o XGBoost, LightGBM.

Por qué así (medido en FakeNex, ver «Lecciones»):

- **arrays** (por defecto en árboles y redes): todos los nodos o pesos van en unos pocos literales binarios (base64) que
  la consulta descodifica una vez (`from_base64` + `from_ieee754_32`, también en Athena) y se recorren con
  `reduce(sequence(1, profundidad), raíz, ...)` / `zip_with`. Sin bucles y con pocos tokens: ~1-2 s de planificación.
- **case** / **terminos**: cada nodo o peso es una expresión. Legible, pero el coste de planificar crece con el modelo:
  `fnxsql` se niega por encima de 12.000 términos (`--forzar` para saltárselo).

### KNN: la bolsa es la base

El "modelo" de un KNN son sus datos de entrenamiento, y no se incrustan en la consulta: no cabrían en Athena (con 195
variables en float32, menos de 250 casos) y son datos de pacientes que no deben viajar. Lo que viaja es **la geometría**
(preproceso, k, ponderación, métrica y pesos por variable); los vecinos se buscan en la base del hospital donde se
ejecuta (en FakeNex, la base original, `patient_ref < 1000000000`). Es la conversión del experimento
(`experimento_fakenex/mortalidad/knn_a_sql.py`, variante «vector», verificada en 620 ingresos con 1,7·10⁻¹⁶ frente a
scikit-learn sobre la misma bolsa), ahora con el contrato y el `sql_variables` del paquete.

Una sola pasada, exacta y sin bucles: `vars` de toda la base (la bolsa) y, en un WITH anidado, las de los pacientes
consultados → un array por ingreso → distancia a cada ingreso de la bolsa de **otros** pacientes con desenlace conocido
(`zip_with` + `reduce`) → `row_number() OVER (PARTITION BY ingreso ORDER BY distancia, vecino) <= k` (empates por
`episode_ref`: determinista) → media o media ponderada por 1 / distancia (con la regla de scikit-learn para distancia 0).
Con pesos `w` (p. ej. una NCA diagonal o pesos de importancia), cada coordenada se multiplica por √w (p = 2) o w (p = 1).
Se queda en float64: no hay arrays de parámetros que encoger y redondear el preproceso podría cambiar qué vecinos entran.

La bolsa se define en el contrato (todo opcional):

```json
{"bolsa": {"etiqueta_sql": "SELECT episode_ref, y FROM {esquema}.mi_tabla WHERE patient_ref {base}",
           "solo_pasado": true,
           "excluir_mismo_paciente": true}}
```

- `etiqueta_sql`: desenlace (0 / 1) de los ingresos de la bolsa; por defecto, muerte durante el ingreso (alta por
  exitus, `adm_disch.mot_ref = 23`). Solo entran en la bolsa los ingresos que devuelve (sin desenlace no hay vecino).
- `solo_pasado` (por defecto `false`, como el experimento): solo vecinos dados de alta antes del momento de evaluación
  del ingreso consultado. En un análisis retrospectivo, sin él un vecino puede ser un ingreso posterior; en producción
  ese futuro no existe, así que es la opción que reproduce el uso real.
- `excluir_mismo_paciente` (por defecto `true`): los ingresos del propio paciente no son vecinos.

Validación (`evaluar`): el if-then con la bolsa de entrenamiento del modelo = scikit-learn (filas sintéticas e ingresos
reales, mismos vecinos; tolerancia 10⁻⁵ porque scikit-learn, con `algorithm="brute"`, calcula la euclídea como
‖x‖² + ‖y‖² − 2x·y y mete hasta ~3·10⁻⁷ en las distancias), y el SQL = if-then sobre la bolsa de FakeNex descargada
(tolerancia 10⁻⁹). La diferencia entre el SQL y el modelo con su bolsa de entrenamiento se informa como aviso: no es un
error de la conversión, es el cambio de bolsa (lo que mide la transportabilidad). El coste crece con ingresos
consultados × bolsa (experimento: 250 pacientes en 44,5 s, < 130 MB).

### Precisión y límite de AWS Athena

DataNex corre en **AWS Athena, que admite consultas de hasta 262.144 bytes**: `fnxsql` no genera nada más grande (con
`--forzar`, hasta el límite del proxy de FakeNex, para probar). Para caber, los parámetros van por defecto en **float32**
(`--precision float32`): un double escrito en texto ocupa ~20 bytes; en binario float32 y base64, 5,3.

| `--precision` | Qué se guarda | Error frente al modelo original (medido) |
|---|---|---|
| `float32` (defecto) | todo parámetro (coeficientes, pesos, sesgos, umbrales, hojas, medias, escalas, imputaciones) en float32; arrays de 4 bytes; cuentas en double | de 4·10⁻⁹ a 2·10⁻⁶ en la probabilidad |
| `float64` | exacto; arrays de 8 bytes | 0 (redondeo de la suma, ~10⁻¹⁶) |
| `int8` | pesos de cada neurona, coeficientes y hojas de cada árbol como enteros de -127 a 127 con una escala float32 por neurona / modelo / árbol; umbrales, sesgos y preproceso en float32 | de 4·10⁻⁴ a 8·10⁻² en la probabilidad (medio, 10⁻⁴ a 3·10⁻³) |

Los errores son los del experimento de mortalidad (620 ingresos de FakeNex, nueve modelos). int8 reduce sobre todo las
redes (red 128+64 de práctica: 85 → 33 KB) y menos los árboles (random forest: 60 → 51 KB), porque los umbrales se quedan
en float32: cuantizarlos mandaría pacientes enteros a otra rama. Úsalo solo si float32 no cabe. Techo aproximado de una
red en int8: ~160.000-180.000 parámetros (float32: ~45.000; float64: ~22.000). Probado con una red de práctica de
152.577 parámetros (40 → 512 → 256 → 1): 224.000 bytes (en float32 serían 827 KB), un paciente en 15,7 s (9,5 s de
planificación, 9 MB de memoria) y SQL idéntico al documento int8.

Los umbrales se redondean según cómo compara el modelo original, para que un valor igual al umbral caiga del mismo lado:

- árboles de scikit-learn: comparan `float32(x) <= t` (convierten x a float32 al entrenar y al predecir); el SQL hace
  `CAST(x AS real) <= t32`, con t32 el mayor float32 ≤ t: las mismas ramas que el original;
- HistGradientBoosting: compara `x <= t` en double y sus umbrales salen de cuantiles, que suelen ser valores observados
  (edad 74,1, creatinina 1,38): t32 = menor float32 ≥ t. Redondear al más cercano cambiaba de rama a esos pacientes;
- XGBoost: ya es float32 (`float32(x) < t`, hojas sumadas en float32) y el SQL lo reproduce con `real` en las dos
  precisiones.

Árboles en binario: nodos en anchura por árbol, de modo que los dos hijos de un nodo quedan seguidos (derecho = izquierdo
\+ 1); variable, dirección del nulo y distancia al hijo izquierdo van en un entero de 1-4 bytes, y el umbral o la hoja, en
float32.

## Validaciones automáticas

`evaluar` comprueba, y deja en el informe:

1. Las variables siguen la gramática; los códigos de laboratorio existen en `dic_lab_loinc` (avisa de los de `rc` sin
   datos en FakeNex: valdrán ausentes).
2. El documento if-then reproduce el modelo original en 300 filas sintéticas con ausentes.
3. La consulta se ejecuta en FakeNex para los pacientes elegidos y cabe en AWS Athena (≤ 262.144 bytes).
4. Las variables calculadas en el servidor son idénticas a las recalculadas **con otra implementación** (pandas, en local)
   desde las tablas crudas de esos pacientes.
5. En esos ingresos reales: el SQL calcula exactamente el documento if-then cuantizado a la precisión elegida (1e-9;
   1e-6 en XGBoost), y el error que añade la precisión frente al modelo original queda medido (máximo y medio): con
   `float32`, tolerancia 1e-5; con `float64`, 1e-9; con `int8` se informa como aviso (pérdida aceptada). El if-then
   exacto = modelo original (1e-9).
6. Opcionalmente, tiempos por tamaño de lote (`--lotes`).

Salidas en `salidas/<modelo>/`: `informe.md` / `informe.json`, `if_then.json`, `consulta.sql`, `comparacion.csv` y
`tablas/` (las filas crudas descargadas).

## Órdenes

| Orden | Para qué | Opciones principales |
|---|---|---|
| `evaluar MODELO` | circuito completo con validaciones e informe | `--contrato`, `--pacientes 1,2` o `--muestra N`, `--sql`, `--precision`, `--lotes`, `--salida`, `--forzar` |
| `nombres MODELO` | comprobar los nombres (y su existencia en FakeNex) | `--contrato`, `--sin-servidor` |
| `normalizar MODELO` | copia del modelo con nombres de DataNex | `--contrato` (renombres), `--nombres metadatos.json`, `--salida` |
| `convertir MODELO` | solo el documento if-then | `--contrato`, `--salida` |
| `sql MODELO` | solo la consulta, sin ejecutarla | `--pacientes` / `--muestra`, `--sql`, `--precision`, `--salida` |
| `practicar` | entrenar modelos de ejemplo con datos de FakeNex | `--modelos`, `--salida` |
| `demo` | `practicar` + `evaluar` de todos | `--modelos`, `--muestra`, `--lotes`, `--sql`, `--precision` |

Globales: `--url` (o `FAKENEX_URL`) y `--usuario` (o `FAKENEX_USER`; por defecto uno aleatorio `fnxsql-xxxxxxxx`).

## Buen uso del servidor

FakeNex es compartido: 384 MB por consulta, 10 min, 4 consultas a la vez. `fnxsql` comprueba la salud del servidor
(`/v1/info`) antes de cada consulta y espera si está arrancando, no genera SQL de más de 262.144 bytes (AWS Athena; con
`--forzar`, 950 KB, por debajo del 1 MiB que admite el proxy),
detiene la batería si una consulta tarda más de 3 min, usa un usuario propio y filtra siempre por `patient_ref` (lo que,
además, no dispara el crecimiento automático de la base). Las consultas van de una en una.

## Resultados de referencia

Todo contra FakeNex (noviembre de 2026). Tiempos de reloj en el cliente, en segundos, con la red incluida; FakeNex es
compartido, así que hay variabilidad entre ejecuciones.

**`python -m fnxsql demo --lotes 1,10,50`** (modelos de práctica entrenados con la base de FakeNex; validación con 3
pacientes de test):

| Modelo | Resultado | SQL | KB (float32) | SQL vs modelo original | 1 pac. | 10 | 50 |
|---|---|---|---|---|---|---|---|
| logistica | PASA | terminos | 18 | 6,4·10⁻⁸ | 4,6 | 5,7 | 5,2 |
| arbol | PASA | arrays | 10 | 3,8·10⁻⁹ | 3,9 | 4,9 | 5,6 |
| random_forest | PASA | arrays | 59 | 3,6·10⁻¹⁰ | 5,8 | 5,8 | 7,4 |
| boosting | PASA | arrays | 55 | 1,3·10⁻⁸ | 6,4 | 6,7 | 8,9 |
| red_pequena | PASA | arrays | 14 | 2,0·10⁻⁸ | 5,3 | 5,1 | 6,8 |
| red_mediana | PASA | arrays | 19 | 3,0·10⁻⁸ | 7,3 | 6,3 | 6,4 |
| red_grande | PASA | arrays | 84 | 2,3·10⁻⁸ | 5,3 | 6,8 | 7,5 |
| xgboost | PASA | arrays | 62 | 2,5·10⁻⁸ (float32 nativo) | 5,1 | 6,2 | 9,0 |

Pesos y diferencias con `--precision float32` (por defecto; con JSON en double eran 18 / 10 / 205 / 178 / 19 / 39 / 285 /
167 KB); tiempos de la medición anterior.

**Un modelo real** (XGBoost de deterioro clínico entrenado con DataNex, 800 árboles, 195 variables; no incluido):
`normalizar` con `ejemplos/contrato_deterioro.json` (30 variables renombradas, mismas predicciones) y `evaluar`: PASA,
204.595 bytes con 7 pacientes (cabe en Athena; con JSON en double eran 537 KB), 195 variables idénticas servidor / local,
SQL vs XGBoost original 5,8·10⁻⁸; 1 / 10 / 50 pacientes en 8,1 / 6,9 / 10,3 s (medición anterior).

**Consultas de referencia** (para extrapolar a DataNex: ejecutándolas allí con N pacientes, el tiempo de un modelo se
estima escalando su tiempo en FakeNex por la razón de las referencias, sobre todo la de «variables»). Reloj (servidor):

| Consulta | 1 pac. | 10 | 50 | 100 | 250 |
|---|---|---|---|---|---|
| `SELECT 1` | 4,5 (1,7) | 4,5 (2,2) | 4,5 (2,3) | 4,9 (2,2) | 4,9 (2,3) |
| ingresos (`episodes` por paciente) | 4,7 (2,3) | 4,1 (1,7) | 5,5 (2,4) | 5,5 (2,0) | 6,7 (1,5) |
| variables de un modelo de mortalidad (47) | 9,3 (6,6) | 5,8 (2,6) | 7,3 (3,3) | 9,3 (3,7) | 14,8 (5,0) |
| variables del XGBoost real (195) | 7,2 (2,6) | 7,4 (3,5) | 7,2 (3,1) | 8,4 (3,2) | 16,2 (3,1) |

**Experimento de desarrollo** (modelos de mortalidad entrenados con 15.000 pacientes sintéticos generados con FakeNex;
versiones SQL finales; diferencia máxima de probabilidad entre el SQL y el modelo original en 620 ingresos de 250
pacientes de FakeNex). Reloj (servidor):

| Versión (float32) | KB 1 pac. | KB 250 pac. | Dif. máx. prob. | 1 pac. | 10 | 50 | 100 | 250 |
|---|---|---|---|---|---|---|---|---|
| Regresión logística | 21 | 40 | 3·10⁻⁷ | 7,1 (3,4) | 7,3 (3,3) | 7,8 (4,7) | 7,8 (3,4) | 8,4 (4,1) |
| Gradient boosting | 49 | 69 | 2·10⁻⁸ | 7,2 (2,7) | 9,1 (3,2) | 11,8 (4,2) | 14,7 (3,5) | 33,3 (10,9) |
| Random forest | 113 | 133 | 4·10⁻⁹ | 8,9 (3,5) | 9,6 (4,0) | 9,6 (3,9) | 17,2 (5,0) | 21,0 (5,0) |
| Red 8 | 25 | 44 | 8·10⁻⁷ | 5,4 (2,8) | 5,9 (2,7) | 6,2 (3,1) | 8,3 (3,8) | 12,0 (5,2) |
| Red 32 | 37 | 57 | 2·10⁻⁶ | 7,0 (2,7) | 6,7 (3,3) | 7,1 (3,0) | 8,3 (4,3) | 9,9 (6,0) |
| Red 128+64 | 130 | 150 | 1·10⁻⁶ | 8,0 (3,3) | 8,1 (2,9) | 12,0 (5,4) | 10,3 (5,7) | 17,4 (10,4) |
| PCA (60) + boosting | 109 | 129 | 2·10⁻⁸ | 20,9 (10,7) | 20,6 (9,5) | 25,6 (13,4) | 25,0 (12,5) | 57,1 (15,2) |
| KNN (k = 150, bolsa = FakeNex) ¹ | 40 | 60 | 2·10⁻¹⁶ | 22,4 (17,5) | 18,8 (15,1) | 26,6 (21,5) | 33,9 (30,0) | 58,3 (54,0) |
| XGBoost real ² | 188 | 199 | 9·10⁻⁸ | 7,0 (3,2) | 7,1 (3,1) | 8,5 (4,1) | 7,6 (3,4) | 14,3 (6,6) |

Pesos y diferencias de las versiones float32; tiempos de la medición anterior (float64), que en float32 son parecidos.
Con JSON en double, el random forest (399 KB), el PCA + boosting (515), la red 128+64 (461) y el XGBoost real (347) no
cabían en Athena. ¹ El KNN no tiene arrays de pesos (su bolsa es la propia base) y se queda en float64; desde 0.3.0
está en `fnxsql` (núcleo `vecinos`). ² XGBoost ya es
float32: el SQL hace las mismas cuentas en real que el modelo.
(PCA + boosting se probó en el experimento pero no está en `fnxsql`: el PCA no es un paso de Pipeline admitido.)

## Lecciones (medidas durante el desarrollo)
- **Trino no materializa los CTE** y funde las proyecciones que se usan una sola vez: 77 árboles en columnas separadas
  acabaron en una única expresión de 230 KB que no compilaba. Usar cada columna dos veces (suma + array) lo evita; los
  arrays lo resuelven de raíz.
- **El texto también cuesta.** Una red de 20.672 pesos escrita término a término tardó 82 s y, con un lote de 10
  pacientes, reinició el coordinador de Trino. Con los pesos como literales JSON: 8,7 s y 1,3 s de planificación.
- **El float32 importa.** scikit-learn y XGBoost comparan en float32: sin reescribir umbrales, una creatinina de 1,16 iba
  por la rama contraria.
- **El límite de Athena obliga a compactar.** Con literales JSON en double, el random forest (399 KB), el PCA + boosting
  (515 KB), la red 128+64 (461 KB) y el XGBoost real (347 KB) no cabían en 262.144 bytes. En binario float32: 113, 109,
  130 y 188 KB, con diferencias de 4·10⁻⁹ a 2·10⁻⁶ en la probabilidad.
- **Redondear un umbral no es neutro.** Los umbrales de HistGradientBoosting coinciden con valores observados; al
  redondearlos al float32 más cercano, 9 recorridos de 620 ingresos cambiaban de rama (5·10⁻⁴ en la probabilidad).
  Redondeando hacia fuera según la comparación: 0.

### KNN de una métrica aprendida (PERISIM)

Una métrica de PERISIM (pesos NCA diagonales sobre funciones de forma por variable) se exporta en PERISIM
(`src/deploy/fnxsql_export.py`, formato `perisim_fnxsql_v1`: solo agregados) y `fnxsql.perisim.documento` la convierte en
un documento con:

- pasos de preproceso nuevos, uno por dimensión de la métrica: `tabla` (función escalonada sobre los cortes de los bins;
  opcionalmente con valor propio en cada corte), `interpolar` (lineal a trozos entre nodos estrictamente crecientes,
  constante fuera: la colocación por longitud de arco) y `categoria` (código numérico → valor; otro; nulo). En SQL, un
  `CASE` cada uno;
- núcleo `vecinos` con `ponderacion: "suavizada"`: p = (n₁ + 2α·π) / (n + 2α), con π la proporción de y = 1 en la bolsa
  (o `prior` del documento) y π si no hay vecinos;
- bolsa con `un_vecino_por_paciente`: cada paciente cuenta una vez, por su ingreso más cercano (SQL: `row_number()`
  por consulta y paciente antes del top-k).

Los pasos largos (más de 8 cortes o nodos) van como arrays en un CTE `formas`, escritos una sola vez y consultados con
`cardinality(filter(...))` (número de cortes ≤ v); los cortos, como `CASE`. Un artefacto PERISIM se evalúa como
cualquier modelo (`fnxsql evaluar perisim.json`): `ir.cargar` lo reconoce por su formato; la comparación con el modelo
original (el caso-base de PERISIM en Python) se hace en PERISIM (`scripts/fnxsql_fakenex.py`).

Tests: el SQL de los pasos = el intérprete (en DuckDB, también en su forma de arrays) y el KNN suavizado con un vecino por paciente = una referencia
por fuerza bruta; en PERISIM (`tests/test_deploy_fnxsql.py`), los pasos reproducen el embedding de PERISIM y el KNN de
fnxsql da las predicciones del caso-base de PERISIM (salvo cuasi-empates: PERISIM busca en float32).

## Estructura

```
fnxsql/
  contrato.py   gramática de nombres, SQL de variables (servidor) y cálculo local con pandas (validación)
  ir.py         carga de modelos, documento if-then exacto e intérprete
  perisim.py    documento de una métrica PERISIM exportada (pasos de forma + KNN suavizado)
  sql.py        compilador if-then -> SQL (terminos / case / arrays) con salvaguardas
  binario.py    precisión (float64 / float32 / int8): cuantización del documento, arrays base64, árboles en anchura
  cliente.py    API REST de Trino con controles de salud y límites
  validar.py    evaluación con comprobaciones automáticas e informe
  normalizar.py renombrado de variables a códigos de DataNex
  practicar.py  modelos de práctica con datos de FakeNex
  __main__.py   orquestador (python -m fnxsql)
ejemplos/       contratos de ejemplo
tests/          tests sin red (gramática, exactitud del if-then por tipo de modelo, compilación SQL)
```

## Licencia

MIT (ver `LICENSE`). FakeNex y sus datos son sintéticos; no subas a ningún repositorio pesos de modelos entrenados con
datos reales sin la autorización de su responsable (el `.gitignore` excluye `modelos/`, `*.joblib` y `*.pkl`).
