# ML_sqlization (`fnxsql`) — de tu modelo a una consulta SQL que se ejecuta en FakeNex

**Tienes un modelo de machine learning entrenado con datos de DataNex y quieres comprobar que se puede ejecutar dentro
del data warehouse, como una consulta SQL, sin servidores de inferencia.** `fnxsql` lo convierte en reglas if-then
exactas, genera a partir de ellas **una única consulta SQL** (Trino / Athena) que calcula las variables desde las tablas
crudas y puntúa a los pacientes, la ejecuta en [FakeNex](https://fakenex.sliplane.app/guide) (réplica sintética de
DataNex, de solo lectura) y **verifica automáticamente** que el resultado coincide con el modelo original.

```
tu modelo (.json XGBoost / .joblib scikit-learn)
   │  1. nombres  → cada variable es un código de DataNex (FC_mean, LAB2467_max, ...): de ahí sale cómo calcularla
   ▼
documento if-then exacto (reglas, árboles if-then anidados, capas con ReLU = "si z > 0 entonces z si no 0")
   │  2. compilar → una consulta SQL: variables desde rc / labs / demographics + el modelo (arrays, sin bucles)
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

Cualquiera de los de scikit-learn puede ir en un `Pipeline` con `SimpleImputer` y / o `StandardScaler` delante: se
traducen como reglas de preproceso. **No admitidos**: KNN (su "modelo" son los datos de entrenamiento), multiclase,
regresión, variables categóricas de HistGradientBoosting o XGBoost, LightGBM.

Por qué así (medido en FakeNex, ver «Lecciones»):

- **arrays** (por defecto en árboles y redes): todos los nodos o pesos van en unos pocos literales JSON y se recorren con
  `reduce(sequence(1, profundidad), raíz, ...)` / `zip_with`. Sin bucles y con pocos tokens: ~1 s de planificación.
- **case** / **terminos**: cada nodo o peso es una expresión. Legible, pero el coste de planificar crece con el modelo:
  `fnxsql` se niega por encima de 12.000 términos (`--forzar` para saltárselo).
- **float32**: XGBoost compara `float32(x) < umbral` y suma las hojas en float32; el SQL lo reproduce con el tipo `real`.
  Los árboles de scikit-learn comparan en float32 sobre double: los umbrales se reescriben a su equivalente double.

## Validaciones automáticas

`evaluar` comprueba, y deja en el informe:

1. Las variables siguen la gramática; los códigos de laboratorio existen en `dic_lab_loinc` (avisa de los de `rc` sin
   datos en FakeNex: valdrán ausentes).
2. El documento if-then reproduce el modelo original en 300 filas sintéticas con ausentes.
3. La consulta se ejecuta en FakeNex para los pacientes elegidos.
4. Las variables calculadas en el servidor son idénticas a las recalculadas **con otra implementación** (pandas, en local)
   desde las tablas crudas de esos pacientes.
5. SQL = modelo original = if-then en esos ingresos reales (tolerancia 1e-9; 1e-6 en XGBoost por el float32).
6. Opcionalmente, tiempos por tamaño de lote (`--lotes`).

Salidas en `salidas/<modelo>/`: `informe.md` / `informe.json`, `if_then.json`, `consulta.sql`, `comparacion.csv` y
`tablas/` (las filas crudas descargadas).

## Órdenes

| Orden | Para qué | Opciones principales |
|---|---|---|
| `evaluar MODELO` | circuito completo con validaciones e informe | `--contrato`, `--pacientes 1,2` o `--muestra N`, `--sql`, `--lotes`, `--salida`, `--forzar` |
| `nombres MODELO` | comprobar los nombres (y su existencia en FakeNex) | `--contrato`, `--sin-servidor` |
| `normalizar MODELO` | copia del modelo con nombres de DataNex | `--contrato` (renombres), `--nombres metadatos.json`, `--salida` |
| `convertir MODELO` | solo el documento if-then | `--contrato`, `--salida` |
| `sql MODELO` | solo la consulta, sin ejecutarla | `--pacientes` / `--muestra`, `--sql`, `--salida` |
| `practicar` | entrenar modelos de ejemplo con datos de FakeNex | `--modelos`, `--salida` |
| `demo` | `practicar` + `evaluar` de todos | `--modelos`, `--muestra`, `--lotes`, `--sql` |

Globales: `--url` (o `FAKENEX_URL`) y `--usuario` (o `FAKENEX_USER`; por defecto uno aleatorio `fnxsql-xxxxxxxx`).

## Buen uso del servidor

FakeNex es compartido: 384 MB por consulta, 10 min, 4 consultas a la vez. `fnxsql` comprueba la salud del servidor
(`/v1/info`) antes de cada consulta y espera si está arrancando, rechaza SQL de más de 950 KB (el proxy admite 1 MiB),
detiene la batería si una consulta tarda más de 3 min, usa un usuario propio y filtra siempre por `patient_ref` (lo que,
además, no dispara el crecimiento automático de la base). Las consultas van de una en una.

## Resultados de referencia

Todo contra FakeNex (noviembre de 2026). Tiempos de reloj en el cliente, en segundos, con la red incluida; FakeNex es
compartido, así que hay variabilidad entre ejecuciones.

**`python -m fnxsql demo --lotes 1,10,50`** (modelos de práctica entrenados con la base de FakeNex; validación con 3
pacientes de test):

| Modelo | Resultado | SQL | KB | SQL vs modelo | 1 pac. | 10 | 50 |
|---|---|---|---|---|---|---|---|
| logistica | PASA | terminos | 18 | 4,9·10⁻¹⁷ | 4,6 | 5,7 | 5,2 |
| arbol | PASA | arrays | 10 | 0 | 3,9 | 4,9 | 5,6 |
| random_forest | PASA | arrays | 205 | 6,9·10⁻¹⁸ | 5,8 | 5,8 | 7,4 |
| boosting | PASA | arrays | 178 | 0 | 6,4 | 6,7 | 8,9 |
| red_pequena | PASA | arrays | 19 | 1,1·10⁻¹⁶ | 5,3 | 5,1 | 6,8 |
| red_mediana | PASA | arrays | 39 | 1,1·10⁻¹⁶ | 7,3 | 6,3 | 6,4 |
| red_grande | PASA | arrays | 285 | 4,2·10⁻¹⁷ | 5,3 | 6,8 | 7,5 |
| xgboost | PASA | arrays | 167 | 1,8·10⁻⁸ (float32) | 5,1 | 6,2 | 9,0 |

**Un modelo real** (XGBoost de deterioro clínico entrenado con DataNex, 800 árboles, 195 variables; no incluido):
`normalizar` con `ejemplos/contrato_deterioro.json` (30 variables renombradas, mismas predicciones) y `evaluar`: PASA,
537 KB, 195 variables idénticas servidor / local, SQL vs XGBoost original 4,0·10⁻⁸; 1 / 10 / 50 pacientes en 8,1 / 6,9 /
10,3 s.

**Consultas de referencia** (para extrapolar a DataNex: ejecutándolas allí con N pacientes, el tiempo de un modelo se
estima escalando su tiempo en FakeNex por la razón de las referencias, sobre todo la de «variables»). Reloj (servidor):

| Consulta | 1 pac. | 10 | 50 | 100 | 250 |
|---|---|---|---|---|---|
| `SELECT 1` | 4,5 (1,7) | 4,5 (2,2) | 4,5 (2,3) | 4,9 (2,2) | 4,9 (2,3) |
| ingresos (`episodes` por paciente) | 4,7 (2,3) | 4,1 (1,7) | 5,5 (2,4) | 5,5 (2,0) | 6,7 (1,5) |
| variables de un modelo de mortalidad (47) | 9,3 (6,6) | 5,8 (2,6) | 7,3 (3,3) | 9,3 (3,7) | 14,8 (5,0) |
| variables del XGBoost real (195) | 7,2 (2,6) | 7,4 (3,5) | 7,2 (3,1) | 8,4 (3,2) | 16,2 (3,1) |

**Experimento de desarrollo** (modelos de mortalidad entrenados con 15.000 pacientes sintéticos generados con FakeNex;
versiones SQL finales; AUC sobre 620 ingresos de 250 pacientes de FakeNex, 44 muertes). Reloj (servidor):

| Versión | KB | AUC modelo | AUC SQL | Dif. | 1 pac. | 10 | 50 | 100 | 250 |
|---|---|---|---|---|---|---|---|---|---|
| Regresión logística — términos | 21 | 0,7730 | 0,7730 | 0 | 7,1 (3,4) | 7,3 (3,3) | 7,8 (4,7) | 7,8 (3,4) | 8,4 (4,1) |
| Gradient boosting — arrays | 161 | 0,8119 | 0,8119 | 0 | 7,2 (2,7) | 9,1 (3,2) | 11,8 (4,2) | 14,7 (3,5) | 33,3 (10,9) |
| Random forest — arrays | 399 | 0,7756 | 0,7756 | 0 | 8,9 (3,5) | 9,6 (4,0) | 9,6 (3,9) | 17,2 (5,0) | 21,0 (5,0) |
| Red 8 — arrays | 37 | 0,7573 | 0,7573 | 0 | 5,4 (2,8) | 5,9 (2,7) | 6,2 (3,1) | 8,3 (3,8) | 12,0 (5,2) |
| Red 32 — arrays | 87 | 0,7967 | 0,7967 | 0 | 7,0 (2,7) | 6,7 (3,3) | 7,1 (3,0) | 8,3 (4,3) | 9,9 (6,0) |
| Red 128+64 — arrays | 461 | 0,8181 | 0,8181 | 0 | 8,0 (3,3) | 8,1 (2,9) | 12,0 (5,4) | 10,3 (5,7) | 17,4 (10,4) |
| PCA (60) + boosting — CASE | 515 | 0,7814 | 0,7814 | 0 | 20,9 (10,7) | 20,6 (9,5) | 25,6 (13,4) | 25,0 (12,5) | 57,1 (15,2) |
| KNN (k = 150, bolsa = FakeNex) — vector | 41 | 0,7374 | 0,7374 | 0 | 22,4 (17,5) | 18,8 (15,1) | 26,6 (21,5) | 33,9 (30,0) | 58,3 (54,0) |
| XGBoost real — arrays | 347 | — | — | — | 7,0 (3,2) | 7,1 (3,1) | 8,5 (4,1) | 7,6 (3,4) | 14,3 (6,6) |

(PCA + boosting y KNN se probaron en el experimento pero no están en `fnxsql`: el PCA no es un paso de Pipeline admitido
y el "modelo" de un KNN son sus datos de entrenamiento.)

## Lecciones (medidas durante el desarrollo)

- **Sin bucles.** SQL no tiene bucles; Trino solo admite `WITH RECURSIVE` limitado. Los árboles se recorren con `reduce`
  sobre un número fijo de pasos (la profundidad).
- **Trino no materializa los CTE** y funde las proyecciones que se usan una sola vez: 77 árboles en columnas separadas
  acabaron en una única expresión de 230 KB que no compilaba. Usar cada columna dos veces (suma + array) lo evita; los
  arrays lo resuelven de raíz.
- **El texto también cuesta.** Una red de 20.672 pesos escrita término a término tardó 82 s y, con un lote de 10
  pacientes, reinició el coordinador de Trino. Con los pesos como literales JSON: 8,7 s y 1,3 s de planificación.
- **El float32 importa.** scikit-learn y XGBoost comparan en float32: sin reescribir umbrales, una creatinina de 1,16 iba
  por la rama contraria.

## Estructura

```
fnxsql/
  contrato.py   gramática de nombres, SQL de variables (servidor) y cálculo local con pandas (validación)
  ir.py         carga de modelos, documento if-then exacto e intérprete
  sql.py        compilador if-then -> SQL (terminos / case / arrays) con salvaguardas
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
