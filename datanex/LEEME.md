# Prueba en DataNex (AWS Athena)

Consultas generadas por `python -m fnxsql datanex preparar` para lanzarlas a mano en Athena. Cada una devuelve una tabla
que se descarga como **CSV** y se guarda en `datanex/resultados/` con **el mismo nombre que la consulta** (`.txt` → `.csv`).
`datanex/resultados/` está en `.gitignore`: ahí quedan datos reales y el modelo entrenado con ellos; no se sube.

Esquema: `datascope_gestor_prod`. Los pacientes se eligen dentro de cada consulta con un hash de `patient_ref`, entre los que
tienen algún ingreso HOSP / HOSP_IQ: todas las consultas de inferencia usan los mismos 10 pacientes
(semilla 1) y todas las de entrenamiento los mismos 1000
(semilla 0). Si la base cambia entre una consulta y otra, `continuar` lo detecta.

## 1. Inferencia de los modelos (10 pacientes al azar)

| Consulta | Qué hace |
|---|---|
| `consultas/00_variables_10_pacientes.txt` | las 37 variables de esos ingresos (para comprobar las probabilidades en local) |
| `consultas/01_inferencia_logistica.txt` | inferencia de **logistica** (terminos, 18 KB) |
| `consultas/02_inferencia_arbol.txt` | inferencia de **arbol** (arrays, 10 KB) |
| `consultas/03_inferencia_random_forest.txt` | inferencia de **random_forest** (arrays, 59 KB) |
| `consultas/04_inferencia_boosting.txt` | inferencia de **boosting** (arrays, 56 KB) |
| `consultas/05_inferencia_red_pequena.txt` | inferencia de **red_pequena** (arrays, 15 KB) |
| `consultas/06_inferencia_red_mediana.txt` | inferencia de **red_mediana** (arrays, 20 KB) |
| `consultas/07_inferencia_red_grande.txt` | inferencia de **red_grande** (arrays, 84 KB) |
| `consultas/08_inferencia_xgboost.txt` | inferencia de **xgboost** (arrays, 63 KB) |
| `consultas/09_inferencia_boosting_sql.txt` | inferencia de **boosting_sql** (arrays, 21 KB) |
| `consultas/10_inferencia_xgboost_local.txt` | inferencia de **xgboost_local** (arrays, 21 KB) |

Cada CSV de inferencia tiene una fila por ingreso: `patient_ref, episode_ref, ingreso, momento, probabilidad`.
Los modelos son los de práctica (entrenados con los datos sintéticos de FakeNex): la prueba es de funcionamiento, no de
rendimiento clínico.

## 2. Entrenamiento en SQL (5 consultas)

Gradient boosting (algoritmo de XGBoost con histogramas) entrenado **dentro de la consulta**: 100 árboles de
profundidad 3, eta 0.1, 256 intervalos, 20 árboles por consulta,
con 1000 pacientes al azar (20 % de prueba por paciente). Variables: las 37 de
`fnxsql practicar`; etiqueta: muerte en el ingreso (`adm_disch`, `mot_ref` 23). Cada CSV tiene **una sola fila** con los
árboles y las pérdidas en JSON; no contiene datos de pacientes.

1. Ejecutar `consultas/entrenamiento_01.txt` (91 KB) y guardar `resultados/entrenamiento_01.csv`.
2. `python -m fnxsql datanex continuar` → escribe `resultados/entrenamiento_02.txt` (la tanda siguiente, con los cortes y
   los árboles ya hechos dentro de la consulta). Ejecutarla y guardar `resultados/entrenamiento_02.csv`.
3. Repetir `continuar` hasta la tanda 5. Tras la última, `continuar` escribe el modelo
   (`resultados/boosting_datanex.json`, formato XGBoost) y `resultados/inferencia_boosting_datanex.txt`: ejecutarla y guardar
   su CSV (inferencia del modelo recién entrenado sobre los mismos 10 pacientes).

## 3. Comprobar

`python -m fnxsql datanex comprobar` → `resultados/comprobacion.csv` (y en pantalla):

- cada inferencia: mismos ingresos que la consulta de variables y probabilidad del SQL = modelo evaluado en local
  (tolerancia 1e-6 en los modelos float32 de por sí, como XGBoost; 1e-9 en los demás);
- entrenamiento: misma cohorte en todas las tandas, cada tanda empieza con la pérdida en que acabó la anterior, pérdida
  final de entrenamiento y de prueba.

Si algo falla en Athena (sintaxis, tiempo, memoria), el mensaje de error es lo que hay que traer.
