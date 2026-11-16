# Entrenar con SQL frente a descargar y entrenar XGBoost

Método SQL: todo el entrenamiento dentro de la consulta (entrenar_una.py).

2026-11-16 04:09:26 · 37 variables · 2583 ingresos de entrenamiento y 679 de prueba (51 muertes) · 100 árboles, profundidad 3, eta 0.1, lambda 1.0, min_child_weight 1.0, 256 intervalos

## Tiempo y datos que salen del servidor

| | A. consultas SQL | B. descarga + XGBoost |
|---|---|---|
| Tiempo total (s) | 377.5 | 8.5 (descarga 8.4 + entrenamiento 0.039) |
| Consultas | 5 | 1 |
| Tiempo por consulta (mediana / máx., s) | 75.39 / 78.63 | 8.4 |
| Servidor: reloj / planificación / CPU (s) | 354.4 / 28.4 / 107.6 | 6.1 / - / - |
| SQL enviado (total / máx. por consulta) | 0.6 MB / 135 KB | - |
| Filas de pacientes que salen | **0** (solo 5 filas: los árboles y las pérdidas) | **3262** (137,004 valores) |

Memoria máxima por consulta en A: 41.0 MB. XGBoost 3.4.1 con 4 hilos.

## Calidad en los pacientes de prueba

| Modelo | AUC | Pérdida logística | Brier |
|---|---|---|---|
| A. SQL | 0.8750 | 0.1808 | 0.0484 |
| B. XGBoost local | 0.8657 | 0.1819 | 0.0482 |
| B2. XGBoost local con los cortes de A | 0.8750 | 0.1808 | 0.0484 |
| solo tasa base | 0.5000 | 0.2670 | 0.0695 |

Diferencia de probabilidad A frente a B (prueba): máxima 0.123, media 0.0086, p95 0.0373; correlación de Spearman 0.9817.
A frente a B2 (mismos cortes): máxima 6.3e-08, media 3.3e-09, p95 2.1e-08.
Entrenamiento: AUC A 0.9859, B 0.9862.

## ¿Calcula el SQL exactamente lo previsto?

- Réplica en numpy con los mismos cortes: 0 nodos distintos de 1396 (100 árboles); hojas: diferencia máxima 0.0e+00; probabilidades: 0.0e+00.
- Modelo de A cargado en la librería XGBoost (JSON exportado) frente a numpy: 7.0e-08; documento if-then: 3.9e-08.

## Pérdida logística por número de árboles

| Árboles | A prueba | B prueba | A entrenamiento | B entrenamiento |
|---|---|---|---|---|
| 0 | 0.2670 | - | 0.2508 | - |
| 1 | 0.2571 | 0.2571 | 0.2362 | 0.2362 |
| 5 | 0.2363 | 0.2364 | 0.2078 | 0.2078 |
| 10 | 0.2211 | 0.2210 | 0.1883 | 0.1882 |
| 25 | 0.2022 | 0.2029 | 0.1565 | 0.1570 |
| 50 | 0.1915 | 0.1929 | 0.1281 | 0.1278 |
| 100 | 0.1808 | 0.1819 | 0.0937 | 0.0929 |
