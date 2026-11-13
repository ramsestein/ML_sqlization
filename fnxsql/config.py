"""Constantes compartidas: servidor, esquema, ventanas por defecto y límites de seguridad."""
import os
import secrets

# ---------------------------------------------------------------- servidor FakeNex (Trino con el dialecto de Athena)
URL = os.environ.get("FAKENEX_URL", "https://fakenex.sliplane.app")
# Usuario propio por ejecución: quien comparte nombre ve y cancela tus consultas (documentación de FakeNex).
USUARIO = os.environ.get("FAKENEX_USER") or f"fnxsql-{secrets.token_hex(4)}"
CATALOGO = "awsdatacatalog"
ESQUEMA = "datascope_gestor_prod"
BASE = "< 1000000000"  # pacientes originales; los >= 1e9 los añade el crecimiento automático de FakeNex

# ---------------------------------------------------------------- unidad de análisis y ventana por defecto
# Unidad: ingreso hospitalario (episodes HOSP / HOSP_IQ). Momento de evaluación: inicio + MOMENTO_HORAS (o el alta si es
# antes). Las variables se calculan en [momento - VENTANA_HORAS, momento]. Ambos se pueden cambiar en el contrato.
TIPOS_INGRESO = ("HOSP", "HOSP_IQ")
MOMENTO_HORAS = 24
VENTANA_HORAS = 48

# ---------------------------------------------------------------- límites de seguridad (lecciones del experimento)
# DataNex corre en AWS Athena, que admite consultas de hasta 262.144 bytes: es el límite por defecto. Con --forzar se
# permite llegar al del proxy de FakeNex (rechaza más de 1 MiB) para probar algo que no se podría llevar a DataNex.
MAX_SQL_BYTES = 262_144
MAX_SQL_BYTES_FAKENEX = 950 * 1024
MAX_SEGUNDOS_CONSULTA = 180  # una consulta que tarda más detiene la batería
# Una representación "término a término" de ~20.000 operaciones tumbó el coordinador de Trino al planificar: por encima
# de este número de términos se exige la representación con arrays (o --forzar).
MAX_TERMINOS_SQL = 12_000

# ---------------------------------------------------------------- precisión de los parámetros (binario.py)
# float32: todo parámetro redondeado a float32 y los arrays en binario de 4 bytes (cuentas en double); float64: exacto;
# int8: pesos, coeficientes y hojas como enteros de 8 bits con una escala float32 por neurona / modelo / árbol.
PRECISION = "float32"

# ---------------------------------------------------------------- validación
SEMILLA = 42
TOLERANCIA_DOUBLE = 1e-9    # modelos que trabajan en double (scikit-learn)
TOLERANCIA_FLOAT32 = 1e-6   # XGBoost compara y suma en float32 (con cualquier precisión: ya es float32)
TOLERANCIA_PARAMETROS_FLOAT32 = 1e-5  # modelos en double con los parámetros redondeados a float32 (--precision float32)
# KNN frente a scikit-learn: con algorithm="brute" calcula la euclídea como |x|² + |y|² - 2 x·y, con errores de hasta
# ~3e-7 en las distancias (un ingreso que está en la bolsa le sale a ~1e-7 de sí mismo, no a 0); con 1 / distancia eso
# llega a ~4e-6 en la probabilidad (medido con el KNN de práctica). El if-then y el SQL restan coordenada a coordenada.
# Los vecinos elegidos se comprueban aparte; SQL = if-then sobre la misma bolsa sigue con TOLERANCIA_DOUBLE.
TOLERANCIA_KNN_SKLEARN = 1e-5
