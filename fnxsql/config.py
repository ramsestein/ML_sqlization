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
MAX_SQL_BYTES = 950 * 1024   # el proxy de FakeNex rechaza más de 1 MiB
MAX_SEGUNDOS_CONSULTA = 180  # una consulta que tarda más detiene la batería
# Una representación "término a término" de ~20.000 operaciones tumbó el coordinador de Trino al planificar: por encima
# de este número de términos se exige la representación con arrays (o --forzar).
MAX_TERMINOS_SQL = 12_000

# ---------------------------------------------------------------- validación
SEMILLA = 42
TOLERANCIA_DOUBLE = 1e-9    # modelos que trabajan en double (scikit-learn)
TOLERANCIA_FLOAT32 = 1e-6   # XGBoost compara y suma en float32
