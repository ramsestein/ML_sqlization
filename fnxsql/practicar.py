"""Generador de modelos de práctica: dataset de FakeNex con nombres de DataNex y varios modelos para evaluar con fnxsql.

No hace falta para evaluar un modelo propio; sirve para tener con qué practicar. Datos: todos los ingresos HOSP / HOSP_IQ
de la base original de FakeNex (sintéticos), con las variables de VARIABLES calculadas en el servidor con el mismo
contrato que la evaluación, y como etiqueta la muerte durante el ingreso (alta por exitus). Partición por paciente
(80 / 20): los pacientes de test se guardan para evaluar con pacientes que el modelo no vio. El rendimiento no significa
nada clínico (datos sintéticos): es un banco de pruebas de la conversión a SQL.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C, contrato as K

VARIABLES = ([f"{c}_{e}" for c in ("FC", "TAS", "FR", "TEMP", "PULSIOX") for e in ("last", "mean", "min", "max")]
             + [f"{c}_{e}" for c in ("LAB2467", "LAB110", "LABRPAK", "LAB1300", "LABHBST", "LAB2575") for e in ("last", "max")]
             + [f"ESTADO_CONCIENC_ESTADO_CONCIENC_{k}_last" for k in (1, 2, 3)] + ["age", "sex_2"])


def modelos():
    """Nombre -> (constructor, formato de guardado)."""
    from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.tree import DecisionTreeClassifier
    S = C.SEMILLA
    prep = lambda: [("imputar", SimpleImputer(strategy="median", add_indicator=True)), ("estandarizar", StandardScaler())]
    red = lambda capas: Pipeline(prep() + [("red", MLPClassifier(hidden_layer_sizes=capas, alpha=1e-3, max_iter=400,
                                                                 early_stopping=True, random_state=S))])
    m = {"logistica": (lambda: Pipeline(prep() + [("logistica", LogisticRegression(max_iter=3000, C=0.5))]), "joblib"),
         "arbol": (lambda: DecisionTreeClassifier(max_depth=5, min_samples_leaf=20, random_state=S), "joblib"),
         "random_forest": (lambda: RandomForestClassifier(n_estimators=100, max_leaf_nodes=32, min_samples_leaf=10,
                                                          random_state=S, n_jobs=-1), "joblib"),
         "boosting": (lambda: HistGradientBoostingClassifier(learning_rate=0.05, max_iter=200, max_leaf_nodes=15,
                                                             random_state=S), "joblib"),
         "red_pequena": (lambda: red((8,)), "joblib"), "red_mediana": (lambda: red((32,)), "joblib"),
         "red_grande": (lambda: red((128, 64)), "joblib"),
         # KNN: lo que viaja es la geometría (preproceso, k, pesos); la bolsa al evaluar es la propia base de FakeNex
         "knn": (lambda: Pipeline(prep() + [("knn", KNeighborsClassifier(n_neighbors=150, weights="distance"))]), "joblib")}
    try:
        import xgboost as xgb
        m["xgboost"] = (lambda: xgb.XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, random_state=S,
                                                  eval_metric="logloss"), "json")
    except ImportError:
        pass
    return m


def practicar(salida="salidas/practica", solo=None):
    """Descarga el dataset, entrena los modelos de práctica y los guarda en <salida>/modelos."""
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupShuffleSplit
    import joblib
    salida = Path(salida)
    (salida / "modelos").mkdir(parents=True, exist_ok=True)
    contrato = K.Contrato()
    variables, errores = K.variables_del_modelo(VARIABLES, contrato)
    assert not errores, errores
    cache = salida / "dataset.csv"
    if cache.exists():
        df = pd.read_csv(cache)
        print(f"dataset en caché: {cache} ({len(df)} ingresos)")
    else:
        print("descargando el dataset de FakeNex (variables de toda la base original, una consulta)...", flush=True)
        df, reloj, st = K.descargar_variables(variables, contrato, None, con_etiqueta=True)
        df.to_csv(cache, index=False)
        print(f"  {len(df)} ingresos de {df.patient_ref.nunique()} pacientes, {df.muere.mean():.1%} fallecidos, {reloj:.0f} s")
    tr, te = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=C.SEMILLA).split(df, df.muere, df.patient_ref))
    Xtr, Xte = df.iloc[tr][VARIABLES], df.iloc[te][VARIABLES]
    ytr, yte = df.iloc[tr].muere.values, df.iloc[te].muere.values
    resumen = {"variables": VARIABLES, "ingresos": len(df), "pacientes_test": sorted(map(int, df.iloc[te].patient_ref.unique())),
               "modelos": {}}
    for nombre, (hacer, fmt) in modelos().items():
        if solo and nombre not in solo:
            continue
        m = hacer().fit(Xtr, ytr)
        p = m.predict_proba(Xte)[:, 1]
        ruta = salida / "modelos" / f"{nombre}.{fmt}"
        if fmt == "json":
            m.save_model(ruta)
        else:
            joblib.dump(m, ruta)
        auc = float(roc_auc_score(yte, p)) if len(set(yte)) > 1 else float("nan")
        resumen["modelos"][nombre] = {"fichero": str(ruta), "auc_test": round(auc, 3)}
        print(f"  {nombre:14s} AUC test {auc:.3f} -> {ruta}", flush=True)
    (salida / "practica.json").write_text(json.dumps(resumen, indent=1), encoding="utf-8")
    return resumen
