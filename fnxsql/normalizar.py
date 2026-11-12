"""Renombrado de las variables de un modelo a los códigos de DataNex / FakeNex (sin tocar el modelo original).

Ejemplo (XGBoost con los nombres en un JSON de metadatos aparte y renombres en el contrato):
  fnxsql normalizar modelo.json --nombres metadatos.json --contrato contrato.json --salida modelo_fakenex.json
Renombres: {"P_ART_S": "TAS"} cambia la base de <base>_<estadístico> (P_ART_S_mean -> TAS_mean); también se admite el
nombre completo ({"edad": "age"}). El modelo resultante predice exactamente igual (se comprueba).
"""
import copy
import json
from pathlib import Path

import numpy as np

from . import contrato as K, ir


def renombrar(nombre, renombres):
    if nombre in renombres:
        return renombres[nombre]
    if "_" in nombre:
        base, est = nombre.rsplit("_", 1)
        if base in renombres:
            return f"{renombres[base]}_{est}"
    return nombre


def normalizar(ruta, salida, ruta_nombres=None, ruta_contrato=None):
    obj, tipo = ir.cargar(ruta)
    contrato = K.Contrato.leer(ruta_contrato)
    if ruta_nombres:
        meta = json.loads(Path(ruta_nombres).read_text(encoding="utf-8"))
        originales = meta["feature_names"] if isinstance(meta, dict) else list(meta)
    else:
        originales = ir.nombres(obj, tipo)
    nuevos = [renombrar(n, contrato.renombres) for n in originales]
    if len(set(nuevos)) != len(nuevos):
        raise ValueError("el renombrado produce nombres repetidos")
    _, errores = K.variables_del_modelo(nuevos, contrato)
    salida = Path(salida)
    salida.parent.mkdir(parents=True, exist_ok=True)
    if tipo == "xgboost":
        nuevo = copy.deepcopy(obj)
        nf = int(nuevo["learner"]["learner_model_param"]["num_feature"])
        if nf != len(nuevos):
            raise ValueError(f"el modelo tiene {nf} variables y la lista de nombres {len(nuevos)}")
        nuevo["learner"]["feature_names"] = nuevos
        nuevo["learner"]["feature_types"] = ["float"] * len(nuevos)
        salida.write_text(json.dumps(nuevo), encoding="utf-8")
    else:
        import joblib
        nuevo = copy.deepcopy(obj)
        # un Pipeline toma feature_names_in_ de su primer paso; un estimador suelto, de sí mismo
        primero = nuevo.steps[0][1] if hasattr(nuevo, "steps") else nuevo
        primero.feature_names_in_ = np.array(nuevos, dtype=object)
        joblib.dump(nuevo, salida)
    cambios = [(a, b) for a, b in zip(originales, nuevos) if a != b]
    print(f"{len(nuevos)} variables, {len(cambios)} renombradas -> {salida}")
    for base in sorted({(a.rsplit('_', 1)[0], b.rsplit('_', 1)[0]) for a, b in cambios}):
        print(f"  {base[0]} -> {base[1]}")
    if errores:
        print(f"AVISO: {len(errores)} nombres siguen fuera de la gramática (decláralos en 'constantes' o renómbralos):")
        for e in errores[:20]:
            print(f"  {e}")
    # comprobación: mismas predicciones con los mismos valores
    o2, t2 = ir.cargar(salida)
    rng = np.random.default_rng(0)
    import pandas as pd
    X = pd.DataFrame(rng.normal(0, 50, (200, len(nuevos))), columns=nuevos)
    try:
        if tipo == "xgboost" and not obj["learner"].get("feature_names"):
            Xo = X.copy()
            Xo.columns = [f"f{i}" for i in range(len(nuevos))]
            o = copy.deepcopy(obj)
            o["learner"]["feature_names"] = list(Xo.columns)
            o["learner"]["feature_types"] = ["float"] * len(nuevos)
            p0 = ir.predecir_nativo(o, tipo, Xo)
        else:
            Xo = X.copy()
            Xo.columns = originales
            p0 = ir.predecir_nativo(obj, tipo, Xo)
        d = float(np.abs(p0 - ir.predecir_nativo(o2, t2, X)).max())
        print(f"modelo renombrado vs original (200 filas aleatorias): diferencia máxima {d:.1e}")
    except ImportError as e:
        print(f"(no se comprueban las predicciones: {e})")
    return salida
