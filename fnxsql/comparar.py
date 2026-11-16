"""Estudio: entrenar con consultas SQL (sin descargar filas de pacientes) frente a descargar los datos y entrenar XGBoost.

Mismas variables, mismos ingresos, misma partición por paciente (hash de patient_ref) y mismos hiperparámetros:
  A. SQL        entrenar.entrenar: una consulta por árbol y nivel, el modelo en curso como arrays binarios.
  B. descarga   una consulta con las variables de todos los ingresos + xgboost.XGBClassifier(tree_method="hist") en local.
  C. réplica    entrenar.entrenar_local: el algoritmo del SQL en numpy sobre los datos de B y con los cortes de A
                (¿calcula el SQL exactamente lo previsto?).
Se miden tiempos (reloj, servidor, planificación), bytes y filas que salen del servidor, la calidad en los pacientes de
prueba (AUC, pérdida logística, Brier) y la diferencia de probabilidad entre los modelos. Salida: informe.md / .json.
"""
import json
import os
import statistics
import time
from pathlib import Path

import numpy as np

from . import contrato as K, entrenar as E, ir

PUNTOS_CURVA = (0, 1, 5, 10, 25, 50, 100, 200, 300, 500, 1000)


def _metricas(y, prob):
    from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
    prob = np.clip(prob, 1e-15, 1 - 1e-15)
    return {"auc": float(roc_auc_score(y, prob)), "perdida_logistica": float(log_loss(y, prob, labels=[0, 1])),
            "brier": float(brier_score_loss(y, prob))}


def _dif(a, b):
    d = np.abs(np.asarray(a, dtype=float) - np.asarray(b, dtype=float))
    return {"maxima": float(d.max()), "media": float(d.mean()), "p95": float(np.quantile(d, 0.95))}


def xgboost_mismos_cortes(X, y, prueba, cortes, tasa, p):
    """XGBoost local con los MISMOS cortes que el SQL: cada valor se sustituye por su intervalo (0, 1, 2...; ausente = NaN),
    así XGBoost evalúa exactamente los mismos cortes candidatos. Separa el efecto de cómo se eligen los cortes (quantile
    sketch de XGBoost frente a approx_percentile de Trino) del resto (gradientes en float32, desempates)."""
    import xgboost as xgb
    Bn = E.intervalos(X, cortes).astype(float)
    Bn[Bn < 0] = np.nan
    ent = ~prueba
    m = xgb.XGBClassifier(n_estimators=p.arboles, max_depth=p.profundidad, learning_rate=p.eta, reg_lambda=p.lambda_,
                          min_child_weight=p.min_hijo, max_bin=2 * p.bins + 2, tree_method="hist", base_score=tasa, gamma=0.0,
                          random_state=0).fit(Bn[ent], y[ent])
    return m.predict_proba(Bn)[:, 1]


def completar(carpeta):
    """Añade a un informe ya hecho el control «XGBoost con los mismos cortes» (sin servidor) y rehace informe.md."""
    carpeta = Path(carpeta)
    import pandas as pd
    inf = json.loads((carpeta / "informe.json").read_text(encoding="utf-8"))
    ent_json = json.loads((carpeta / "sql" / "entrenamiento.json").read_text(encoding="utf-8"))
    p = E.Parametros(**inf["parametros"])
    df = pd.read_csv(carpeta / "dataset.csv")
    pr = np.load(carpeta / "predicciones.npz")
    X, y, prueba = df[inf["variables"]].to_numpy(dtype=float), df.muere.to_numpy(dtype=float), pr["prueba"]
    p_b2 = xgboost_mismos_cortes(X, y, prueba, ent_json["cortes"], ent_json["tasa"], p)
    inf["D_calidad"]["prueba"]["xgboost_mismos_cortes"] = _metricas(y[prueba], p_b2[prueba])
    inf["D_calidad"]["diferencia_prob_sql_vs_xgboost_mismos_cortes"] = {"prueba": _dif(pr["p_sql"][prueba], p_b2[prueba]),
                                                                        "todos": _dif(pr["p_sql"], p_b2)}
    (carpeta / "informe.json").write_text(json.dumps(inf, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    (carpeta / "informe.md").write_text(informe_md(inf), encoding="utf-8")
    return inf


def comparar(nombres, contrato=None, p=None, salida="salidas/comparacion", repeticiones=5, forzar=False, metodo="una",
             arboles_por_consulta=None):
    """metodo: "una" (todo el entrenamiento en una consulta, entrenar_una.py) o "niveles" (una consulta por árbol y nivel)."""
    import xgboost as xgb
    from . import entrenar_una as U
    contrato, p = contrato or K.Contrato(), p or E.Parametros()
    carpeta = Path(salida)
    carpeta.mkdir(parents=True, exist_ok=True)
    variables, errores = K.variables_del_modelo(nombres, contrato)
    if errores:
        raise ValueError("; ".join(errores))
    inf = {"parametros": vars(p), "variables": nombres, "fecha": time.strftime("%Y-%m-%d %H:%M:%S"), "metodo": metodo}

    # ---------------------------------------------------------------- A. SQL
    print(f"== A. entrenamiento con consultas SQL ({p.arboles} árboles, profundidad {p.profundidad}, {p.bins} intervalos)",
          flush=True)
    res = (U.entrenar(nombres, contrato, p, arboles_por_consulta, forzar=forzar) if metodo == "una"
           else E.entrenar(nombres, contrato, p, forzar=forzar))
    E.guardar(res, carpeta / "sql")
    cs = res["consultas"]
    pasos = [c for c in cs if c["que"] != "inicio"] or cs
    inf["A_sql"] = {"segundos": round(res["segundos"], 1), "consultas": len(cs),
                    "reloj_s": round(sum(c["reloj_s"] for c in cs), 1), "servidor_s": round(sum(c["servidor_s"] for c in cs), 1),
                    "planificacion_s": round(sum(c["planificacion_s"] for c in cs), 1),
                    "cpu_servidor_s": round(sum(c["cpu_s"] for c in cs), 1),
                    "por_consulta_mediana_s": statistics.median(c["reloj_s"] for c in pasos),
                    "por_consulta_max_s": max(c["reloj_s"] for c in pasos),
                    "memoria_max_mb": max(c["memoria_mb"] for c in cs),
                    "bytes_sql_enviados": sum(c["bytes_sql"] for c in cs), "bytes_sql_max": max(c["bytes_sql"] for c in cs),
                    "filas_devueltas": sum(c["filas_devueltas"] for c in cs), "filas_de_pacientes_devueltas": 0,
                    "ingresos": res["ingresos"]}

    # ---------------------------------------------------------------- B. descarga + XGBoost
    print("== B. descarga de las variables de todos los ingresos + XGBoost en local", flush=True)
    t = time.time()
    df, reloj, st = K.descargar_variables(variables, contrato, None, con_etiqueta=True)
    t_desc = time.time() - t
    df.to_csv(carpeta / "dataset.csv", index=False)
    prueba = E.es_prueba(df.patient_ref.to_numpy(), p)
    X, y = df[nombres].to_numpy(dtype=float), df.muere.to_numpy(dtype=float)
    ent = ~prueba
    assert ent.sum() == res["ingresos"]["entrenamiento"], "la partición local no coincide con la del servidor"
    kw = dict(n_estimators=p.arboles, max_depth=p.profundidad, learning_rate=p.eta, reg_lambda=p.lambda_,
              min_child_weight=p.min_hijo, max_bin=p.bins, tree_method="hist", base_score=res["tasa"], gamma=0.0,
              random_state=0, eval_metric="logloss")
    tiempos = []
    for _ in range(repeticiones):
        t = time.time()
        modelo = xgb.XGBClassifier(**kw).fit(X[ent], y[ent])
        tiempos.append(time.time() - t)
    curva = xgb.XGBClassifier(**kw).fit(X[ent], y[ent], eval_set=[(X[ent], y[ent]), (X[prueba], y[prueba])], verbose=False)
    ev = curva.evals_result()
    modelo.get_booster().feature_names = list(nombres)
    modelo.save_model(carpeta / "xgboost_local.json")
    inf["B_descarga"] = {"descarga_s": round(t_desc, 1), "descarga_servidor_s": round(st.get("elapsedTimeMillis", 0) / 1000, 1),
                         "entrenamiento_xgboost_s_mediana": round(statistics.median(tiempos), 3),
                         "entrenamiento_xgboost_s": [round(x, 3) for x in tiempos], "hilos": os.cpu_count(),
                         "segundos": round(t_desc + statistics.median(tiempos), 1), "consultas": 1,
                         "filas_de_pacientes_devueltas": len(df), "valores_devueltos": int(df.shape[0] * df.shape[1]),
                         "xgboost": xgb.__version__}

    # ---------------------------------------------------------------- C. réplica local del algoritmo del SQL
    print("== C. el algoritmo del SQL en numpy (verificación)", flush=True)
    t = time.time()
    if metodo == "una":  # mismas operaciones y en el mismo orden que la consulta: debe coincidir bit a bit
        cd = [c_ or [0.0] for c_ in res["cortes"]]
        arb_loc = [U._arbol_doc(f_, b_, i_, v_, [0] * len(f_), [0] * len(f_), [0] * len(f_), [0] * len(f_), cd, nombres, p)
                   for f_, b_, i_, v_ in U.local(X, y, prueba, df.episode_ref.to_numpy(), res["cortes"], res["margen"], p)]
        hist_loc = res["historial"]
    else:
        arb_loc, hist_loc = E.entrenar_local(X, y, prueba, res["cortes"], (res["tasa"], res["margen"]), nombres, p)
    t_rep = time.time() - t
    arb_sql = res["documento"]["nucleo"]["arboles"]
    distintos, dhoja, nodos = E.comparar_arboles(arb_sql, arb_loc)
    dperd = max(abs(a["perdida_entrenamiento"] - b["perdida_entrenamiento"]) for a, b in zip(res["historial"], hist_loc))
    inf["C_replica"] = {"segundos": round(t_rep, 1), "arboles_sql": len(arb_sql), "arboles_local": len(arb_loc),
                        "nodos_comparados": nodos, "nodos_distintos": distintos, "dif_max_hojas": dhoja,
                        "dif_max_perdida_entrenamiento": dperd}

    # ---------------------------------------------------------------- D. predicciones y calidad
    m_sql = E.margen_modelo(arb_sql, res["margen"], X, nombres).astype(float)
    p_sql = 1 / (1 + np.exp(-m_sql))
    bst = xgb.Booster()
    bst.load_model(bytearray(json.dumps(res["xgboost"]).encode("utf-8")))
    p_sql_lib = bst.predict(xgb.DMatrix(X, feature_names=list(nombres), missing=np.nan))
    p_rep = 1 / (1 + np.exp(-E.margen_modelo(arb_loc, res["margen"], X, nombres).astype(float)))
    p_xgb = modelo.predict_proba(X)[:, 1]
    p_b2 = xgboost_mismos_cortes(X, y, prueba, res["cortes"], res["tasa"], p)
    muestra = np.random.default_rng(0).choice(len(X), size=min(200, len(X)), replace=False)
    p_ir = np.array([ir.evaluar(res["documento"], dict(zip(nombres, X[i]))) for i in muestra])
    from scipy.stats import spearmanr
    inf["D_calidad"] = {
        "prueba": {"ingresos": int(prueba.sum()), "positivos": int(y[prueba].sum()),
                   "sql": _metricas(y[prueba], p_sql[prueba]), "xgboost_local": _metricas(y[prueba], p_xgb[prueba]),
                   "xgboost_mismos_cortes": _metricas(y[prueba], p_b2[prueba]),
                   "solo_tasa_base": _metricas(y[prueba], np.full(prueba.sum(), res["tasa"]))},
        "entrenamiento": {"sql": _metricas(y[ent], p_sql[ent]), "xgboost_local": _metricas(y[ent], p_xgb[ent])},
        "diferencia_prob_sql_vs_xgboost_local": {"prueba": _dif(p_sql[prueba], p_xgb[prueba]), "todos": _dif(p_sql, p_xgb),
                                                 "spearman_prueba": float(spearmanr(p_sql[prueba], p_xgb[prueba])[0])},
        "diferencia_prob_sql_vs_xgboost_mismos_cortes": {"prueba": _dif(p_sql[prueba], p_b2[prueba]), "todos": _dif(p_sql, p_b2)},
        "diferencia_prob_sql_vs_replica": _dif(p_sql, p_rep),
        "modelo_sql_en_libreria_xgboost_vs_numpy": _dif(p_sql_lib, p_sql),
        "modelo_sql_if_then_vs_numpy_200": _dif(p_ir, p_sql[muestra])}
    perd_x = ev["validation_1"]["logloss"]
    curva_t = []
    for k in PUNTOS_CURVA:
        if k <= p.arboles and k < len(res["historial"]):
            curva_t.append({"arboles": k, "sql_prueba": res["historial"][k]["perdida_prueba"],
                            "xgboost_prueba": perd_x[k - 1] if k >= 1 else None,
                            "sql_entrenamiento": res["historial"][k]["perdida_entrenamiento"],
                            "xgboost_entrenamiento": ev["validation_0"]["logloss"][k - 1] if k >= 1 else None})
    inf["curva"] = curva_t
    np.savez_compressed(carpeta / "predicciones.npz", p_sql=p_sql, p_xgb=p_xgb, p_rep=p_rep, prueba=prueba, y=y,
                        episode_ref=df.episode_ref.to_numpy())
    (carpeta / "informe.json").write_text(json.dumps(inf, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    (carpeta / "informe.md").write_text(informe_md(inf), encoding="utf-8")
    print(informe_md(inf))
    return inf


def informe_md(inf):
    A, Bd, R, D, P = inf["A_sql"], inf["B_descarga"], inf["C_replica"], inf["D_calidad"], inf["parametros"]
    q, e = D["prueba"], D["diferencia_prob_sql_vs_xgboost_local"]
    f = lambda x: f"{x:.4f}"
    L = [f"# Entrenar con SQL frente a descargar y entrenar XGBoost", "",
         f"Método SQL: {'todo el entrenamiento dentro de la consulta (entrenar_una.py)' if inf.get('metodo', 'niveles') == 'una' else 'una consulta por árbol y nivel (entrenar.py)'}.", "",
         f"{inf['fecha']} · {len(inf['variables'])} variables · "
         f"{A['ingresos']['entrenamiento']} ingresos de entrenamiento y {q['ingresos']} de prueba ({q['positivos']} muertes) · "
         f"{P['arboles']} árboles, profundidad {P['profundidad']}, eta {P['eta']}, lambda {P['lambda_']}, "
         f"min_child_weight {P['min_hijo']}, {P['bins']} intervalos", "",
         "## Tiempo y datos que salen del servidor", "",
         "| | A. consultas SQL | B. descarga + XGBoost |", "|---|---|---|",
         f"| Tiempo total (s) | {A['segundos']} | {Bd['segundos']} (descarga {Bd['descarga_s']} + entrenamiento "
         f"{Bd['entrenamiento_xgboost_s_mediana']}) |",
         f"| Consultas | {A['consultas']} | {Bd['consultas']} |",
         f"| Tiempo por consulta (mediana / máx., s) | {A['por_consulta_mediana_s']} / {A['por_consulta_max_s']} | "
         f"{Bd['descarga_s']} |",
         f"| Servidor: reloj / planificación / CPU (s) | {A['servidor_s']} / {A['planificacion_s']} / {A['cpu_servidor_s']} | "
         f"{Bd['descarga_servidor_s']} / - / - |",
         f"| SQL enviado (total / máx. por consulta) | {A['bytes_sql_enviados'] / 1e6:.1f} MB / {A['bytes_sql_max'] / 1024:.0f} KB | - |",
         f"| Filas de pacientes que salen | **0** (solo {A['filas_devueltas']} filas: "
         f"{'los árboles y las pérdidas' if inf.get('metodo') == 'una' else 'sumas agregadas por nodo'}) | "
         f"**{Bd['filas_de_pacientes_devueltas']}** ({Bd['valores_devueltos']:,} valores) |", "",
         f"Memoria máxima por consulta en A: {A['memoria_max_mb']} MB. XGBoost {Bd['xgboost']} con {Bd['hilos']} hilos.", "",
         "## Calidad en los pacientes de prueba", "",
         "| Modelo | AUC | Pérdida logística | Brier |", "|---|---|---|---|"]
    filas = [("A. SQL", q["sql"]), ("B. XGBoost local", q["xgboost_local"])]
    if "xgboost_mismos_cortes" in q:
        filas.append(("B2. XGBoost local con los cortes de A", q["xgboost_mismos_cortes"]))
    for nombre, m in filas + [("solo tasa base", q["solo_tasa_base"])]:
        L.append(f"| {nombre} | {f(m['auc'])} | {f(m['perdida_logistica'])} | {f(m['brier'])} |")
    L += ["", f"Diferencia de probabilidad A frente a B (prueba): máxima {e['prueba']['maxima']:.3f}, media "
              f"{e['prueba']['media']:.4f}, p95 {e['prueba']['p95']:.4f}; correlación de Spearman {e['spearman_prueba']:.4f}."]
    if "diferencia_prob_sql_vs_xgboost_mismos_cortes" in D:
        e2 = D["diferencia_prob_sql_vs_xgboost_mismos_cortes"]["prueba"]
        L.append(f"A frente a B2 (mismos cortes): máxima {e2['maxima']:.1e}, media {e2['media']:.1e}, p95 {e2['p95']:.1e}.")
    L += [
          f"Entrenamiento: AUC A {f(D['entrenamiento']['sql']['auc'])}, B {f(D['entrenamiento']['xgboost_local']['auc'])}.", "",
          "## ¿Calcula el SQL exactamente lo previsto?", "",
          f"- Réplica en numpy con los mismos cortes: {R['nodos_distintos']} nodos distintos de {R['nodos_comparados']} "
          f"({R['arboles_sql']} árboles); hojas: diferencia máxima {R['dif_max_hojas']:.1e}; probabilidades: "
          f"{D['diferencia_prob_sql_vs_replica']['maxima']:.1e}.",
          f"- Modelo de A cargado en la librería XGBoost (JSON exportado) frente a numpy: "
          f"{D['modelo_sql_en_libreria_xgboost_vs_numpy']['maxima']:.1e}; documento if-then: "
          f"{D['modelo_sql_if_then_vs_numpy_200']['maxima']:.1e}.", "",
          "## Pérdida logística por número de árboles", "",
          "| Árboles | A prueba | B prueba | A entrenamiento | B entrenamiento |", "|---|---|---|---|---|"]
    for c in inf["curva"]:
        g = lambda x: "-" if x is None else f"{x:.4f}"
        L.append(f"| {c['arboles']} | {g(c['sql_prueba'])} | {g(c['xgboost_prueba'])} | {g(c['sql_entrenamiento'])} | "
                 f"{g(c['xgboost_entrenamiento'])} |")
    return "\n".join(L) + "\n"
