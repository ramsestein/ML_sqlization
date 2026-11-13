"""Representación intermedia if-then (formato `fnxsql_if_then_v1`): carga de modelos, exportación exacta e intérprete.

Modelos admitidos (clasificación binaria, probabilidad de la clase 1):
  XGBoost        .json nativo (save_model): gbtree, binary:logistic, cortes numéricos. Sin necesidad de xgboost instalado.
  scikit-learn   .joblib / .pkl con LogisticRegression, DecisionTreeClassifier, RandomForestClassifier,
                 HistGradientBoostingClassifier (sin variables categóricas), MLPClassifier (ReLU) o KNeighborsClassifier
                 (euclídea / manhattan / minkowski p = 1 o 2, con pesos w opcionales), solos o en un Pipeline
                 precedidos de SimpleImputer y/o StandardScaler. ¡joblib/pickle ejecuta código al cargar: solo ficheros
                 de confianza!
Las variables del modelo deben llamarse con la gramática del contrato (contrato.py): son las entradas del documento.

Documento:
  {"formato": "fnxsql_if_then_v1", "modelo": {...}, "contrato": {...},
   "variables": [{"nombre", "fuente", "codigo", "valor", "estadistico"}, ...],        entradas, en el orden del modelo
   "preproceso": [paso, ...],                                                         en orden; cada paso crea variables
      {"tipo": "imputar", "variable": v, "salida": s, "valor": m}        si v es nulo entonces m si no v
      {"tipo": "indicador_nulo", "variable": v, "salida": s}             si v es nulo entonces 1 si no 0
      {"tipo": "estandarizar", "variable": v, "salida": s, "media": m, "escala": e}   (v - m) / e
      {"tipo": "tabla", "variable": v, "salida": s, "cortes": [c_1 < ... < c_m], "valores": [m + 1],
       "en_corte": [m] | null, "nulo": n}       función escalonada: nulo -> n; v = c_i (si hay en_corte) -> en_corte[i];
                                               si no valores[#{c_i <= v}] (un valor igual a un corte va a la derecha)
      {"tipo": "interpolar", "variable": v, "salida": s, "nodos_x": [x_1 < ... < x_m] (estrictamente crecientes),
       "nodos_y": [m], "nulo": n}
                                               lineal a trozos entre nodos, constante fuera (np.interp); nulo -> n
      {"tipo": "categoria", "variable": v, "salida": s, "valores": {"codigo": valor}, "otro": o, "nulo": n}
                                               v (código numérico) -> su valor; código no listado -> o; nulo -> n
      (tabla, interpolar y categoria: funciones de forma de una métrica aprendida, p. ej. PERISIM, ver perisim.py)
   "entrada_nucleo": [nombres],                                          vector que recibe el núcleo
   "nucleo": uno de
      {"tipo": "puntuacion", "base": b, "reglas": [{"si": [cond], "entonces": {"suma": c} | {"suma_por": [var, coef]}}]}
            logit = base + suma de las reglas que se cumplen; probabilidad = sigmoide
      {"tipo": "arboles", "precision": "float64" | "float32", "combinacion": "suma_sigmoide" | "media", "base": b,
       "arboles": [nodo, ...]}   nodo = {"hoja": valor} | {"si": {"variable", "op": "<=" | "<", "valor", "nulo_a_entonces"},
                                                           "entonces": nodo, "si_no": nodo}
            float32 (XGBoost): float32(x) < valor y la suma de hojas en float32; float64 (scikit-learn): x <= valor en double
            (los umbrales de los árboles de scikit-learn, que compara en float32, se reescriben a su equivalente double y el
            núcleo lleva "entrada_float32": true, para que el SQL en float32 compare como scikit-learn: float32(x) <= t32)
      {"tipo": "red", "capas": [{"pesos": [[...] por neurona], "sesgos": [...], "activacion": "relu" | "identidad"}]}
            relu: si z > 0 entonces z si no 0; la última capa da el logit; probabilidad = sigmoide
      {"tipo": "vecinos", "k": k, "ponderacion": "uniforme" | "distancia" | "suavizada", "p": 1 | 2,
       "pesos": [w por entrada] | null, "alfa": a, "prior": pi | null,                       (alfa / prior: solo suavizada)
       "bolsa": {"etiqueta_sql": sql | null, "solo_pasado": bool, "excluir_mismo_paciente": bool,
                 "un_vecino_por_paciente": bool}}
            el "modelo" de un KNN son sus datos: la bolsa NO viaja en el documento, es la propia base (FakeNex / DataNex)
            del hospital donde se ejecuta. distancia = (suma w_j |x_j - y_j|^p)^(1/p) sobre la entrada del núcleo; los
            k ingresos más cercanos (empates por episode_ref) de otros pacientes con etiqueta conocida; probabilidad =
            proporción de y = 1 (uniforme) o media ponderada por 1 / distancia (si hay vecinos a distancia 0, la media de
            esos, como scikit-learn). etiqueta_sql: SELECT episode_ref, y (0 / 1) de los ingresos de la bolsa ({esquema}
            y {base} se sustituyen); null = muerte durante el ingreso (alta por exitus). solo_pasado: solo vecinos dados
            de alta antes del momento de evaluación del ingreso consultado (sin información del futuro).
            suavizada (PERISIM): p = (n_1 + 2 alfa pi) / (n + 2 alfa), n = vecinos encontrados, n_1 los de y = 1, pi =
            prior o, si es null, la proporción de y = 1 en la bolsa; sin vecinos -> pi. un_vecino_por_paciente: cada
            paciente de la bolsa cuenta una vez, por su ingreso más cercano (empates por episode_ref).
  cond = {"variable", "op": "es_nulo" | "no_es_nulo" | "<=" | ">" | "==" | "en", "valor"/"valores", "o_nulo"}
"""
import json
import math
from pathlib import Path

import numpy as np

FORMATO = "fnxsql_if_then_v1"


# ------------------------------------------------------------------ carga y nombres
def cargar(ruta):
    """(objeto, tipo) con tipo "xgboost" (dict del JSON) o "sklearn" (estimador o Pipeline)."""
    ruta = Path(ruta)
    if ruta.suffix.lower() == ".json":
        d = json.loads(ruta.read_text(encoding="utf-8"))
        if "learner" not in d:
            raise ValueError(f"{ruta}: no parece un modelo XGBoost guardado con save_model (falta 'learner')")
        return d, "xgboost"
    import joblib
    return joblib.load(ruta), "sklearn"


def nombres(obj, tipo):
    if tipo == "xgboost":
        n = obj["learner"].get("feature_names") or []
        if not n:
            raise ValueError("el modelo XGBoost no trae nombres de variables: usa `fnxsql normalizar --nombres metadatos.json`")
        return list(n)
    n = getattr(obj, "feature_names_in_", None)
    if n is None:
        raise ValueError("el modelo de scikit-learn no trae feature_names_in_ (entrénalo con un DataFrame con los nombres)")
    return [str(x) for x in n]


def predecir_nativo(obj, tipo, X):
    """Probabilidad de la clase 1 con la librería original (X: DataFrame con los nombres del modelo)."""
    if tipo == "sklearn":
        return obj.predict_proba(X)[:, list(obj.classes_).index(1)]
    import xgboost as xgb
    b = xgb.Booster()
    b.load_model(bytearray(json.dumps(obj).encode("utf-8")))
    return b.predict(xgb.DMatrix(X.to_numpy(dtype=np.float64), feature_names=list(X.columns), missing=np.nan))


# ------------------------------------------------------------------ utilidades numéricas
def umbral_f64(t):
    """Umbral double equivalente a un corte de árbol de scikit-learn, que compara float32(x) <= t: con t32 el mayor float32
    <= t, float32(x) <= t equivale a x <= punto medio entre t32 y el float32 siguiente (salvo empate exacto)."""
    if math.isinf(t):
        return float(t)
    t32 = np.float32(t)
    if float(t32) > t:
        t32 = np.nextafter(t32, np.float32(-np.inf))
    return (float(t32) + float(np.nextafter(t32, np.float32(np.inf)))) / 2


MAX_DOUBLE = 1.7976931348623157e308  # x <= MAX_DOUBLE es cierto para todo valor presente (cortes "presente / ausente")


def finito(t):
    """Umbral sin infinitos (JSON estándar): +inf -> el mayor double, que equivale para cualquier valor presente."""
    return MAX_DOUBLE if math.isinf(t) and t > 0 else (-MAX_DOUBLE if math.isinf(t) else float(t))


def sigmoide(z):
    return 1.0 / (1.0 + math.exp(-z)) if z >= 0 else math.exp(z) / (1.0 + math.exp(z))


# ------------------------------------------------------------------ exportación
def a_ir(obj, tipo, variables, contrato, nombre_modelo):
    """Documento if-then exacto del modelo. `variables`: lista de contrato.Variable en el orden del modelo."""
    doc = {"formato": FORMATO, "modelo": {"nombre": nombre_modelo, "origen": tipo}, "contrato": contrato.a_dict(),
           "variables": [{k: getattr(v, k) for k in ("nombre", "fuente", "codigo", "valor", "estadistico", "constante")
                          if getattr(v, k) is not None} for v in variables],
           "preproceso": []}
    actuales = [v.nombre for v in variables]
    if tipo == "xgboost":
        doc["modelo"]["clase"] = "XGBoost"
        doc["entrada_nucleo"], doc["nucleo"] = actuales, _xgboost(obj)
        return doc
    from sklearn.pipeline import Pipeline
    pasos = obj.steps if isinstance(obj, Pipeline) else [("modelo", obj)]
    *previos, (_, est) = pasos
    for _, p in previos:
        actuales = _preproceso(p, actuales, doc["preproceso"])
    doc["modelo"]["clase"] = type(est).__name__
    doc["entrada_nucleo"] = actuales
    doc["nucleo"] = _nucleo_sklearn(est)
    if doc["nucleo"]["tipo"] == "vecinos":
        doc["nucleo"]["bolsa"] = bolsa_de(contrato)
        if contrato.si_falta is None and not any(p["tipo"] == "imputar" for p in doc["preproceso"]):
            raise ValueError("KNN sin SimpleImputer y con si_falta = null: un ausente daría distancias nulas "
                             "(pon un SimpleImputer en el Pipeline o un si_falta numérico en el contrato)")
    return doc


BOLSA = {"etiqueta_sql": None, "solo_pasado": False, "excluir_mismo_paciente": True, "un_vecino_por_paciente": False}


def bolsa_de(contrato):
    """Definición de la bolsa de un KNN: la del contrato ("bolsa": {...}) sobre los valores por defecto."""
    d = dict(BOLSA)
    extra = getattr(contrato, "bolsa", None) or {}
    desconocidas = set(extra) - set(BOLSA)
    if desconocidas:
        raise ValueError(f"claves de 'bolsa' no reconocidas: {sorted(desconocidas)} (admitidas: {sorted(BOLSA)})")
    d.update(extra)
    return d


def _preproceso(p, entrada, pasos):
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    if isinstance(p, SimpleImputer):
        if p.strategy not in ("mean", "median", "most_frequent", "constant"):
            raise ValueError(f"SimpleImputer(strategy={p.strategy!r}) no admitido")
        salida = []
        for v, m in zip(entrada, p.statistics_):
            s = f"imp({v})"
            pasos.append({"tipo": "imputar", "variable": v, "salida": s, "valor": float(m)})
            salida.append(s)
        if p.indicator_ is not None:
            for j in p.indicator_.features_:
                s = f"nulo({entrada[j]})"
                pasos.append({"tipo": "indicador_nulo", "variable": entrada[j], "salida": s})
                salida.append(s)
        return salida
    if isinstance(p, StandardScaler):
        mu = p.mean_ if p.with_mean else np.zeros(len(entrada))
        sc = p.scale_ if p.with_std else np.ones(len(entrada))
        salida = []
        for v, m, e in zip(entrada, mu, sc):
            s = f"est({v})"
            pasos.append({"tipo": "estandarizar", "variable": v, "salida": s, "media": float(m), "escala": float(e)})
            salida.append(s)
        return salida
    raise ValueError(f"paso de Pipeline no admitido: {type(p).__name__} (solo SimpleImputer y StandardScaler)")


def _xgboost(d):
    L = d["learner"]
    if L["objective"]["name"] != "binary:logistic":
        raise ValueError(f"objetivo {L['objective']['name']!r} no admitido (solo binary:logistic)")
    gb = L["gradient_booster"]
    if gb.get("name", "gbtree") != "gbtree":
        raise ValueError("solo boosters gbtree")
    base = float(str(L["learner_model_param"]["base_score"]).strip("[]"))
    arboles = []
    for t in gb["model"]["trees"]:
        if any(int(s) != 0 for s in t["split_type"]):
            raise ValueError("cortes categóricos de XGBoost no admitidos")
        Lc, Rc, S, Cn, DL = t["left_children"], t["right_children"], t["split_indices"], t["split_conditions"], t["default_left"]

        def nodo(i):
            if Lc[i] == -1:
                return {"hoja": float(np.float32(Cn[i]))}
            return {"si": {"variable_indice": int(S[i]), "op": "<", "valor": float(np.float32(Cn[i])), "nulo_a_entonces": bool(DL[i])},
                    "entonces": nodo(Lc[i]), "si_no": nodo(Rc[i])}
        arboles.append(nodo(0))
    return {"tipo": "arboles", "precision": "float32", "combinacion": "suma_sigmoide",
            "base": float(np.float32(math.log(base / (1 - base)))), "arboles": arboles}


def _nucleo_sklearn(est):
    from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.tree import DecisionTreeClassifier
    if list(est.classes_) != [0, 1] and len(est.classes_) != 2:
        raise ValueError("solo clasificación binaria")
    k1 = list(est.classes_).index(est.classes_[1])
    if isinstance(est, LogisticRegression):
        coef, b = est.coef_[0], float(est.intercept_[0])
        return {"tipo": "puntuacion", "base": b,
                "reglas": [{"si": [{"variable_indice": j, "op": "no_es_nulo"}], "entonces": {"suma_por": [j, float(c)]}}
                           for j, c in enumerate(coef)]}
    if isinstance(est, (RandomForestClassifier, DecisionTreeClassifier)):
        ests = est.estimators_ if isinstance(est, RandomForestClassifier) else [est]
        return {"tipo": "arboles", "precision": "float64", "combinacion": "media", "base": 0.0, "entrada_float32": True,
                "arboles": [_arbol_sklearn(e.tree_, k1) for e in ests]}
    if isinstance(est, HistGradientBoostingClassifier):
        if est.is_categorical_ is not None and np.any(est.is_categorical_):
            raise ValueError("HistGradientBoosting con variables categóricas no admitido")
        pp = getattr(est, "_preprocessor", None)
        orden = list(range(est.n_features_in_))
        if pp is not None:  # versiones recientes: preprocesador interno que puede reordenar columnas
            orden = []
            for nombre, trans, mask in pp.transformers_:
                if nombre != "remainder" and trans != "drop":
                    orden += [i for i, on in enumerate(mask) if on]
        arboles = []
        for (tree,) in est._predictors:
            nd = tree.nodes

            def nodo(i):
                n = nd[i]
                if n["is_leaf"]:
                    return {"hoja": float(n["value"])}
                t = finito(float(n["num_threshold"]))
                return {"si": {"variable_indice": orden[int(n["feature_idx"])], "op": "<=", "valor": t,
                               "nulo_a_entonces": bool(n["missing_go_to_left"])},
                        "entonces": nodo(int(n["left"])), "si_no": nodo(int(n["right"]))}
            arboles.append(nodo(0))
        return {"tipo": "arboles", "precision": "float64", "combinacion": "suma_sigmoide",
                "base": float(np.ravel(est._baseline_prediction)[0]), "arboles": arboles}
    if isinstance(est, MLPClassifier):
        if est.activation != "relu" or est.out_activation_ != "logistic":
            raise ValueError("MLPClassifier: solo activation='relu' con salida logística (binaria)")
        capas = [{"pesos": [[float(x) for x in W[:, n]] for n in range(W.shape[1])], "sesgos": [float(x) for x in b],
                  "activacion": "relu"} for W, b in zip(est.coefs_[:-1], est.intercepts_[:-1])]
        W, b = est.coefs_[-1], est.intercepts_[-1]
        capas.append({"pesos": [[float(x) for x in W[:, 0]]], "sesgos": [float(b[0])], "activacion": "identidad"})
        return {"tipo": "red", "capas": capas}
    if isinstance(est, KNeighborsClassifier):
        return _knn(est)
    raise ValueError(f"modelo no admitido: {type(est).__name__}")


def _knn(est):
    if est.weights not in ("uniform", "distance"):
        raise ValueError(f"KNeighborsClassifier(weights={est.weights!r}) no admitido (solo 'uniform' o 'distance')")
    met = est.effective_metric_
    par = dict(est.effective_metric_params_ or {})
    w = par.pop("w", None)
    if met == "euclidean":
        p = 2
    elif met == "manhattan":
        p = 1
    elif met == "minkowski":
        p = par.pop("p", est.p)
    else:
        raise ValueError(f"métrica {met!r} no admitida (euclidean, manhattan o minkowski con p = 1 o 2)")
    if p not in (1, 2) or par:
        raise ValueError(f"métrica minkowski con p = {p} y parámetros {par} no admitida (p = 1 o 2, pesos w opcionales)")
    if w is not None:
        w = [float(x) for x in np.ravel(w)]
        if len(w) != est.n_features_in_ or min(w) < 0:
            raise ValueError("pesos w del KNN: uno por variable de entrada y no negativos")
    return {"tipo": "vecinos", "k": int(est.n_neighbors), "ponderacion": {"uniform": "uniforme", "distance": "distancia"}[est.weights],
            "p": int(p), "pesos": w}


def _arbol_sklearn(t_, k1):
    def nodo(i):
        if t_.children_left[i] == -1:
            v = t_.value[i][0]
            return {"hoja": float(v[k1] / v.sum())}
        thr = float(t_.threshold[i])
        return {"si": {"variable_indice": int(t_.feature[i]), "op": "<=", "valor": finito(umbral_f64(thr)),
                       "nulo_a_entonces": bool(t_.missing_go_to_left[i]) if hasattr(t_, "missing_go_to_left") else False},
                "entonces": nodo(int(t_.children_left[i])), "si_no": nodo(int(t_.children_right[i]))}
    return nodo(0)


def nombrar(doc):
    """Sustituye variable_indice por el nombre de la variable de entrada del núcleo (más legible); devuelve doc."""
    ent = doc["entrada_nucleo"]

    def rec(n):
        if "si" in n:
            n["si"]["variable"] = ent[n["si"].pop("variable_indice")]
            rec(n["entonces"])
            rec(n["si_no"])
    nuc = doc["nucleo"]
    if nuc["tipo"] == "arboles":
        for a in nuc["arboles"]:
            rec(a)
    elif nuc["tipo"] == "puntuacion":
        for r in nuc["reglas"]:
            for c in r["si"]:
                c["variable"] = ent[c.pop("variable_indice")]
            if "suma_por" in r["entonces"]:
                r["entonces"]["suma_por"][0] = ent[r["entonces"]["suma_por"][0]]
    return doc


def resumen(doc):
    nuc = doc["nucleo"]
    if nuc["tipo"] == "arboles":
        nodos = [0]

        def cuenta(n):
            nodos[0] += 1
            if "si" in n:
                cuenta(n["entonces"])
                cuenta(n["si_no"])
        prof = lambda n: 0 if "hoja" in n else 1 + max(prof(n["entonces"]), prof(n["si_no"]))
        for a in nuc["arboles"]:
            cuenta(a)
        return {"tipo": "arboles", "arboles": len(nuc["arboles"]), "nodos": nodos[0],
                "profundidad": max(prof(a) for a in nuc["arboles"]), "precision": nuc["precision"]}
    if nuc["tipo"] == "red":
        return {"tipo": "red", "capas": [len(c["sesgos"]) for c in nuc["capas"]],
                "pesos": sum(len(p) for c in nuc["capas"] for p in c["pesos"])}
    if nuc["tipo"] == "vecinos":
        return {"tipo": "vecinos", "k": nuc["k"], "ponderacion": nuc["ponderacion"], "p": nuc["p"],
                "ponderada": nuc["pesos"] is not None, "dimensiones": len(doc["entrada_nucleo"]), **nuc["bolsa"]}
    return {"tipo": "puntuacion", "reglas": len(nuc["reglas"])}


# ------------------------------------------------------------------ intérprete
def _nulo(x):
    return x is None or (isinstance(x, float) and math.isnan(x))


def _preprocesar(doc, fila):
    x = {v["nombre"]: (None if _nulo(fila.get(v["nombre"])) else float(fila[v["nombre"]])) for v in doc["variables"]}
    for p in doc["preproceso"]:
        v = x[p["variable"]]
        if p["tipo"] == "imputar":
            x[p["salida"]] = p["valor"] if v is None else v
        elif p["tipo"] == "indicador_nulo":
            x[p["salida"]] = 1.0 if v is None else 0.0
        elif p["tipo"] in FORMAS:
            x[p["salida"]] = forma(p, v)
        else:
            x[p["salida"]] = None if v is None else (v - p["media"]) / p["escala"]
    return x


FORMAS = ("tabla", "interpolar", "categoria")


def forma(p, v):
    """Valor de un paso tabla / interpolar / categoria para v (None = ausente)."""
    if v is None:
        return float(p["nulo"])
    if p["tipo"] == "tabla":
        c = p["cortes"]
        if p.get("en_corte") is not None:
            for ci, e in zip(c, p["en_corte"]):
                if v == ci:
                    return float(e)
        return float(p["valores"][int(np.searchsorted(np.asarray(c, float), v, side="right"))])
    if p["tipo"] == "interpolar":
        if np.any(np.diff(p["nodos_x"]) <= 0):
            raise ValueError("interpolar: nodos_x estrictamente crecientes")
        return float(np.interp(v, p["nodos_x"], p["nodos_y"]))
    for cod, val in p["valores"].items():
        if v == float(cod):
            return float(val)
    return float(p["otro"])


def entrada(doc, filas):
    """Matriz (n x d) de la entrada del núcleo (tras el preproceso del documento) para una lista de dicts de variables;
    ausentes sin imputar = NaN."""
    ent = doc["entrada_nucleo"]
    return np.array([[np.nan if (z := _preprocesar(doc, f)[n]) is None else z for n in ent] for f in filas], dtype=float).reshape(-1, len(ent))


def evaluar(doc, fila):
    """Probabilidad según el documento, a partir de un dict nombre de variable -> valor (None / NaN = ausente).
    Un núcleo "vecinos" necesita su bolsa: usa vecinos()."""
    x = _preprocesar(doc, fila)
    nuc = doc["nucleo"]
    if nuc["tipo"] == "vecinos":
        raise ValueError("un KNN no se evalúa fila a fila: necesita su bolsa (ir.vecinos)")
    if nuc["tipo"] == "puntuacion":
        z = nuc["base"]
        for r in nuc["reglas"]:
            if all(_cumple(c, x) for c in r["si"]):
                e = r["entonces"]
                z += e["suma"] if "suma" in e else e["suma_por"][1] * x[e["suma_por"][0]]
        return sigmoide(z)
    if nuc["tipo"] == "red":
        a = [x[n] for n in doc["entrada_nucleo"]]
        for capa in nuc["capas"]:
            z = [b + sum(w * v for w, v in zip(fila_w, a)) for fila_w, b in zip(capa["pesos"], capa["sesgos"])]
            a = [zi if (capa["activacion"] == "identidad" or zi > 0) else 0.0 for zi in z]  # ReLU: si z > 0 entonces z si no 0
        return sigmoide(a[0])
    f32 = nuc["precision"] == "float32"
    ef32 = bool(nuc.get("entrada_float32"))
    hojas = [_recorre(a, x, f32, ef32) for a in nuc["arboles"]]
    if nuc["combinacion"] == "media":
        return float(sum(hojas) / len(hojas))
    if f32:
        s = np.float32(nuc["base"])
        for h in hojas:
            s = np.float32(s + np.float32(h))
        return float(np.float32(1) / (np.float32(1) + np.exp(-s)))
    return sigmoide(nuc["base"] + sum(hojas))


def _recorre(n, x, f32, ef32=False):
    """ef32: el árbol compara float32(x) (scikit-learn); con el umbral double equivalente o con el float32 de
    binario.cuantizar el resultado es el mismo. La comparación va en double (NumPy 2 bajaría el umbral a float32)."""
    while "si" in n:
        c = n["si"]
        v = x[c["variable"]]
        if v is None:
            ok = c["nulo_a_entonces"]
        elif c["op"] == "<":
            ok = (np.float32(v) < np.float32(c["valor"])) if f32 else v < c["valor"]
        else:
            ok = (float(np.float32(v)) if ef32 else v) <= c["valor"]
        n = n["entonces"] if ok else n["si_no"]
    return n["hoja"]


def _cumple(c, x):
    v = x[c["variable"]]
    op = c["op"]
    if op == "es_nulo":
        return v is None
    if op == "no_es_nulo":
        return v is not None
    if v is None:
        return bool(c.get("o_nulo", False))
    return {"<=": lambda: v <= c["valor"], ">": lambda: v > c["valor"], "==": lambda: v == c["valor"],
            "en": lambda: v in c["valores"]}[op]()


def escala_vecinos(nuc, d):
    """Factor por dimensión que el SQL aplica a cada coordenada: sqrt(w) con p = 2 y w con p = 1 (así la distancia
    ponderada es la distancia ordinaria entre vectores escalados); 1 sin pesos."""
    if nuc.get("pesos") is None:
        return np.ones(d)
    w = np.asarray(nuc["pesos"], dtype=float)
    return np.sqrt(w) if nuc["p"] == 2 else w


def vecinos(doc, Xq, Xb, yb, ids_b=None, pac_q=None, pac_b=None, t_q=None, fin_b=None, indices=False):
    """Probabilidad del KNN del documento para cada fila de Xq (entrada del núcleo, ver entrada()) con la bolsa Xb / yb.
    ids_b: desempate (episode_ref; por defecto el orden de la bolsa). pac_q / pac_b: patient_ref (excluye los vecinos del
    mismo paciente si la bolsa lo pide). t_q / fin_b: momento de evaluación de la consulta y alta del vecino (solo_pasado).
    Sin vecinos admisibles -> NaN. Misma regla que el SQL y que scikit-learn. indices=True: (probabilidades, lista con
    los índices de la bolsa de los vecinos de cada fila)."""
    nuc = doc["nucleo"]
    bolsa = {**BOLSA, **nuc.get("bolsa", {})}
    Xq, Xb, yb = np.asarray(Xq, float), np.asarray(Xb, float), np.asarray(yb, float)
    prior = nuc.get("prior")
    prior = float(yb.mean()) if prior is None and len(yb) else prior
    s = escala_vecinos(nuc, Xb.shape[1])
    Zq, Zb = Xq * s, Xb * s
    ids_b = np.arange(len(Xb)) if ids_b is None else np.asarray(ids_b)
    out = np.full(len(Xq), np.nan)
    elegidos = [np.array([], int)] * len(Xq)
    for i in range(len(Xq)):
        ok = np.ones(len(Xb), bool)
        if pac_q is not None and bolsa.get("excluir_mismo_paciente", True):
            ok &= np.asarray(pac_b) != pac_q[i]
        if t_q is not None and bolsa.get("solo_pasado"):
            ok &= np.asarray(fin_b) <= t_q[i]
        idx = np.flatnonzero(ok)
        if not len(idx):
            if nuc["ponderacion"] == "suavizada":
                out[i] = prior
            continue
        dif = Zb[idx] - Zq[i]
        d = np.sqrt((dif * dif).sum(1)) if nuc["p"] == 2 else np.abs(dif).sum(1)
        orden = np.lexsort((ids_b[idx], d))
        if bolsa["un_vecino_por_paciente"]:  # el ingreso más cercano de cada paciente (el primero en el orden)
            _, primero = np.unique(np.asarray(pac_b)[idx][orden], return_index=True)
            orden = orden[np.sort(primero)]
        orden = orden[: nuc["k"]]
        dk, yk = d[orden], yb[idx][orden]
        elegidos[i] = idx[orden]
        if nuc["ponderacion"] == "suavizada":
            a = float(nuc["alfa"])
            out[i] = (yk.sum() + 2 * a * prior) / (len(yk) + 2 * a)
        elif nuc["ponderacion"] == "uniforme":
            out[i] = yk.mean()
        elif (dk == 0).any():
            out[i] = yk[dk == 0].mean()
        else:
            out[i] = (yk / dk).sum() / (1 / dk).sum()
    return (out, elegidos) if indices else out


def bolsa_entrenamiento(obj):
    """(Xb, yb) del KNN ajustado, en el espacio de la entrada del núcleo (para comprobar el documento frente al modelo)."""
    from sklearn.pipeline import Pipeline
    est = obj.steps[-1][1] if isinstance(obj, Pipeline) else obj
    return np.asarray(est._fit_X, float), (np.asarray(est._y) == 1).astype(float)
