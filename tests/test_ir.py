"""El documento if-then reproduce cada tipo de modelo admitido (sin servidor)."""
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from fnxsql import ir
from fnxsql.contrato import Contrato, variables_del_modelo

NOMBRES = ["FC_mean", "TAS_min", "LAB2467_max", "ESTADO_CONCIENC_ESTADO_CONCIENC_1_last", "age"]


def datos(n=600, ausentes=True, semilla=0):
    rng = np.random.default_rng(semilla)
    X = pd.DataFrame({"FC_mean": rng.normal(80, 15, n), "TAS_min": rng.normal(120, 20, n),
                      "LAB2467_max": np.round(rng.lognormal(0, 0.4, n), 2), "ESTADO_CONCIENC_ESTADO_CONCIENC_1_last":
                      rng.integers(0, 2, n).astype(float), "age": np.round(rng.uniform(20, 95, n), 1)})
    y = (0.03 * (X.age - 60) + 1.5 * (X.LAB2467_max - 1) - 0.02 * (X.TAS_min - 120) + rng.normal(0, 1, n) > 1).astype(int)
    if ausentes:
        for c in ["FC_mean", "TAS_min", "LAB2467_max"]:
            X.loc[rng.random(n) < 0.1, c] = np.nan
    return X, y


def doc_de(m):
    vs, err = variables_del_modelo(NOMBRES)
    assert not err
    return ir.nombrar(ir.a_ir(m, "sklearn", vs, Contrato(), "prueba"))


def comparar(m, X, tol=1e-12):
    d = doc_de(m)
    p = np.array([ir.evaluar(d, r) for r in X.to_dict("records")])
    assert np.abs(p - m.predict_proba(X)[:, 1]).max() <= tol


PREP = lambda: [("imp", SimpleImputer(strategy="median", add_indicator=True)), ("sc", StandardScaler())]


@pytest.mark.parametrize("modelo, ausentes", [
    (lambda: Pipeline(PREP() + [("lr", LogisticRegression(max_iter=2000))]), True),
    (lambda: DecisionTreeClassifier(max_depth=6, random_state=0), True),
    (lambda: RandomForestClassifier(n_estimators=20, max_leaf_nodes=16, random_state=0), True),
    (lambda: HistGradientBoostingClassifier(max_iter=40, random_state=0), True),
    (lambda: Pipeline(PREP() + [("mlp", MLPClassifier(hidden_layer_sizes=(8, 4), max_iter=300, random_state=0))]), True),
])
def test_if_then_exacto(modelo, ausentes):
    X, y = datos(ausentes=ausentes)
    m = modelo().fit(X, y)
    comparar(m, X)


def test_xgboost_float32():
    xgb = pytest.importorskip("xgboost")
    X, y = datos()
    m = xgb.XGBClassifier(n_estimators=50, max_depth=3, random_state=0).fit(X, y)
    import json
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as d:
        m.save_model(Path(d) / "m.json")
        obj, tipo = ir.cargar(Path(d) / "m.json")
    vs, _ = variables_del_modelo(ir.nombres(obj, tipo))
    doc = ir.nombrar(ir.a_ir(obj, tipo, vs, Contrato(), "xgb"))
    assert json.dumps(doc)  # JSON estándar, sin Infinity
    p = np.array([ir.evaluar(doc, r) for r in X.to_dict("records")])
    assert np.abs(p - ir.predecir_nativo(obj, tipo, X)).max() <= 1e-6


def test_umbral_f64_reproduce_float32():
    # 1.16 está exactamente en el umbral float32(1.16): scikit-learn va a la izquierda; en double hay que reescribirlo
    t = float(np.float32(1.16))
    assert 1.16 > t and 1.16 <= ir.umbral_f64(t)
