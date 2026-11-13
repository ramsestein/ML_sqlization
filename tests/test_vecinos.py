"""KNN (núcleo "vecinos"): el documento + intérprete reproduce a scikit-learn y el SQL lleva la bolsa = la base."""
import numpy as np
import pytest
from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from fnxsql import binario as B, ir, sql
from fnxsql.contrato import Contrato, variables_del_modelo
from test_ir import NOMBRES, datos

PREP = lambda: [("imp", SimpleImputer(strategy="median", add_indicator=True)), ("sc", StandardScaler())]


def doc_knn(m, bolsa=None):
    vs, err = variables_del_modelo(NOMBRES)
    assert not err
    return ir.nombrar(ir.a_ir(m, "sklearn", vs, Contrato(bolsa=bolsa or {}), "knn"))


def con_bolsa_de_entrenamiento(m, X):
    d = doc_knn(m)
    Xb, yb = ir.bolsa_entrenamiento(m)
    return ir.vecinos(d, ir.entrada(d, X.to_dict("records")), Xb, yb)


@pytest.mark.parametrize("knn", [
    lambda: KNeighborsClassifier(n_neighbors=15, weights="distance"),
    lambda: KNeighborsClassifier(n_neighbors=25),
    lambda: KNeighborsClassifier(n_neighbors=10, weights="distance", metric="manhattan"),
    lambda: KNeighborsClassifier(n_neighbors=10, weights="distance", algorithm="brute", metric="minkowski", p=2,
                                 metric_params={"w": np.array([2.0, 0.5, 3.0, 1.0, 0.1, 1.0, 1.0, 0.0])}),
    lambda: KNeighborsClassifier(n_neighbors=10, algorithm="brute", metric="minkowski", p=1,
                                 metric_params={"w": np.array([2.0, 0.5, 3.0, 1.0, 0.1, 1.0, 1.0, 0.0])}),
])
def test_if_then_exacto_con_la_bolsa_de_entrenamiento(knn):
    X, y = datos()
    m = Pipeline(PREP() + [("knn", knn())]).fit(X, y)
    Xn, _ = datos(n=200, semilla=1)  # consultas nuevas, con ausentes
    assert np.abs(con_bolsa_de_entrenamiento(m, Xn) - m.predict_proba(Xn)[:, 1]).max() <= 1e-12


def test_distancia_cero():
    X, y = datos(ausentes=False)
    m = Pipeline(PREP() + [("knn", KNeighborsClassifier(n_neighbors=7, weights="distance"))]).fit(X, y)
    assert np.abs(con_bolsa_de_entrenamiento(m, X.iloc[:50]) - m.predict_proba(X.iloc[:50])[:, 1]).max() <= 1e-12


def test_excluye_al_propio_paciente_y_el_futuro():
    X, y = datos()
    rng = np.random.default_rng(3)
    pac = rng.integers(0, 150, len(X))
    t = rng.uniform(0, 100, len(X))           # momento de evaluación de cada ingreso
    fin = t + rng.uniform(1, 20, len(X))      # alta
    m = Pipeline(PREP() + [("knn", KNeighborsClassifier(n_neighbors=9, weights="distance"))]).fit(X, y)
    d = doc_knn(m, {"solo_pasado": True})
    Z = ir.entrada(d, X.to_dict("records"))
    p = ir.vecinos(d, Z[:40], Z, y, pac_q=pac[:40], pac_b=pac, t_q=t[:40], fin_b=fin)
    clf = m.named_steps["knn"]
    for i in range(40):
        ok = (pac != pac[i]) & (fin <= t[i])
        if ok.sum() < 9:
            continue
        ref = clone(clf).fit(Z[ok], y[ok]).predict_proba(Z[i:i + 1])[0, 1]
        assert abs(p[i] - ref) <= 1e-12


def test_bolsa_y_restricciones():
    X, y = datos()
    with pytest.raises(ValueError, match="SimpleImputer"):
        doc_knn(Pipeline([("sc", StandardScaler()), ("knn", KNeighborsClassifier())]).fit(X.fillna(0), y))
    with pytest.raises(ValueError, match="no reconocidas"):
        doc_knn(Pipeline(PREP() + [("knn", KNeighborsClassifier())]).fit(X, y), {"etiqueta": "x"})
    with pytest.raises(ValueError, match="no admitida"):
        doc_knn(Pipeline(PREP() + [("knn", KNeighborsClassifier(metric="cosine"))]).fit(X, y))


def test_sql_vecinos():
    X, y = datos()
    w = [2.0, 0.5, 3.0, 1.0, 0.1, 1.0, 1.0, 0.0]
    m = Pipeline(PREP() + [("knn", KNeighborsClassifier(n_neighbors=150, weights="distance", algorithm="brute",
                                                        metric="minkowski", metric_params={"w": np.array(w)}))]).fit(X, y)
    d = doc_knn(m)
    s, info = sql.compilar(d, [11, 22], "auto")
    assert info["k"] == 150 and info["ponderada"] and not info["solo_pasado"]
    assert "consulta_vars AS (" in s and "WHERE patient_ref IN (11, 22)" in s   # consulta: solo sus pacientes
    assert "patient_ref < 1000000000" in s and "mot_ref = 23" in s           # bolsa: toda la base, con etiqueta
    assert "b.patient_ref <> c.patient_ref" in s and "b.fin <= c.t_eval" not in s
    assert "row_number() OVER (PARTITION BY episode_ref ORDER BY dist, vecino)" in s and "rk <= 150" in s
    assert "sum(y / dist) / sum(1E0 / dist)" in s and s.count("sqrt(") == 1
    assert info["precision"] == "float64" and f"* {sql.lit(np.sqrt(3.0))}" in s  # KNN: float64, escala sqrt(w)
    d2 = doc_knn(m, {"solo_pasado": True, "etiqueta_sql": "SELECT episode_ref, y FROM {esquema}.mi_etiqueta"})
    s2, info2 = sql.compilar(d2, [11], "auto")
    assert "b.fin <= c.t_eval" in s2 and "datascope_gestor_prod.mi_etiqueta" in s2 and "mot_ref = 23" not in s2
    with pytest.raises(ValueError):
        sql.compilar(d, [1], "case")
    with pytest.raises(ValueError, match="bolsa"):
        ir.evaluar(d, X.iloc[0].to_dict())
