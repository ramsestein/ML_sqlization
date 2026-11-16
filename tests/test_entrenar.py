"""Entrenamiento con SQL, sin servidor: el algoritmo local, la exportación (if-then / XGBoost) y la consulta generada."""
import json

import numpy as np
import pytest

from fnxsql import binario as B, config, contrato as K, entrenar as E, ir, sql as S

NOMBRES = ["FC_mean", "TAS_last", "LAB2467_max", "age", "sex_2"]


def datos(n=600, semilla=0):
    rng = np.random.default_rng(semilla)
    X = np.column_stack([rng.normal(80, 15, n), rng.normal(120, 20, n), rng.lognormal(0, 0.5, n), rng.uniform(18, 95, n),
                         rng.integers(0, 2, n).astype(float)])
    X[rng.random(X.shape) < 0.15] = np.nan
    z = 0.04 * (np.nan_to_num(X[:, 0], nan=80) - 80) + 1.2 * np.log(np.nan_to_num(X[:, 2], nan=1)) + 0.03 * (X[:, 3] - 60) - 2
    y = (rng.random(n) < 1 / (1 + np.exp(-z))).astype(float)
    pids = rng.integers(1_000_000, 9_000_000, n)
    return X, y, pids


def modelo(p=None):
    p = p or E.Parametros(arboles=8, profundidad=3, eta=0.3, bins=32)
    X, y, pids = datos()
    prueba = E.es_prueba(pids, p)
    ent = ~prueba
    cortes = [E.cortes_de(np.nanquantile(X[ent, j], np.arange(1, p.bins) / p.bins)) for j in range(X.shape[1])]
    tm = E.margen_base(y[ent].sum(), ent.sum())
    arboles, hist = E.entrenar_local(X, y, prueba, cortes, tm, NOMBRES, p)
    vs, _ = K.variables_del_modelo(NOMBRES)
    doc = E.documento(arboles, tm[1], vs, K.Contrato(), "prueba", p)
    return X, y, prueba, cortes, tm, arboles, hist, doc


def test_particion_como_en_sql():
    p = E.Parametros(prueba=0.2)
    pids = np.arange(1_000_000, 1_100_000)
    m = E.es_prueba(pids, p)
    assert abs(m.mean() - 0.2) < 0.01
    assert m[0] == (((1_000_000 * 2654435761) % 2 ** 32) < int(round(0.2 * 2 ** 32)))


def test_entrena_y_mejora():
    X, y, prueba, cortes, tm, arboles, hist, doc = modelo()
    assert len(arboles) == 8 and len(hist) == 9
    assert hist[-1]["perdida_entrenamiento"] < hist[0]["perdida_entrenamiento"]
    assert all("hoja" in a or a["si"]["op"] == "<" for a in arboles)


def test_exporta_if_then_y_xgboost():
    xgb = pytest.importorskip("xgboost")
    X, y, prueba, cortes, tm, arboles, hist, doc = modelo()
    p_np = 1 / (1 + np.exp(-E.margen_modelo(arboles, tm[1], X, NOMBRES).astype(float)))
    p_ir = np.array([ir.evaluar(doc, dict(zip(NOMBRES, x))) for x in X])
    assert np.abs(p_ir - p_np).max() < 1e-6
    js = E.a_xgboost(doc, tm[0])
    b = xgb.Booster()
    b.load_model(bytearray(json.dumps(js).encode("utf-8")))
    p_lib = b.predict(xgb.DMatrix(X, feature_names=NOMBRES, missing=np.nan))
    assert np.abs(p_lib - p_np).max() < 1e-6
    # el JSON exportado es un XGBoost como los demás para fnxsql: mismo documento if-then
    vs, _ = K.variables_del_modelo(NOMBRES)
    d2 = ir.nombrar(ir.a_ir(js, "xgboost", vs, K.Contrato(), "x"))
    assert np.abs(np.array([ir.evaluar(d2, dict(zip(NOMBRES, x))) for x in X[:100]]) - p_np[:100]).max() < 1e-6


def test_intervalos_como_width_bucket():
    cortes = [[1.0, 2.0, 3.0]]
    X = np.array([[0.5], [1.0], [1.5], [2.0], [3.0], [9.0], [np.nan]])
    assert E.intervalos(X, cortes)[:, 0].tolist() == [0, 1, 1, 2, 3, 3, -1]  # medido en FakeNex con width_bucket


def test_consulta_del_paso():
    X, y, prueba, cortes, tm, arboles, hist, doc = modelo()
    p = E.Parametros(arboles=8, profundidad=3, eta=0.3, bins=32)
    vs, _ = K.variables_del_modelo(NOMBRES)
    base = E.sql_base(vs, K.Contrato(), p)
    parcial = {0: {"j": 2, "b": 5, "izq": True, "ganancia": 1.0}}
    sql = E.sql_paso(base, E.sql_cortes(cortes), [len(c) for c in cortes], NOMBRES, arboles, tm[1], parcial, [1, 2], p)
    assert "from_base64(" in sql and "width_bucket(" in sql and "max_by(ROW(j, bin, izq" in sql
    assert "(CASE WHEN element_at(bb, 3) <= 5 THEN 1 ELSE 2 END)" in sql
    assert len(sql.encode()) < config.MAX_SQL_BYTES
    # cada CTE pesado se usa una sola vez (Trino no materializa los CTE: si no, recalcularía las variables)
    for cte in ("FROM t)", "FROM s0)", "FROM gh ", "FROM hist ", "FROM win ", "FROM cand\n"):
        assert sql.count(cte) == 1, cte
    assert "FROM s)" in sql and sql.count("FROM s)") == 1
    sin = E.sql_paso(base, E.sql_cortes(cortes), [len(c) for c in cortes], NOMBRES, [], tm[1], {}, [0], p)
    assert "arboles a" not in sin and "REAL '" in sin


def test_nodo_con_ausentes_a_la_derecha():
    parcial = {0: {"j": 0, "b": 3, "izq": False, "ganancia": 1.0}}
    assert E.sql_nodo(parcial, {1, 2}) == "(CASE WHEN element_at(bb, 1) BETWEEN 0 AND 3 THEN 1 ELSE 2 END)"
    assert E.sql_nodo(parcial, {2}) == "(CASE WHEN element_at(bb, 1) BETWEEN 0 AND 3 THEN NULL ELSE 2 END)"
