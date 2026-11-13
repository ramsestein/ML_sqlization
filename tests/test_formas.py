"""Pasos tabla / interpolar / categoria (funciones de forma de una métrica aprendida) y KNN suavizado con un vecino por
paciente (PERISIM): intérprete frente a la definición y SQL de los pasos frente al intérprete (en DuckDB)."""
import numpy as np
import pytest

from fnxsql import ir, sql

TABLA = {"tipo": "tabla", "variable": "x", "salida": "t", "cortes": [-1.0, 0.0, 2.5], "valores": [0.1, 0.2, 0.3, 0.4],
         "en_corte": None, "nulo": 0.25}
RANGO = {"tipo": "tabla", "variable": "x", "salida": "r", "cortes": [-1.0, 0.0, 2.5], "valores": [0.0, 0.3, 0.6, 1.0],
         "en_corte": [0.15, 0.45, 0.8], "nulo": 0.45}
INTERP = {"tipo": "interpolar", "variable": "x", "salida": "i", "nodos_x": [-1.0, 0.0, 0.5, 2.5],
          "nodos_y": [0.0, 1.0, 1.5, 4.0], "nulo": 1.0}
CATEG = {"tipo": "categoria", "variable": "x", "salida": "c", "valores": {"1": 0.7, "2": -0.3, "5": 0.0}, "otro": 0.1,
         "nulo": 0.2}
VALORES = [None, -3.0, -1.0, -0.5, 0.0, 1.0, 2.5, 7.0, 2.0, 5.0, 3.0]


def test_tabla_igual_que_searchsorted():
    for v in VALORES:
        esperado = TABLA["nulo"] if v is None else TABLA["valores"][int(np.searchsorted(TABLA["cortes"], v, side="right"))]
        assert ir.forma(TABLA, v) == esperado


def test_tabla_con_valor_en_corte():
    assert ir.forma(RANGO, 0.0) == 0.45 and ir.forma(RANGO, 0.1) == 0.6 and ir.forma(RANGO, -0.1) == 0.3


def test_interpolar_igual_que_np_interp():
    for v in VALORES[1:]:
        assert ir.forma(INTERP, v) == pytest.approx(float(np.interp(v, INTERP["nodos_x"], INTERP["nodos_y"])), abs=1e-15)
    assert ir.forma(INTERP, None) == 1.0


def test_interpolar_exige_nodos_crecientes():
    with pytest.raises(ValueError, match="crecientes"):
        ir.forma(dict(INTERP, nodos_x=[0.0, 0.0, 1.0, 2.0]), 0.0)


def test_categoria():
    assert [ir.forma(CATEG, v) for v in (1.0, 2.0, 5.0, 3.0, None)] == [0.7, -0.3, 0.0, 0.1, 0.2]


def test_sql_de_los_pasos_igual_que_el_interprete():
    duckdb = pytest.importorskip("duckdb")
    con = duckdb.connect()
    vals = VALORES + list(np.random.default_rng(0).uniform(-3, 6, 50))
    con.execute("CREATE TABLE t (v DOUBLE)")
    con.executemany("INSERT INTO t VALUES (?)", [[None if v is None else float(v)] for v in vals])
    for p in (TABLA, RANGO, INTERP, CATEG):
        got = [r[0] for r in con.execute(f"SELECT {sql._forma(p, 'v')} FROM t").fetchall()]
        ref = [ir.forma(p, None if v is None else float(v)) for v in vals]
        assert np.allclose(got, ref, rtol=0, atol=1e-12), p["tipo"]


def _doc(k, alfa, prior=None, unico=True, pesos=None):
    return {"entrada_nucleo": ["a", "b"], "nucleo": {"tipo": "vecinos", "k": k, "ponderacion": "suavizada", "p": 2,
            "pesos": pesos, "alfa": alfa, "prior": prior,
            "bolsa": {"excluir_mismo_paciente": True, "un_vecino_por_paciente": unico}}}


def _referencia(Zq, Zb, yb, pac_q, pac_b, k, alfa, prior, unico):
    out = []
    for i in range(len(Zq)):
        idx = np.flatnonzero(pac_b != pac_q[i])
        d = np.sqrt(((Zb[idx] - Zq[i]) ** 2).sum(1))
        best = {}
        for j, dj in zip(idx, d):
            key = pac_b[j] if unico else j
            if key not in best or (dj, j) < best[key]:
                best[key] = (dj, j)
        sel = sorted(best.values())[:k]
        y = yb[[j for _, j in sel]]
        out.append((y.sum() + 2 * alfa * prior) / (len(y) + 2 * alfa))
    return np.array(out)


@pytest.mark.parametrize("unico", [True, False])
def test_vecinos_suavizado_un_vecino_por_paciente(unico):
    rng = np.random.default_rng(1)
    n = 400
    pac = rng.integers(0, 120, n)                   # varias filas por paciente
    Z = rng.normal(size=(n, 2))
    y = (rng.random(n) < 0.3).astype(float)
    w = [2.0, 0.5]
    p = ir.vecinos(_doc(15, 3.0, unico=unico, pesos=w), Z[:60], Z, y, pac_q=pac[:60], pac_b=pac)
    s = np.sqrt(w)
    ref = _referencia(Z[:60] * s, Z * s, y, pac[:60], pac, 15, 3.0, y.mean(), unico)
    assert np.abs(p - ref).max() <= 1e-12


def test_vecinos_suavizado_sin_vecinos_da_el_prior():
    Z = np.zeros((3, 2))
    p = ir.vecinos(_doc(5, 2.0, prior=0.2), Z[:1], Z, np.array([1.0, 1.0, 1.0]), pac_q=np.array([7]),
                   pac_b=np.array([7, 7, 7]))
    assert p[0] == 0.2


def test_sql_vecinos_suavizado():
    from fnxsql.contrato import Contrato, variables_del_modelo
    from test_ir import NOMBRES
    vs, err = variables_del_modelo(NOMBRES[:2])
    assert not err
    doc = {"formato": ir.FORMATO, "modelo": {"nombre": "perisim", "clase": "PERISIM"}, "contrato": Contrato().a_dict(),
           "variables": [{"nombre": v.nombre, "fuente": v.fuente, "codigo": v.codigo, "valor": v.valor,
                          "estadistico": v.estadistico} for v in vs],
           "preproceso": [dict(TABLA, variable=NOMBRES[0], salida="f0"), dict(INTERP, variable=NOMBRES[1], salida="f1")],
           "entrada_nucleo": ["f0", "f1"], "nucleo": dict(_doc(150, 2.5, pesos=[1.5, 0.4])["nucleo"])}
    doc["nucleo"]["bolsa"] = ir.bolsa_de(type("c", (), {"bolsa": {"un_vecino_por_paciente": True}})())
    s, info = sql.compilar(doc, [11, 22], "auto")
    assert info["un_vecino_por_paciente"] and info["ponderacion"] == "suavizada"
    assert "PARTITION BY episode_ref, paciente_vecino ORDER BY dist, vecino" in s and "FROM unicos d" in s
    assert "(SELECT avg(y) FROM bolsa) AS pi" in s and "coalesce(a.probabilidad, (SELECT pi FROM prior))" in s
    assert "rk <= 150" in s and "WHEN v000 < " in s
