"""Evaluación de un modelo contra FakeNex con comprobaciones automáticas e informe.

Pasos (cada comprobación queda en el informe como PASA / FALLA / AVISO; el proceso termina con código 1 si alguna falla):
  1. nombres: todas las variables siguen la gramática de DataNex / FakeNex (contrato.py)
  2. códigos: los de laboratorio existen en dic_lab_loinc de FakeNex; aviso por los de rc sin datos en FakeNex
  3. if-then: documento exacto (ir.py) = modelo original en filas sintéticas (con ausentes)
  4. SQL: compilado desde el if-then (sql.py) y ejecutado en FakeNex para los pacientes elegidos
  5. variables: las del servidor = las calculadas en local con pandas desde las tablas crudas descargadas
  6. predicción: SQL = if-then cuantizado a la precisión elegida (binario.cuantizar) y error de esa precisión frente al
     modelo original (float64 / float32: con tolerancia; int8: aviso con el error medido)
  7. tiempos: opcionalmente, lotes de varios tamaños (solo medida)
"""
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import binario as B, cliente, contrato as K, ir, sql as S
from . import config as C


class Informe:
    def __init__(self, nombre):
        self.d = {"modelo": nombre, "fecha": time.strftime("%Y-%m-%d %H:%M:%S"), "servidor": C.URL, "usuario": C.USUARIO,
                  "comprobaciones": [], "avisos": [], "datos": {}}

    def check(self, nombre, ok, detalle=""):
        self.d["comprobaciones"].append({"comprobacion": nombre, "resultado": "PASA" if ok else "FALLA", "detalle": detalle})
        print(f"  [{'PASA' if ok else 'FALLA'}] {nombre}{(': ' + str(detalle)) if detalle else ''}", flush=True)
        return ok

    def aviso(self, texto):
        self.d["avisos"].append(texto)
        print(f"  [AVISO] {texto}", flush=True)

    @property
    def ok(self):
        return all(c["resultado"] == "PASA" for c in self.d["comprobaciones"])

    def guardar(self, carpeta):
        carpeta = Path(carpeta)
        carpeta.mkdir(parents=True, exist_ok=True)
        (carpeta / "informe.json").write_text(json.dumps(self.d, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        md = [f"# Informe: {self.d['modelo']}", "", f"{self.d['fecha']} · servidor {self.d['servidor']} · usuario {self.d['usuario']}",
              "", f"**Resultado: {'PASA' if self.ok else 'FALLA'}**", "", "| Comprobación | Resultado | Detalle |", "|---|---|---|"]
        md += [f"| {c['comprobacion']} | {c['resultado']} | {c['detalle']} |" for c in self.d["comprobaciones"]]
        if self.d["avisos"]:
            md += ["", "## Avisos", ""] + [f"- {a}" for a in self.d["avisos"]]
        for k, v in self.d["datos"].items():
            md += ["", f"## {k}", "", "```json", json.dumps(v, indent=1, ensure_ascii=False, default=str), "```"]
        (carpeta / "informe.md").write_text("\n".join(md) + "\n", encoding="utf-8")


def pacientes_base(n, semilla=C.SEMILLA, excluir=(), solo=None):
    """n pacientes de la base original de FakeNex con algún ingreso (muestra reproducible)."""
    df, _, _ = cliente.ejecutar(f"SELECT DISTINCT patient_ref FROM {C.ESQUEMA}.episodes WHERE patient_ref {C.BASE} AND "
                                f"episode_type_ref IN ({', '.join(map(K.q, C.TIPOS_INGRESO))}) ORDER BY patient_ref")
    todos = [int(x) for x in df.iloc[:, 0] if int(x) not in set(excluir) and (solo is None or int(x) in set(solo))]
    return random.Random(semilla).sample(todos, min(n, len(todos)))


def sinteticas(variables, n, semilla, con_ausentes):
    """Filas sintéticas para comparar if-then y modelo: magnitudes variadas, ceros, 0/1 para indicadores y ausentes."""
    rng = np.random.default_rng(semilla)
    cols = {}
    for v in variables:
        if v.fuente in ("rc_codigo", "sexo_ind"):
            x = rng.integers(0, 2, n).astype(float)
        elif v.fuente == "constante":
            x = np.full(n, v.constante)
        else:
            x = rng.normal(0, 1, n) * rng.choice([0, 1, 10, 100], n) + rng.choice([0, 50], n)
        if con_ausentes and v.fuente != "constante":
            x[rng.random(n) < 0.15] = np.nan
        cols[v.nombre] = x
    return pd.DataFrame(cols)


def evaluar(ruta_modelo, ruta_contrato=None, pacientes=None, muestra=3, tipo="auto", lotes=(), salida="salidas",
            forzar=False, semilla=C.SEMILLA, solo_pacientes=None, precision=None):
    ruta_modelo = Path(ruta_modelo)
    nombre = ruta_modelo.stem
    carpeta = Path(salida) / nombre
    inf = Informe(nombre)
    print(f"== {nombre} ({ruta_modelo})", flush=True)
    try:
        _evaluar(inf, ruta_modelo, ruta_contrato, pacientes, muestra, tipo, lotes, carpeta, forzar, semilla, solo_pacientes,
                 precision or C.PRECISION)
    except cliente.ErrorFakeNex as e:
        inf.check("ejecución en FakeNex", False, str(e))
    except S.SQLDemasiadoGrande as e:
        inf.check("tamaño del SQL dentro de los límites (AWS Athena: 262.144 bytes)", False, str(e))
    except ValueError as e:
        inf.check("modelo admitido", False, str(e))
    inf.guardar(carpeta)
    print(f"  -> {'PASA' if inf.ok else 'FALLA'}  ({carpeta / 'informe.md'})", flush=True)
    return inf


def _evaluar(inf, ruta_modelo, ruta_contrato, pacientes, muestra, tipo, lotes, carpeta, forzar, semilla, solo_pacientes,
             precision):
    obj, origen = ir.cargar(ruta_modelo)
    nombres = ir.nombres(obj, origen)
    contrato = K.Contrato.leer(ruta_contrato)
    variables, errores = K.variables_del_modelo(nombres, contrato)
    if not inf.check("nombres con la gramática de DataNex / FakeNex", not errores, "; ".join(errores) or f"{len(nombres)} variables"):
        return
    err2, avisos = K.comprobar_en_fakenex(variables)
    for a in avisos:
        inf.aviso(a)
    if not inf.check("códigos de laboratorio en dic_lab_loinc", not err2, "; ".join(err2)):
        return
    doc = ir.nombrar(ir.a_ir(obj, origen, variables, contrato, ruta_modelo.stem))
    carpeta.mkdir(parents=True, exist_ok=True)
    (carpeta / "if_then.json").write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
    inf.d["datos"]["if_then"] = {**ir.resumen(doc), "kb": round((carpeta / "if_then.json").stat().st_size / 1024)}
    tol = C.TOLERANCIA_FLOAT32 if doc["nucleo"].get("precision") == "float32" else C.TOLERANCIA_DOUBLE  # if-then: exacto
    # El SQL calcula el documento cuantizado (binario.cuantizar): se compara con él (exacto) y, aparte, se mide el error
    # que añade la precisión frente al modelo original (float32: con tolerancia; int8: se informa, es una pérdida aceptada)
    dq = B.cuantizar(doc, precision)
    f32n = doc["nucleo"].get("precision") == "float32"
    tol_prec = {"float64": tol, "float32": tol if f32n else C.TOLERANCIA_PARAMETROS_FLOAT32, "int8": None}[precision]
    try:
        X = sinteticas(variables, 300, semilla, con_ausentes=contrato.si_falta is None)
        d = np.abs(np.array([ir.evaluar(doc, r) for r in X.to_dict("records")]) - ir.predecir_nativo(obj, origen, X)).max()
        inf.check("if-then = modelo original (300 filas sintéticas)", d <= tol, f"diferencia máxima {d:.1e} (tolerancia {tol:.0e})")
        nativo = True
    except ImportError as e:
        inf.aviso(f"no se puede ejecutar el modelo original ({e}); se comparan if-then y SQL")
        nativo = False
    pids = pacientes or pacientes_base(muestra, semilla, solo=solo_pacientes)
    sql, info = S.compilar(doc, pids, tipo, forzar, precision)
    (carpeta / "consulta.sql").write_text(sql, encoding="utf-8")
    srv, reloj, st = cliente.ejecutar(sql)
    srv.columns = [c.lower() for c in srv.columns]
    inf.d["datos"]["sql"] = {**info, "pacientes": pids, "ingresos": len(srv), **cliente.resumen_stats(st, reloj)}
    inf.check("SQL ejecutado en FakeNex", len(srv) > 0, f"{len(srv)} ingresos en {reloj:.1f} s ({info['tipo_sql']}, "
                                                        f"{info['precision']}, {info['bytes']:,} bytes)")
    inf.check("SQL dentro del límite de AWS Athena", info["bytes"] <= C.MAX_SQL_BYTES,
              f"{info['bytes']:,} de {C.MAX_SQL_BYTES:,} bytes ({len(pids)} pacientes)")
    vsrv, _, _ = K.descargar_variables(variables, contrato, pids)
    filas = K.descargar_tablas(variables, pids, carpeta / "tablas")
    loc = K.variables_locales(variables, contrato, carpeta / "tablas")
    a, b = vsrv.set_index("episode_ref").sort_index(), loc.set_index("episode_ref").sort_index()
    malas = [v.nombre for v in variables if not np.allclose(a[v.nombre].loc[b.index].to_numpy(dtype=float),
                                                            b[v.nombre].to_numpy(dtype=float), rtol=1e-12, atol=1e-9, equal_nan=True)]
    inf.check("variables del servidor = variables calculadas en local", not malas and len(a) == len(b),
              f"{len(variables)} variables x {len(b)} ingresos" + (f"; distintas: {malas[:10]}" if malas else ""))
    con_datos = sorted({v.nombre for v in variables if v.fuente != "constante" and b[v.nombre].notna().any()
                        and (b[v.nombre] != (contrato.si_falta if contrato.si_falta is not None else np.nan)).any()})
    inf.d["datos"]["variables"] = {"con_datos_en_estos_ingresos": len(con_datos), "total": len(variables), "filas_descargadas": filas}
    filas_b = b[nombres].to_dict("records")
    p_ir = np.array([ir.evaluar(doc, r) for r in filas_b])
    res = pd.DataFrame({"episode_ref": b.index, "p_ifthen": p_ir, "p_ifthen_" + precision: [ir.evaluar(dq, r) for r in filas_b]})
    if nativo:
        res["p_modelo"] = ir.predecir_nativo(obj, origen, b[nombres])
    res = res.merge(srv[["episode_ref", "probabilidad"]].rename(columns={"probabilidad": "p_sql"}), on="episode_ref")
    res.to_csv(carpeta / "comparacion.csv", index=False)
    d0 = float(np.abs(res.p_sql - res["p_ifthen_" + precision]).max())
    inf.check(f"SQL = if-then en {precision} ({len(res)} ingresos reales)", d0 <= tol,
              f"diferencia máxima {d0:.1e} (tolerancia {tol:.0e})")
    ref = "p_modelo" if nativo else "p_ifthen"
    dif = np.abs(res.p_sql - res[ref])
    texto = (f"SQL {precision} frente al {'modelo original' if nativo else 'if-then exacto'} ({len(res)} ingresos reales): "
             f"diferencia máxima {dif.max():.1e}, media {dif.mean():.1e}")
    inf.d["datos"]["error_precision"] = {"precision": precision, "maxima": float(dif.max()), "media": float(dif.mean())}
    if tol_prec is None:
        inf.aviso(texto + " (int8: pérdida aceptada, sin tolerancia)")
    else:
        inf.check(f"error de la precisión {precision} frente al {'modelo original' if nativo else 'if-then exacto'}",
                  dif.max() <= tol_prec, f"diferencia máxima {dif.max():.1e}, media {dif.mean():.1e} (tolerancia {tol_prec:.0e})")
    if nativo:
        d2 = float(np.abs(res.p_ifthen - res.p_modelo).max())
        inf.check("if-then = modelo original (ingresos reales)", d2 <= tol, f"diferencia máxima {d2:.1e}")
    tiempos = []
    for n in lotes:
        ps = pacientes_base(n, semilla + 1, solo=solo_pacientes)
        sq, _ = S.compilar(doc, ps, tipo, forzar, precision)
        df, rl, stn = cliente.ejecutar(sq)
        tiempos.append({"pacientes": len(ps), "ingresos": len(df), **cliente.resumen_stats(stn, rl)})
        print(f"  lote {len(ps)}: {tiempos[-1]}", flush=True)
    if tiempos:
        inf.d["datos"]["tiempos"] = tiempos
