"""DataNex (AWS Athena): consultas en .txt para lanzarlas a mano y lectura de los CSV que devuelve Athena.

En DataNex no hay API: se pega cada consulta en Athena y se descarga su resultado como CSV. Flujo:

  preparar     escribe en datanex/consultas/:
                 00_variables_10_pacientes.txt        las variables de los ingresos de 10 pacientes al azar
                 NN_inferencia_<modelo>.txt           la probabilidad de cada modelo para esos mismos ingresos
                 entrenamiento_01.txt                 la primera tanda del entrenamiento en SQL (entrenar_una.py)
               y en datanex/documentos/ el documento if-then (float32) de cada modelo, para comprobar.
               Los pacientes se eligen DENTRO de cada consulta por un hash de patient_ref (contrato.Cohorte): las mismas
               consultas dan los mismos pacientes mientras la base no cambie, sin pasar listas de un paso a otro.
  continuar    lee resultados/entrenamiento_NN.csv y escribe resultados/entrenamiento_NN+1.txt; tras la última tanda,
               el modelo (JSON de XGBoost e if-then) y resultados/inferencia_boosting_datanex.txt
  comprobar    compara cada resultados/<consulta>.csv de inferencia con el modelo evaluado en local sobre
               resultados/00_variables_10_pacientes.csv y resume el entrenamiento: resultados/comprobacion.csv
  ensayar      (opcional) ejecuta un .txt en FakeNex y guarda el CSV como lo daría Athena, para ensayar el flujo

datanex/resultados/ está en .gitignore: los CSV son datos reales y el modelo entrenado también.
"""
import csv
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from . import binario as B, cliente, config as C, contrato as K, entrenar as E, entrenar_una as U, ir, sql as S

VARIABLES_CSV = "00_variables_10_pacientes"
NOMBRE_DATANEX = "boosting_datanex"


def _escribir(ruta, texto):
    ruta = Path(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    ruta.write_text(texto, encoding="utf-8")
    return ruta


def _cabecera(titulo, csv_nombre, extra=""):
    return (f"-- {titulo}\n-- Generado por fnxsql (datanex). Ejecutar en Athena (DataNex) y descargar el resultado como CSV en\n"
            f"-- datanex/resultados/{csv_nombre}.csv{extra}\n")


def leer_csv(ruta):
    """Filas de un CSV de Athena como lista de dicts (cabeceras en minúsculas; vacío = None)."""
    with open(ruta, newline="", encoding="utf-8-sig") as f:
        csv.field_size_limit(1 << 30)
        return [{k.strip().lower(): (v if v != "" else None) for k, v in fila.items()} for fila in csv.DictReader(f)]


def _doc_modelo(ruta, contrato):
    obj, origen = ir.cargar(ruta)
    variables, errores = K.variables_del_modelo(ir.nombres(obj, origen), contrato)
    if errores:
        raise ValueError(f"{ruta}: {'; '.join(errores)}")
    return ir.nombrar(ir.a_ir(obj, origen, variables, contrato, Path(ruta).stem))


def _tolerancia(doc):
    """Como `evaluar`: el SQL calcula exactamente el documento cuantizado (1e-6 en los modelos float32 de por sí, que
    suman en real; 1e-9 en los demás)."""
    return C.TOLERANCIA_FLOAT32 if doc["nucleo"].get("precision") == "float32" else C.TOLERANCIA_DOUBLE


# ------------------------------------------------------------------ preparar
def preparar(modelos, salida="datanex", pacientes=10, semilla_pacientes=1, cohorte=1000, semilla_cohorte=0, p=None,
             arboles_por_consulta=20, variables_entrenamiento=None, precision="float32"):
    """modelos: lista de (nombre, ruta del modelo). Escribe las consultas .txt, los documentos y plan.json."""
    from .practicar import VARIABLES
    carpeta = Path(salida)
    p = p or E.Parametros()
    contrato = K.Contrato()
    muestra = K.Cohorte(pacientes, semilla_pacientes)
    plan = {"esquema": C.ESQUEMA, "pacientes": {"n": pacientes, "semilla": semilla_pacientes}, "modelos": [], "variables": []}
    docs = []
    for nombre, ruta in modelos:
        docs.append((nombre, _doc_modelo(ruta, contrato)))
    for _, doc in docs:  # variables de todos los modelos (para una sola consulta de variables)
        for v in doc["variables"]:
            if v["nombre"] not in plan["variables"]:
                plan["variables"].append(v["nombre"])
    vs, _ = K.variables_del_modelo(plan["variables"], contrato)
    sql = (_cabecera(f"Variables ({len(vs)}) de los ingresos de {pacientes} pacientes al azar (para comprobar la inferencia)",
                     VARIABLES_CSV, "\n-- Columnas v000, v001...: " + ", ".join(f"v{i:03d} = {n}" for i, n in enumerate(plan["variables"])))
           + K.sql_variables(vs, contrato, None, muestra) + "\nSELECT * FROM vars ORDER BY patient_ref, t0\n")
    _escribir(carpeta / "consultas" / f"{VARIABLES_CSV}.txt", sql)
    for k, (nombre, doc) in enumerate(docs, start=1):
        base = f"{k:02d}_inferencia_{nombre}"
        texto, info = S.compilar(doc, None, "auto", precision=precision, cohorte=muestra)
        _escribir(carpeta / "consultas" / f"{base}.txt", _cabecera(f"Inferencia: {nombre} ({doc['modelo'].get('clase', '')}) "
                                                                  f"para los ingresos de {pacientes} pacientes al azar", base) + texto)
        dq = B.cuantizar(doc, precision)
        _escribir(carpeta / "documentos" / f"{nombre}.json", json.dumps(dq, ensure_ascii=False))
        plan["modelos"].append({"nombre": nombre, "consulta": base, "documento": f"documentos/{nombre}.json",
                                "tipo_sql": info["tipo_sql"], "precision": precision, "bytes": info["bytes"]})
    nombres = list(variables_entrenamiento or VARIABLES)
    estado = U.nuevo_estado(nombres, contrato, p, arboles_por_consulta, K.Cohorte(cohorte, semilla_cohorte))
    sql, n_t = U.siguiente_consulta(estado)
    tandas = math.ceil(p.arboles / estado["arboles_por_consulta"])
    _escribir(carpeta / "consultas" / "entrenamiento_01.txt",
              _cabecera(f"Entrenamiento en SQL, tanda 1 de {tandas}: árboles 1-{n_t} de {p.arboles} (profundidad {p.profundidad}, "
                        f"eta {p.eta}, {p.bins} intervalos) con {cohorte} pacientes al azar", "entrenamiento_01",
                        "\n-- Después: python -m fnxsql datanex continuar   (escribe la tanda siguiente)") + sql)
    plan["entrenamiento"] = estado
    plan["entrenamiento_tandas"] = tandas
    _escribir(carpeta / "plan.json", json.dumps(plan, indent=1, ensure_ascii=False))
    _escribir(carpeta / "LEEME.md", leeme(plan, docs, tandas, len(sql.encode())))
    return plan


def leeme(plan, docs, tandas, bytes_e1):
    ent = plan["entrenamiento"]
    par = ent["parametros"]
    filas = "\n".join(f"| `consultas/{m['consulta']}.txt` | inferencia de **{m['nombre']}** ({m['tipo_sql']}, {m['bytes'] / 1024:.0f} KB) |"
                      for m in plan["modelos"])
    return f"""# Prueba en DataNex (AWS Athena)

Consultas generadas por `python -m fnxsql datanex preparar` para lanzarlas a mano en Athena. Cada una devuelve una tabla
que se descarga como **CSV** y se guarda en `datanex/resultados/` con **el mismo nombre que la consulta** (`.txt` → `.csv`).
`datanex/resultados/` está en `.gitignore`: ahí quedan datos reales y el modelo entrenado con ellos; no se sube.

Esquema: `{plan['esquema']}`. Los pacientes se eligen dentro de cada consulta con un hash de `patient_ref`, entre los que
tienen algún ingreso HOSP / HOSP_IQ: todas las consultas de inferencia usan los mismos {plan['pacientes']['n']} pacientes
(semilla {plan['pacientes']['semilla']}) y todas las de entrenamiento los mismos {ent['cohorte']['pacientes']}
(semilla {ent['cohorte']['semilla']}). Si la base cambia entre una consulta y otra, `continuar` lo detecta.

## 1. Inferencia de los modelos (10 pacientes al azar)

| Consulta | Qué hace |
|---|---|
| `consultas/{VARIABLES_CSV}.txt` | las {len(plan['variables'])} variables de esos ingresos (para comprobar las probabilidades en local) |
{filas}

Cada CSV de inferencia tiene una fila por ingreso: `patient_ref, episode_ref, ingreso, momento, probabilidad`.
Los modelos son los de práctica (entrenados con los datos sintéticos de FakeNex): la prueba es de funcionamiento, no de
rendimiento clínico.

## 2. Entrenamiento en SQL ({tandas} consultas)

Gradient boosting (algoritmo de XGBoost con histogramas) entrenado **dentro de la consulta**: {par['arboles']} árboles de
profundidad {par['profundidad']}, eta {par['eta']}, {par['bins']} intervalos, {ent['arboles_por_consulta']} árboles por consulta,
con {ent['cohorte']['pacientes']} pacientes al azar (20 % de prueba por paciente). Variables: las {len(ent['nombres'])} de
`fnxsql practicar`; etiqueta: muerte en el ingreso (`adm_disch`, `mot_ref` 23). Cada CSV tiene **una sola fila** con los
árboles y las pérdidas en JSON; no contiene datos de pacientes.

1. Ejecutar `consultas/entrenamiento_01.txt` ({bytes_e1 / 1024:.0f} KB) y guardar `resultados/entrenamiento_01.csv`.
2. `python -m fnxsql datanex continuar` → escribe `resultados/entrenamiento_02.txt` (la tanda siguiente, con los cortes y
   los árboles ya hechos dentro de la consulta). Ejecutarla y guardar `resultados/entrenamiento_02.csv`.
3. Repetir `continuar` hasta la tanda {tandas}. Tras la última, `continuar` escribe el modelo
   (`resultados/{NOMBRE_DATANEX}.json`, formato XGBoost) y `resultados/inferencia_{NOMBRE_DATANEX}.txt`: ejecutarla y guardar
   su CSV (inferencia del modelo recién entrenado sobre los mismos 10 pacientes).

## 3. Comprobar

`python -m fnxsql datanex comprobar` → `resultados/comprobacion.csv` (y en pantalla):

- cada inferencia: mismos ingresos que la consulta de variables y probabilidad del SQL = modelo evaluado en local
  (tolerancia 1e-6 en los modelos float32 de por sí, como XGBoost; 1e-9 en los demás);
- entrenamiento: misma cohorte en todas las tandas, cada tanda empieza con la pérdida en que acabó la anterior, pérdida
  final de entrenamiento y de prueba.

Si algo falla en Athena (sintaxis, tiempo, memoria), el mensaje de error es lo que hay que traer.
"""


# ------------------------------------------------------------------ continuar el entrenamiento
def _estado(carpeta):
    r = Path(carpeta) / "resultados" / "estado_entrenamiento.json"
    if r.exists():
        return json.loads(r.read_text(encoding="utf-8"))
    return json.loads((Path(carpeta) / "plan.json").read_text(encoding="utf-8"))["entrenamiento"]


def continuar(carpeta="datanex", ruta_csv=None):
    """Incorpora el CSV de la última tanda y escribe la consulta siguiente (o el modelo final y su inferencia)."""
    carpeta = Path(carpeta)
    res_dir = carpeta / "resultados"
    plan = json.loads((carpeta / "plan.json").read_text(encoding="utf-8"))
    estado = _estado(carpeta)
    k = len(estado["consultas"]) + 1
    ruta_csv = Path(ruta_csv) if ruta_csv else res_dir / f"entrenamiento_{k:02d}.csv"
    if not ruta_csv.exists():
        raise FileNotFoundError(f"falta {ruta_csv}: es el resultado de entrenamiento_{k:02d}.txt")
    filas = leer_csv(ruta_csv)
    if len(filas) != 1:
        raise ValueError(f"{ruta_csv}: se esperaba una fila y hay {len(filas)}")
    avisos = U.incorporar(estado, filas[0], {"tanda": k, "csv": str(ruta_csv)})
    _escribir(res_dir / "estado_entrenamiento.json", json.dumps(estado, ensure_ascii=False))
    e, q_ = estado["perdidas_por_consulta"][-1][-1]
    print(f"tanda {k}: {len(estado['arboles'])} de {estado['parametros']['arboles']} árboles; pérdida entrenamiento {e:.5f}, "
          f"prueba {q_:.5f} ({estado['ingresos']['entrenamiento']} ingresos de entrenamiento, {estado['ingresos']['prueba']} de prueba)")
    for a in avisos:
        print(f"AVISO: {a}")
    if not U.terminado(estado):
        sql, n_t = U.siguiente_consulta(estado)
        hechos = len(estado["arboles"])
        nombre = f"entrenamiento_{k + 1:02d}"
        _escribir(res_dir / f"{nombre}.txt",
                  _cabecera(f"Entrenamiento en SQL, tanda {k + 1} de {plan.get('entrenamiento_tandas', '?')}: árboles "
                            f"{hechos + 1}-{hechos + n_t} (lleva dentro los cortes y los {hechos} árboles ya entrenados)", nombre,
                            "\n-- Después: python -m fnxsql datanex continuar") + sql)
        print(f"-> ejecutar {res_dir / (nombre + '.txt')} ({len(sql.encode()) / 1024:.0f} KB) y guardar {res_dir / (nombre + '.csv')}")
        return estado
    res = U.resultado(estado)
    doc = res["documento"]
    doc["modelo"]["nombre"] = NOMBRE_DATANEX
    _escribir(res_dir / f"{NOMBRE_DATANEX}.json", json.dumps(E.a_xgboost(doc, res["tasa"])))
    _escribir(res_dir / f"{NOMBRE_DATANEX}.if_then.json", json.dumps(doc, ensure_ascii=False))
    pd.DataFrame(res["historial"]).to_csv(res_dir / "entrenamiento_perdidas.csv", index=False)
    muestra = K.Cohorte(plan["pacientes"]["n"], plan["pacientes"]["semilla"])
    texto, info = S.compilar(doc, None, "auto", precision="float32", cohorte=muestra)
    nombre = f"inferencia_{NOMBRE_DATANEX}"
    _escribir(res_dir / f"{nombre}.txt", _cabecera(f"Inferencia del modelo entrenado en DataNex ({len(doc['nucleo']['arboles'])} "
                                                   f"árboles) para los mismos {muestra.pacientes} pacientes", nombre) + texto)
    print(f"entrenamiento terminado: {len(doc['nucleo']['arboles'])} árboles -> {res_dir / (NOMBRE_DATANEX + '.json')} (XGBoost)")
    print(f"-> ejecutar {res_dir / (nombre + '.txt')} ({info['bytes'] / 1024:.0f} KB) y guardar {res_dir / (nombre + '.csv')}; "
          "luego: python -m fnxsql datanex comprobar")
    return estado


# ------------------------------------------------------------------ comprobar
def comprobar(carpeta="datanex"):
    carpeta = Path(carpeta)
    res_dir = carpeta / "resultados"
    plan = json.loads((carpeta / "plan.json").read_text(encoding="utf-8"))
    filas = []
    ruta_v = res_dir / f"{VARIABLES_CSV}.csv"
    if ruta_v.exists():
        dv = pd.DataFrame(leer_csv(ruta_v))
        dv = dv.rename(columns={f"v{i:03d}": n.lower() for i, n in enumerate(plan["variables"])})
        filas.append({"consulta": VARIABLES_CSV, "modelo": "", "resultado": "PASA" if len(dv) else "FALLA",
                      "ingresos": len(dv), "pacientes": dv.patient_ref.nunique() if len(dv) else 0,
                      "detalle": f"{len(plan['variables'])} variables"})
    else:
        dv = None
        filas.append({"consulta": VARIABLES_CSV, "resultado": "FALTA", "detalle": f"no está {ruta_v}"})
    modelos = [(m["nombre"], m["consulta"], carpeta / m["documento"]) for m in plan["modelos"]]
    if (res_dir / f"{NOMBRE_DATANEX}.if_then.json").exists():
        modelos.append((NOMBRE_DATANEX, f"inferencia_{NOMBRE_DATANEX}", None))
    for nombre, consulta, ruta_doc in modelos:
        ruta = res_dir / f"{consulta}.csv"
        if not ruta.exists():
            filas.append({"consulta": consulta, "modelo": nombre, "resultado": "FALTA", "detalle": f"no está {ruta}"})
            continue
        if ruta_doc is None:  # el entrenado en DataNex: se compila en float32 (binario.cuantizar no cambia un núcleo float32)
            doc = B.cuantizar(json.loads((res_dir / f"{NOMBRE_DATANEX}.if_then.json").read_text(encoding="utf-8")), "float32")
        else:
            doc = json.loads(Path(ruta_doc).read_text(encoding="utf-8"))
        ds = pd.DataFrame(leer_csv(ruta))
        fila = {"consulta": consulta, "modelo": nombre, "ingresos": len(ds),
                "pacientes": ds.patient_ref.nunique() if len(ds) else 0}
        if dv is None or not len(ds):
            fila.update(resultado="FALLA" if not len(ds) else "SIN COMPROBAR",
                        detalle="sin filas" if not len(ds) else "falta el CSV de variables")
            filas.append(fila)
            continue
        m = ds.merge(dv, on="episode_ref", how="outer", indicator=True, suffixes=("", "_v"))
        solo = (m["_merge"] != "both").sum()
        m = m[m["_merge"] == "both"]
        p_sql = pd.to_numeric(m.probabilidad).to_numpy(dtype=float)
        p_loc = np.array([ir.evaluar(doc, {v["nombre"]: (None if pd.isna(r[v["nombre"].lower()]) else float(r[v["nombre"].lower()]))
                                           for v in doc["variables"]}) for _, r in m.iterrows()])
        dif = float(np.abs(p_sql - p_loc).max()) if len(m) else math.nan
        tol = _tolerancia(doc)
        ok = len(m) > 0 and solo == 0 and dif <= tol
        fila.update(resultado="PASA" if ok else "FALLA", dif_max=dif, tolerancia=tol,
                    detalle=f"probabilidad SQL = modelo en local (dif. máx. {dif:.1e})" + (f"; {solo} ingresos no coinciden con "
                                                                                          "la consulta de variables" if solo else ""))
        filas.append(fila)
    estado_r = res_dir / "estado_entrenamiento.json"
    if estado_r.exists():
        est = json.loads(estado_r.read_text(encoding="utf-8"))
        pp = est["perdidas_por_consulta"]
        saltos = [max(abs(b[0][0] - a[-1][0]), abs(b[0][1] - a[-1][1])) for a, b in zip(pp, pp[1:])]
        cadena = max(saltos, default=0.0)
        fin = U.terminado(est)
        ok = fin and cadena <= 1e-9 and pp[-1][-1][0] < pp[0][0][0]
        filas.append({"consulta": f"entrenamiento_01-{len(pp):02d}", "modelo": NOMBRE_DATANEX,
                      "resultado": "PASA" if ok else ("EN CURSO" if not fin else "FALLA"),
                      "ingresos": est["ingresos"]["entrenamiento"] + est["ingresos"]["prueba"],
                      "detalle": (f"{len(est['arboles'])} árboles en {len(pp)} consultas; {est['ingresos']['entrenamiento']} ingresos de "
                                  f"entrenamiento ({est['ingresos']['positivos']:.0f} positivos) y {est['ingresos']['prueba']} de prueba; "
                                  f"pérdida entrenamiento {pp[0][0][0]:.4f} -> {pp[-1][-1][0]:.4f}, prueba {pp[0][0][1]:.4f} -> "
                                  f"{pp[-1][-1][1]:.4f}; continuidad entre tandas {cadena:.1e}")})
    out = pd.DataFrame(filas)
    out.to_csv(res_dir / "comprobacion.csv", index=False)
    with pd.option_context("display.max_colwidth", 140, "display.width", 250):
        print(out[[c for c in ("consulta", "resultado", "ingresos", "detalle") if c in out]].to_string(index=False))
    print(f"-> {res_dir / 'comprobacion.csv'}")
    return out


# ------------------------------------------------------------------ ensayo en FakeNex (mismo dialecto que Athena)
def ensayar(txt, ruta_csv):
    """Ejecuta un .txt en FakeNex y guarda su resultado como CSV al estilo de Athena (todo entre comillas, nulo = vacío)."""
    sql = Path(txt).read_text(encoding="utf-8")
    df, reloj, st = cliente.ejecutar(sql, max_segundos=U.MAX_SEGUNDOS)
    Path(ruta_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(ruta_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, quoting=csv.QUOTE_ALL)
        w.writerow(list(df.columns))
        for fila in df.itertuples(index=False):
            w.writerow(["" if v is None or (isinstance(v, float) and math.isnan(v)) else
                        (json.dumps(v) if isinstance(v, (list, dict)) else v) for v in fila])
    return len(df), cliente.resumen_stats(st, reloj)
