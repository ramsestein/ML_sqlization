"""Contrato de variables: de cada NOMBRE de variable del modelo sale qué dato de DataNex / FakeNex es y cómo se calcula.

Condición del repositorio: los nombres de las variables del modelo son códigos exactos de DataNex / FakeNex. Gramática
(<est> = last | first | mean | std | min | max | count, sobre las medidas de la ventana):

  <RC>_<est>                 rc.result_num del código rc_sap_ref <RC>            p. ej. FC_mean, TAS_last, PULSIOX_min
  <LAB>_<est>                labs.result_num del código lab_sap_ref <LAB>       p. ej. LAB2467_max (creatinina)
  <RC>_<RC>_<k>_<est>        ítem codificado: indicador 0/1 de rc.result_txt = '<RC>_<k>' sobre las medidas del ítem
                             p. ej. ESTADO_CONCIENC_ESTADO_CONCIENC_1_last (en DataNex el texto empieza por el código)
  age, age_<est>             edad en años en el momento de evaluación (demographics.birth_date); age_std = 0
  sex, sex_<k>               demographics.sex (código) o su indicador 0/1
  (los <LAB> empiezan por "LAB"; el resto se interpreta como rc_sap_ref)

Lo que no encaja en la gramática solo se acepta si el contrato lo declara constante ("constantes": {"index_last": 0}).

Archivo de contrato (JSON, opcional; todo tiene valor por defecto):
  {"momento_horas": 24,        momento de evaluación = inicio del ingreso + horas (o el alta si es antes)
   "ventana_horas": 48,        las medidas cuentan en [momento - ventana, momento]
   "si_falta": null,           valor de una variable sin datos: null = ausente (NaN para el modelo) o un número (p. ej. 0)
   "constantes": {},           variables sin equivalente en DataNex, con su valor fijo
   "renombres": {}}            solo para `normalizar`: base original -> código de DataNex (p. ej. "P_ART_S": "TAS")

Unidad: cada ingreso HOSP / HOSP_IQ del paciente. Las dos implementaciones (sql_variables en el servidor y
variables_locales con pandas sobre las tablas crudas descargadas) se comparan en la validación.
"""
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from . import cliente
from . import config as C

ESTADISTICOS = ("last", "first", "mean", "std", "min", "max", "count")


@dataclass(frozen=True)
class Variable:
    nombre: str
    fuente: str               # rc | labs | rc_codigo | edad | sexo | sexo_ind | constante
    codigo: str = None        # rc_sap_ref / lab_sap_ref
    valor: str = None         # result_txt (rc_codigo) o código de sexo (sexo_ind)
    estadistico: str = None
    constante: float = None


@dataclass
class Contrato:
    momento_horas: float = C.MOMENTO_HORAS
    ventana_horas: float = C.VENTANA_HORAS
    si_falta: float = None
    constantes: dict = field(default_factory=dict)
    renombres: dict = field(default_factory=dict)

    @classmethod
    def leer(cls, ruta=None):
        if not ruta:
            return cls()
        d = json.loads(Path(ruta).read_text(encoding="utf-8"))
        return cls(**{k: d[k] for k in ("momento_horas", "ventana_horas", "si_falta", "constantes", "renombres") if k in d})

    def a_dict(self):
        return {"momento_horas": self.momento_horas, "ventana_horas": self.ventana_horas, "si_falta": self.si_falta,
                "constantes": self.constantes}


# ------------------------------------------------------------------ gramática
def interpretar(nombre, contrato=None):
    """Variable de un nombre, o ValueError si no sigue la gramática (ni está declarado constante)."""
    contrato = contrato or Contrato()
    if nombre in contrato.constantes:
        return Variable(nombre, "constante", constante=float(contrato.constantes[nombre]))
    if nombre == "age":
        return Variable(nombre, "edad", estadistico="last")
    if nombre == "sex":
        return Variable(nombre, "sexo")
    m = re.fullmatch(r"sex_(\d+)", nombre)
    if m:
        return Variable(nombre, "sexo_ind", valor=m.group(1))
    if "_" not in nombre:
        raise ValueError(f"{nombre!r}: falta el estadístico (_last, _mean...)")
    base, est = nombre.rsplit("_", 1)
    if est not in ESTADISTICOS:
        raise ValueError(f"{nombre!r}: estadístico {est!r} no reconocido ({', '.join(ESTADISTICOS)})")
    if base == "age":
        return Variable(nombre, "edad", estadistico=est)
    if base.startswith("LAB"):
        return Variable(nombre, "labs", codigo=base, estadistico=est)
    partes = base.split("_")
    for i in range(1, len(partes)):  # ítem codificado: <RC>_<RC>_<k>
        cod, resto = "_".join(partes[:i]), "_".join(partes[i:])
        if resto.startswith(cod + "_"):
            return Variable(nombre, "rc_codigo", codigo=cod, valor=resto, estadistico=est)
    return Variable(nombre, "rc", codigo=base, estadistico=est)


def variables_del_modelo(nombres, contrato=None):
    """(variables interpretadas, errores) para la lista de nombres del modelo."""
    vs, errores = [], []
    for n in nombres:
        try:
            vs.append(interpretar(n, contrato))
        except ValueError as e:
            errores.append(str(e))
    return vs, errores


def comprobar_en_fakenex(variables):
    """Avisos sobre los códigos: los de laboratorio deben estar en dic_lab_loinc (diccionario real de DataNex que sirve
    FakeNex); de rc no hay diccionario en el servidor, así que se avisa de los que FakeNex no tiene (valdrán ausentes)."""
    errores, avisos = [], []
    labs = sorted({v.codigo for v in variables if v.fuente == "labs"})
    rcs = sorted({v.codigo for v in variables if v.fuente in ("rc", "rc_codigo")})
    if labs:
        df, _, _ = cliente.ejecutar(f"SELECT DISTINCT lab_sap_ref FROM {C.ESQUEMA}.dic_lab_loinc WHERE lab_sap_ref IN "
                                    f"({', '.join(map(q, labs))})")
        faltan = sorted(set(labs) - set(df.iloc[:, 0]))
        if faltan:
            errores.append(f"códigos de laboratorio que no están en dic_lab_loinc (no son de DataNex): {faltan}")
    if rcs:
        df, _, _ = cliente.ejecutar(f"SELECT DISTINCT rc_sap_ref FROM {C.ESQUEMA}.rc WHERE patient_ref {C.BASE} AND "
                                    f"rc_sap_ref IN ({', '.join(map(q, rcs))})")
        sin = sorted(set(rcs) - set(df.iloc[:, 0]))
        if sin:
            avisos.append(f"códigos de rc sin datos en FakeNex (valdrán ausentes / si_falta): {sin}")
    return errores, avisos


# ------------------------------------------------------------------ SQL (servidor)
def q(s):
    return "'" + str(s).replace("'", "''") + "'"


def columna(i):
    return f"v{i:03d}"


def _agregado(val, t, est):
    return {"last": f"max_by({val}, {t})", "first": f"min_by({val}, {t})", "mean": f"avg({val})",
            "std": f"stddev_samp({val})", "min": f"min({val})", "max": f"max({val})", "count": f"count({val})"}[est]


def sql_variables(variables, contrato, pids=None):
    """CTE `vars`: una fila por ingreso con las variables del modelo en columnas v000, v001... (en su orden).

    pids: lista de patient_ref, o None para toda la base original. Cada tabla se filtra por paciente, así que solo se
    leen sus filas; un filtro por patient_ref no dispara el crecimiento automático de FakeNex."""
    filtro = f"IN ({', '.join(map(str, pids))})" if pids else C.BASE
    S, mo, ve = C.ESQUEMA, contrato.momento_horas, contrato.ventana_horas
    win = lambda t: f"{t} BETWEEN a.t_eval - INTERVAL '{ve}' HOUR AND a.t_eval"
    rc_num, rc_cat, lab, fuera = [], [], [], []
    for i, v in enumerate(variables):
        c = columna(i)
        if v.fuente == "rc":
            rc_num.append(f"{_agregado('r.result_num', 'r.result_date', v.estadistico)} FILTER (WHERE r.rc_sap_ref = {q(v.codigo)}) AS {c}")
        elif v.fuente == "labs":
            lab.append(f"{_agregado('l.result_num', 'l.extrac_date', v.estadistico)} FILTER (WHERE l.lab_sap_ref = {q(v.codigo)}) AS {c}")
        elif v.fuente == "rc_codigo":
            ind = f"CASE WHEN r.result_txt = {q(v.valor)} THEN 1E0 ELSE 0E0 END"
            rc_cat.append(f"{_agregado(ind, 'r.result_date', v.estadistico)} FILTER (WHERE r.rc_sap_ref = {q(v.codigo)}) AS {c}")
    ctes = [f"""adm AS (  -- ingresos; momento de evaluación = inicio + {mo} h (o el alta si es antes)
  SELECT patient_ref, episode_ref, start_date AS t0, least(start_date + INTERVAL '{mo}' HOUR, end_date) AS t_eval
  FROM {S}.episodes WHERE patient_ref {filtro} AND episode_type_ref IN ({', '.join(map(q, C.TIPOS_INGRESO))})
)"""]
    uniones = []
    for nombre, cols, tabla, alias, extra in (
            ("rc_num", rc_num, "rc", "r", f"r.result_num IS NOT NULL AND r.rc_sap_ref IN ({{codes}}) AND {win('r.result_date')}"),
            ("rc_cat", rc_cat, "rc", "r", f"r.rc_sap_ref IN ({{codes}}) AND {win('r.result_date')}"),
            ("lab", lab, "labs", "l", f"l.result_num IS NOT NULL AND l.lab_sap_ref IN ({{codes}}) AND {win('l.extrac_date')}")):
        if not cols:
            continue
        fuentes = {"rc_num": "rc", "rc_cat": "rc_codigo", "lab": "labs"}[nombre]
        codes = ", ".join(map(q, sorted({v.codigo for v in variables if v.fuente == fuentes})))
        ctes.append(f"{nombre} AS (\n  SELECT a.episode_ref AS ep_{nombre},\n         " + ",\n         ".join(cols)
                    + f"\n  FROM adm a JOIN {S}.{tabla} {alias} ON {alias}.patient_ref = a.patient_ref AND {alias}.patient_ref "
                    f"{filtro}\n   AND {extra.format(codes=codes)}\n  GROUP BY a.episode_ref\n)")
        uniones.append(f"LEFT JOIN {nombre} ON {nombre}.ep_{nombre} = a.episode_ref")
    edad = "CAST(date_diff('day', CAST(d.birth_date AS timestamp(3)), a.t_eval) AS double) / 365.25E0"
    sel = []
    for i, v in enumerate(variables):
        c = columna(i)
        if v.fuente in ("rc", "labs", "rc_codigo"):
            e = f"coalesce({c}, 0)" if v.estadistico == "count" else c
        elif v.fuente == "edad":
            e = "0E0" if v.estadistico == "std" else ("1E0" if v.estadistico == "count" else edad)
        elif v.fuente == "sexo":
            e = "d.sex"
        elif v.fuente == "sexo_ind":
            e = f"CASE WHEN d.sex IS NULL THEN NULL WHEN d.sex = {int(v.valor)} THEN 1E0 ELSE 0E0 END"
        else:
            e = lit(v.constante)
        e = f"CAST({e} AS double)"
        if contrato.si_falta is not None and v.fuente != "constante":
            e = f"coalesce({e}, {lit(contrato.si_falta)})"
        sel.append(f"-- {v.nombre}\n         {e} AS {c}")
    ctes.append("vars AS (\n  SELECT a.patient_ref, a.episode_ref, a.t0, a.t_eval,\n         " + ",\n         ".join(sel)
                + f"\n  FROM adm a\n  LEFT JOIN {S}.demographics d ON d.patient_ref = a.patient_ref AND d.patient_ref {filtro}"
                + "".join(f"\n  {u}" for u in uniones) + "\n)")
    return "WITH\n" + ",\n".join(ctes)


def lit(x):
    """Literal DOUBLE de Trino que representa exactamente el double x (17 cifras significativas)."""
    return f"{float(x):.16E}"


def sql_etiqueta(pids=None):
    """CTE `etiqueta`: muerte durante el ingreso (alta por exitus, adm_disch mot_ref 23). Solo para practicar."""
    filtro = f"IN ({', '.join(map(str, pids))})" if pids else C.BASE
    return (f"etiqueta AS (SELECT episode_ref AS ep_et, max(CASE WHEN mot_ref = 23 THEN 1 ELSE 0 END) AS muere\n"
            f"  FROM {C.ESQUEMA}.adm_disch WHERE patient_ref {filtro} AND mot_type = 'END' GROUP BY episode_ref)")


def descargar_variables(variables, contrato, pids=None, con_etiqueta=False):
    """Variables calculadas en el servidor: DataFrame con patient_ref, episode_ref, t0, t_eval y una columna por variable."""
    sql = sql_variables(variables, contrato, pids)
    if con_etiqueta:
        sql += ",\n" + sql_etiqueta(pids) + "\nSELECT vars.*, coalesce(etiqueta.muere, 0) AS muere FROM vars LEFT JOIN etiqueta ON etiqueta.ep_et = vars.episode_ref"
    else:
        sql += "\nSELECT * FROM vars"
    df, reloj, st = cliente.ejecutar(sql)
    df.columns = [c.lower() for c in df.columns]
    ren = {columna(i): v.nombre for i, v in enumerate(variables)}
    df = df.rename(columns=ren)
    for v in variables:
        df[v.nombre] = pd.to_numeric(df[v.nombre], errors="coerce").astype(float)
    return df, reloj, st


# ------------------------------------------------------------------ cálculo local independiente (pandas)
def descargar_tablas(variables, pids, carpeta):
    """Tablas crudas de los pacientes (solo las filas y códigos que usan las variables) en CSV."""
    carpeta = Path(carpeta)
    carpeta.mkdir(parents=True, exist_ok=True)
    lista = ", ".join(map(str, pids))
    rc_cod = sorted({v.codigo for v in variables if v.fuente in ("rc", "rc_codigo")})
    lab_cod = sorted({v.codigo for v in variables if v.fuente == "labs"})
    ts = lambda c: f"date_format({c}, '%Y-%m-%d %H:%i:%s') AS {c}"
    consultas = {"episodes": f"SELECT patient_ref, episode_ref, episode_type_ref, {ts('start_date')}, {ts('end_date')} "
                             f"FROM {C.ESQUEMA}.episodes WHERE patient_ref IN ({lista})",
                 "demographics": f"SELECT patient_ref, CAST(birth_date AS varchar) AS birth_date, sex FROM {C.ESQUEMA}.demographics "
                                 f"WHERE patient_ref IN ({lista})"}
    if rc_cod:
        consultas["rc"] = (f"SELECT patient_ref, {ts('result_date')}, rc_sap_ref, result_num, result_txt FROM {C.ESQUEMA}.rc "
                           f"WHERE patient_ref IN ({lista}) AND rc_sap_ref IN ({', '.join(map(q, rc_cod))})")
    if lab_cod:
        consultas["labs"] = (f"SELECT patient_ref, {ts('extrac_date')}, lab_sap_ref, result_num FROM {C.ESQUEMA}.labs "
                             f"WHERE patient_ref IN ({lista}) AND lab_sap_ref IN ({', '.join(map(q, lab_cod))})")
    filas = {}
    for t, sql in consultas.items():
        df, _, _ = cliente.ejecutar(sql)
        df.to_csv(carpeta / f"{t}.csv", index=False)
        filas[t] = len(df)
    return filas


def variables_locales(variables, contrato, carpeta):
    """Las mismas variables que sql_variables, calculadas con pandas desde las tablas crudas (otra implementación)."""
    carpeta = Path(carpeta)
    leer = lambda t, fechas: (pd.read_csv(carpeta / f"{t}.csv", parse_dates=fechas, keep_default_na=False, na_values=[""])
                              if (carpeta / f"{t}.csv").exists() else None)
    ep = leer("episodes", ["start_date", "end_date"])
    adm = ep[ep.episode_type_ref.isin(C.TIPOS_INGRESO)].copy()
    adm["t_eval"] = np.minimum(adm.start_date + pd.Timedelta(hours=contrato.momento_horas), adm.end_date)
    dm = leer("demographics", ["birth_date"])
    rc, lb = leer("rc", ["result_date"]), leer("labs", ["extrac_date"])
    filas = []
    for a in adm.itertuples():
        lo, hi = a.t_eval - pd.Timedelta(hours=contrato.ventana_horas), a.t_eval
        d = dm[dm.patient_ref == a.patient_ref]
        edad = (a.t_eval - d.birth_date.iloc[0]).days / 365.25 if len(d) and pd.notna(d.birth_date.iloc[0]) else np.nan
        sexo = float(d.sex.iloc[0]) if len(d) and pd.notna(d.sex.iloc[0]) else np.nan
        r = rc[(rc.patient_ref == a.patient_ref) & (rc.result_date >= lo) & (rc.result_date <= hi)] if rc is not None else None
        l = lb[(lb.patient_ref == a.patient_ref) & (lb.extrac_date >= lo) & (lb.extrac_date <= hi)] if lb is not None else None
        fila = {"patient_ref": a.patient_ref, "episode_ref": a.episode_ref}
        for v in variables:
            if v.fuente == "constante":
                x = v.constante
            elif v.fuente == "edad":
                x = 0.0 if v.estadistico == "std" else (1.0 if v.estadistico == "count" else edad)
            elif v.fuente == "sexo":
                x = sexo
            elif v.fuente == "sexo_ind":
                x = np.nan if np.isnan(sexo) else float(sexo == int(v.valor))
            else:
                if v.fuente == "labs":
                    s = l[(l.lab_sap_ref == v.codigo) & l.result_num.notna()].sort_values("extrac_date")
                    serie = s.result_num.astype(float)
                elif v.fuente == "rc_codigo":
                    s = r[r.rc_sap_ref == v.codigo].sort_values("result_date")
                    serie = (s.result_txt == v.valor).astype(float)
                else:
                    s = r[(r.rc_sap_ref == v.codigo) & r.result_num.notna()].sort_values("result_date")
                    serie = s.result_num.astype(float)
                if v.estadistico == "count":
                    x = float(len(serie))
                elif serie.empty:
                    x = np.nan
                else:
                    x = {"last": serie.iloc[-1], "first": serie.iloc[0], "mean": serie.mean(), "min": serie.min(),
                         "max": serie.max(), "std": serie.std() if len(serie) > 1 else np.nan}[v.estadistico]
            if contrato.si_falta is not None and v.fuente != "constante" and (x is None or np.isnan(x)):
                x = float(contrato.si_falta)
            fila[v.nombre] = float(x)
        filas.append(fila)
    return pd.DataFrame(filas)
