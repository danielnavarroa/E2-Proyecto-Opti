import pandas as pd
from gurobipy import Model, GRB
from gurobipy import quicksum
import datetime

def cargar_datos():
    # Comisarías por comuna
    df_C = pd.read_excel("ComisariasXComuna.xlsx")
    C  = df_C["id_comisaría"].tolist()
    S  = df_C["comuna"].unique().tolist()
    dc = df_C.set_index("id_comisaría")["min_carabineros"].to_dict()
    Cs = df_C.groupby("comuna")["id_comisaría"].apply(list).to_dict() 
            #si quiero la cantidad en ves de los id es .apply(len)

    # Franjas horarias
    df_F = pd.read_excel("FranjaHoraria.xlsx", header=None)
    F = df_F.iloc[:,0].tolist()

    # Días
    df_T = pd.read_excel("Dias.xlsx", header=None)
    T = df_T.iloc[:,0].tolist()

    # Sueldos de carabineros
    df_P = pd.read_excel("SueldoXCarabinero.xlsx")
    P = df_P["id_carabinero"].tolist()
    beta = df_P.set_index("id_carabinero")["sueldo"].to_dict()

    # Material complementario
    df_G = pd.read_excel("Materiales.xlsx")
    G = df_G["id_material"].tolist()
    psig = df_G.set_index("id_material")["efectividad"].to_dict()
    lambdag = df_G.set_index("id_material")["costo_fijo"].to_dict()

    # Vehículos
    df_B = pd.read_excel("CostoXVehiculo.xlsx")
    B = df_B["id_vehiculo"].tolist()
    gammab = df_B.set_index("id_vehiculo")["costo_fijo"].to_dict()
    kappab = df_B.set_index("id_vehiculo")["efectividad"].to_dict()
    deltab = df_B.set_index("id_vehiculo")["maximo_carabineros"].to_dict()

    # Parámetros sueltos
    df_param = pd.read_excel("ParametrosSueltos.xlsx")
    xi = df_param.loc[0, "efectividad_patrullaje_sin_vehiculo"]
    epsilon = df_param.loc[0, "dist_maxima_desplazamiento"]
    alpha = df_param.loc[0, "presupuesto_inicial"]

    # Distancias entre comunas (matriz)
    # 1) Lee la tabla usando la primera columna como índice
    df_rho = pd.read_excel(
        "DistanciaEntreComunas.xlsx",
        index_col=0,
        engine="openpyxl"
    )

    # 2) Limpia espacios en nombres (por si acaso)
    df_rho.index   = df_rho.index.str.strip()
    df_rho.columns = df_rho.columns.str.strip()

    # 3) Función que convierte cada celda a float, manejando datetimes
    def to_float(x):
        # Celdas vacías
        if pd.isna(x):
            return 0.0
        # Si es fecha (Timestamp o date), construimos dd.mm
        if isinstance(x, (pd.Timestamp, datetime.datetime, datetime.date)):
            # día + mes/100  →  e.g. 8 + 12/100 = 8.12
            return x.day + x.month/100
        # Si ya es número
        if isinstance(x, (int, float)):
            return float(x)
        # Si es string, lo limpiamos y convertimos
        s = str(x).strip()
        return float(s)

    # 4) Aplica la función a toda la tabla
    #    Usamos apply sobre columnas para evitar el warning de applymap
    df_rho = df_rho.apply(lambda col: col.map(to_float))

    # 5) Finalmente pasamos al diccionario
    rho = {
        (s1, s2): df_rho.at[s1, s2]
        for s1 in df_rho.index
        for s2 in df_rho.columns
    }


    # Índice de criminalidad por (comuna, franja, día)
    df_i = pd.read_excel("IndiceCrimenxComuna.xlsx")
    isft = { (row["Comuna"], row["Franja Horaria"], row["Día"]):
        row["Índice Criminalidad"]
        for _, row in df_i.iterrows() }

    # Mínimos de vehículos y material por comuna
    df_min = pd.read_excel("Comunas.xlsx")
    etas = df_min.set_index("Comuna")["Min_vehiculos"].to_dict()
    ms = df_min.set_index("Comuna")["Min_material"].to_dict()

    data = {
        "C": C,
        "S": S,
        "dc": dc,
        "Cs": Cs,
        "F": F,
        "T": T,
        "P": P,
        "B": B,
        "G": G,
        "gammab": gammab,
        "beta": beta,
        "isft": isft,
        "etas": etas,
        "lambdag": lambdag,
        "psig": psig,
        "kappab": kappab,
        "xi": xi,
        "rho": rho,
        "epsilon": epsilon,
        "deltab": deltab,
        "ms": ms,
        "alpha": alpha
    }
    return data

data = cargar_datos()
print("Hola")
print(data["alpha"]*2)
print(data["rho"]["Lo Barnechea", "Huechuraba"])

