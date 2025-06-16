import pandas as pd
from gurobipy import Model, GRB
from gurobipy import quicksum

def cargar_datos():
    # Comisarías por comuna
    df_C = pd.read_csv("ComisariasXComuna.csv")
    C = df_C["id_comisaria"].tolist()
    S = df_C["comuna"].unique().tolist()
    dc = df_C.set_index("id_comisaria")["min_carabineros"].to_dict()
    Cs = df_C.groupby("comuna")["id_comisaria"].apply(list).to_dict() 
            #si quiero la cantidad en ves de los id es .apply(len)

    # Franjas horarias
    df_F = pd.read_csv("FranjaHoraria.csv", header=None)
    F = df_F.iloc[:,0].tolist()

    # Días
    df_T = pd.read_csv("Dias.csv", header=None)
    T = df_T.iloc[:,0].tolist()

    # Sueldos de carabineros
    df_P = pd.read_csv("SueldoXCarabinero.csv")
    P = df_P["id_carabinero"].tolist()
    beta = df_P.set_index("id_carabinero")["sueldo"].to_dict()

    # Material complementario
    df_G = pd.read_csv("Materiales.csv")
    G = df_G["id_material"].tolist()
    psig = df_G.set_index("id_material")["efectividad"].to_dict()
    lambdag = df_G.set_index("id_material")["costo_fijo"].to_dict()

    # Vehículos
    df_B = pd.read_csv("CostoXVehiculo.csv")
    B = df_B["id_vehiculo"].tolist()
    gammab = df_B.set_index("id_vehiculo")["costo_fijo"].to_dict()
    kappab = df_B.set_index("id_vehiculo")["efectividad"].to_dict()
    deltab = df_B.set_index("id_vehiculo")["maximo_carabineros"].to_dict()

    # Parámetros sueltos
    df_param = pd.read_csv("ParametrosSueltos.csv")
    xi = df_param.loc[0, "efectividad_patrullaje_sin_vehiculo"]
    epsilon = df_param.loc[0, "dist_maxima_desplazamiento"]
    alpha = df_param.loc[0, "presupuesto_inicial"]

    # Distancias entre comunas (matriz)
    df_rho = pd.read_csv(
        "DistanciaEntreComunas2.csv",
        index_col=0,
        header=0
    )
    df_rho.index   = df_rho.index.str.strip()
    df_rho.columns = df_rho.columns.str.strip()
    rho = {
        (s1, s2): df_rho.at[s1, s2]
        for s1 in df_rho.index
        for s2 in df_rho.columns
    }

    # Índice de criminalidad por (comuna, franja, día)
    df_i = pd.read_csv("IndiceCrimenxComuna.csv")
    isft = { (row["Comuna"], row["Día"], row["Franja Horaria"]):
        row["Índice Criminalidad"]
        for _, row in df_i.iterrows() }

    # Mínimos de vehículos y material por comuna
    df_min = pd.read_csv("Comunas.csv")
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



def construir_modelo(data):
    model = Model()
    model.setParam("TimeLimit", 1800)
    S = data["S"]
    F = data["F"]
    T = data["T"]
    C = data["C"]
    C_s = data["Cs"]
    P = data["P"]
    B = data["B"]
    G = data["G"]
    dc = data["dc"]
    alpha =  data["alpha"]
    i = data["isft"]
    gamma = data["gammab"]
    beta = data["beta"]
    lambda_ = data["lambdag"]
    eta = data["etas"]
    delta = data["deltab"]
    xi = data["xi"]
    kappa = data["kappab"]
    psi = data["psig"]
    rho = data["rho"]
    epsilon = data["epsilon"]
    ms = data["ms"]
    M = 1e6

    X = model.addVars(P, C, T, F, vtype=GRB.BINARY, name="x_pctf")
    W = model.addVars(S, B, T, F, vtype = GRB.BINARY, name = "w_sbtf")
    J = model.addVars(S, B, P, T, F, vtype = GRB.BINARY, name = "j_sbptf")
    L = model.addVars(S, P, T, F, vtype = GRB.BINARY, name = "l_sptf")
    U = model.addVars(S, P, T, F, vtype = GRB.BINARY, name = "u_sptf")
    A = model.addVars(G, S, T, vtype = GRB.BINARY, name = "a_gst")
    K = model.addVars(T, vtype = GRB.CONTINUOUS, lb=0, name = "k_t")
    V = model.addVars(S, T, F, vtype = GRB.CONTINUOUS, lb=0, name = "l_sptf")
    model.update()

    """
    model.addConstr(
    K[30] == alpha
    - quicksum(W[s, b, t, f] * gamma[b] for t in range(1, 31) for f in F for s in S for b in B)
    - quicksum(beta[p] for p in P)
    - quicksum(A[g, s, t] * lambda_[g] for t in range(1, 31) for s in S for g in G),
    name="R1.1"
    )

    model.addConstr(
        K[60] == K[30]
        - quicksum(W[s, b, tp, f] * gamma[b] for tp in range(31, 61) for f in F for s in S for b in B)
        - quicksum(beta[p] for p in P)
        - quicksum(A[g, s, tp] * lambda_[g] for tp in range(31, 61) for s in S for g in G),
        name="R1.2"
    )
    """
    model.addConstrs((quicksum(X[p, c, t, f] for p in P) >= dc[c] for c in C for t in T for f in F), name="R2")
    model.addConstrs((quicksum(J[s, b, p, t, f] for s in S) <= 1 for p in P for t in T for b in B for f in F), name="R3")
    model.addConstrs((quicksum(W[s, b, t, f] for b in B) >= eta[s] for s in S for t in T for f in F), name="R4")
    model.addConstrs((quicksum(X[p, c, t, f] for c in C) <= 1 for p in P for t in T for f in F), name="R5")
    model.addConstrs((quicksum(X[p, c, t, f] for f in F for c in C) <= 2 for p in P for t in T), name="R6")
    model.addConstrs((quicksum(J[s, b, p, t, f] for b in B) <= U[s, p, t, f] for s in S for p in P for t in T for f in F), name="R7")
    model.addConstrs((quicksum(J[s, b, p, t, f] for p in P) <= delta[b] for s in S for b in B for t in T for f in F), name="R8")
    model.addConstrs(
    (
        V[s, t, f] ==
        quicksum(L[s, p, t, f] * xi for p in P) +
        quicksum(J[s, b, p, t, f] * kappa[b] for b in B for p in P) +
        quicksum(A[g, s, t] * psi[g] for g in G)
        for s in S for t in T for f in F
    ),
    name="R9"
    )
    model.addConstrs((quicksum(X[p, c, t, f] for p in P for c in C_s[s]) >= quicksum(U[s, p, t, f] for p in P) for s in S for t in T for f in F), name="R10")

    model.addConstrs((0.5 * quicksum(J[s1, b, p, t, f] * rho[s1, s2] for s1 in S for f in F for b in B) <= epsilon for s2 in S for t in T for p in P), name="R11") #media SUS esta restricción, puede causar problemas.
    model.addConstrs((J[s1, b, p, t, f] + J[s2, b, p, t, f] <= 1 for s1 in S for s2 in S if s1 != s2 for b in B for p in P for t in T for f in F), name="R12")
    model.addConstrs((quicksum(A[g, s, t] for t in T for g in G) >= ms[s] for s in S), name="R13")
    model.addConstrs((quicksum(J[s, b, p, t, f] for p in P) <= M * W[s, b, t, f] for s in S for b in B for t in T for f in F), name="R14.1")
    model.addConstrs((W[s, b, t, f] <= quicksum(J[s, b, p, t, f] for p in P) for s in S for b in B for t in T for f in F), name="R14.2")

    model.setObjective(
    quicksum(V[s, t, f] * i[s, t, f] for s in S for t in T for f in F),
    GRB.MAXIMIZE
)
    return model


def resolver_modelo(model):
    model.optimize()
    return model


def imprimir_resultados(model):
    status = model.Status
    if status in (GRB.OPTIMAL, GRB.SUBOPTIMAL, GRB.TIME_LIMIT):
        print("\n--- Resultados ---")
        if model.ObjVal == float("-inf"):
            print("No se encontró ninguna solución factible dentro del límite de tiempo.\n")
        else:
            print(f"Valor de la función objetivo: {model.ObjVal:.2f}\n")
    else:
        print(f"No se encontró solución factible. Status del modelo: {status}")

def imprimir_asignaciones_utiles(model, data):

    S, F, T, P, C, B, G = data["S"], data["F"], data["T"], data["P"], data["C"], data["B"], data["G"]

    df_resultados = []

    for var in model.getVars():
        if var.X > 0.5:
            nombre = var.VarName
            if nombre.startswith("x_pctf["):
                p, c, t, f = nombre.replace("x_pctf[", "").replace("]", "").split(",")
                df_resultados.append(["Asignación Carabinero", p, c, int(t), f, "—"])
            elif nombre.startswith("u_sptf["):
                s, p, t, f = nombre.replace("u_sptf[", "").replace("]", "").split(",")
                df_resultados.append(["Carabinero Patrullando", s, p, int(t), f, "—"])
            elif nombre.startswith("w_sbtf["):
                s, b, t, f = nombre.replace("w_sbtf[", "").replace("]", "").split(",")
                df_resultados.append(["Vehículo Asignado", s, b, int(t), f, "—"])
            elif nombre.startswith("a_gst["):
                g, s, t = nombre.replace("a_gst[", "").replace("]", "").split(",")
                df_resultados.append(["Material Usado", s, g, int(t), "—", "—"])
            # elif nombre.startswith("l_sptf["):
                # s, p, t, f = nombre.replace("l_sptf[", "").replace("]", "").split(",")
                # df_resultados.append(["Carabinero Patrullando sin Vehículo", s, p, int(t), f, "—"])
            elif nombre.startswith("v_stf["):
                s, t, f = nombre.replace("v_stf[", "").replace("]", "").split(",")
                df_resultados.append(["Nivel Patrullaje", s, "—", int(t), f, round(var.X, 2)])


    df = pd.DataFrame(df_resultados, columns=["Tipo", "Comuna / C", "ID", "Día", "Franja", "Valor"])
    df = df.sort_values(["Tipo", "Comuna / C", "Día", "Franja"])
    pd.set_option('display.max_rows', 200)
    print(df)
    return df


def main():
    data = cargar_datos()
    model = construir_modelo(data)
    resultado = resolver_modelo(model)
    imprimir_resultados(resultado)
    imprimir_asignaciones_utiles(model, data)


if __name__ == "__main__":
    main()

