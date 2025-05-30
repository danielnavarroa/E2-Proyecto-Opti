import pandas as pd
from gurobipy import Model, GRB
from gurobipy import quicksum

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
    df_rho = pd.read_excel("DistanciaEntreComunas.xlsx", index_col=None)
    rho = { (s1, s2): df_rho.at[s1, s2]
           for s1 in df_rho.index
           for s2 in df_rho.columns }

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


def construir_modelo(data):
    model = Model()
    model.setParam("TimeLimit", 60)
    #S = data
    #F = data
    #T = data
    #C = data
    #C_s = data
    #P = data
    #B = data
    #G = data
    #dc = data
    #alpha =  data
    #gamma = data
    #beta = data
    #lambda_ = data
    #eta = data
    #delta = data
    #xi = data
    #kappa = data
    #psi = data
    #rho = data
    #epsilon = data
    #ms = data
    M = 1e6

    X = model.addVars(P, C, T, F, vtype = GRB.BINARY, name = "x_pctf")
    W = model.addVars(S, B, T, F, vtype = GRB.BINARY, name = "w_sbtf")
    J = model.addVars(S, B, P, T, F, vtype = GRB.BINARY, name = "j_sbptf")
    L = model.addVars(S, P, T, F, vtype = GRB.BINARY, name = "l_sptf")
    U = model.addVars(S, P, T, F, vtype = GRB.BINARY, name = "u_sptf")
    A = model.addVars(G, S, T, vtype = GRB.BINARY, name = "a_gst")
    K = model.addVars(T, vtype = GRB.CONTINUOUS, lb=0, name = "k_t")
    V = model.addVars(S, T, F, vtype = GRB.CONTINUOUS, lb=0, name = "l_sptf")
    model.update()

    model.addConstr(
    K[30] == alpha
    - quicksum(W[s, b, t, f] * gamma[b] for t in range(1, 31) for f in F for s in S for b in B)
    - quicksum(beta[p] for p in P)
    - quicksum(A[g, s, t] * lambda_[g] for t in range(1, 31) for s in S for g in G),
    name="R1.1"
    )
    for t in range(60, 361, 30):
        model.addConstr(
            K[t] == K[t-30]
            - quicksum(W[s, b, tp, f] * gamma[b] for tp in range(t-29, t+1) for f in F for s in S for b in B)
            - quicksum(beta[p] for p in P)
            - quicksum(A[g, s, tp] * lambda_[g] for tp in range(t-29, t+1) for s in S for g in G),
            name=f"R1.{t/30}"
        )
    model.addConstrs((quicksum(X[p][c][t][f] >= dc[c] for p in P) for c in C for t in T for f in F), name="R2")
    model.addConstrs((quicksum(W[s, b, t, f] for b in B) >= eta[s] for s in S for t in T for f in F), name="R3")
    model.addConstrs((quicksum(X[p][c][t][f] for c in C) <= 1 for p in P for t in T for f in F), name="R4")
    model.addConstrs((quicksum(X[p][c][t][f] for f in F) <= 2 for p in P for t in T for c in C), name="R5")
    model.addConstrs((quicksum(J[s][b][p][t][f] for b in B) <= U[p][s][t][f] for s in S for p in P for t in T for f in F), name="R6")
    model.addConstrs((quicksum(J[s][b][p][t][f] for p in P) <= delta[b] for s in S for b in B for t in T for f in F), name="R7")
    model.addConstrs(
    (
        V[s, t, f] ==
        quicksum(L[s, p, t, f] * xi for p in P) +
        quicksum(J[s, b, p, t, f] * kappa[b] for b in B for p in P) +
        quicksum(A[g, s, t] * psi[g] for g in G)
        for s in S for t in T for f in F
    ),
    name="R8"
    )
    model.addConstrs((quicksum(X[p][c][t][f] for c in C[s]) <= quicksum(U[p][s][t][f] for p in P) for s in S for t in T for f in F), name="R9")
    model.addConstrs((0.5 * quicksum(J[s, b, p, t, f] * rho[s1][s2] for s1 in S for f in F for b in B) <= epsilon for s2 in S for t in T for p in P), name="R10") #media SUS esta restricción, puede causar problemas.
    model.addConstrs((J[s1, b, p, t, f] + J[s2, b, p, t, f] <= 1 for s1 in S for s2 in S if s1 != s2 for b in B for p in P for t in T for f in F), name="R11")
    model.addConstrs((quicksum(A[g][s][t] for t in T for g in G) >= ms for s in S), name="R12")
    model.addConstrs((quicksum(J[s][b][p][t][f] for p in P) <= M * W[s][b][t][f] for s in S for b in B for t in T for f in F), name="R13.1")
    model.addConstrs((W[s][b][t][f] <= quicksum(J[s][b][p][t][f] for p in P) for s in S for b in B for t in T for f in F), name="R13.2")

    model.setObjective(
    quicksum(v[s, t, f] * i[s, t, f] for s in S for t in T for f in F),
    GRB.MAXIMIZE
)
    return model
    

def resolver_modelo(model):

    model.optimize()
    return model

def imprimir_resultados(model):
    if model.Status == GRB.OPTIMAL or model.Status == GRB.TIME_LIMIT:
        print("\n--- Resultados ---")
        print(f"Valor óptimo del beneficio neto: ${model.ObjVal:.2f} pesos\n")

        print("Cantidad producida de cada producto (xᵢ):")
        for var in model.getVars():
            if var.varName.startswith("x"):
                idx = int(var.varName.split('[')[-1].rstrip(']')) + 1
                print(f"  Producto {str(idx)}: {var.X:.2f} unidades")
    else:
        print("No se encontró una solución óptima.")

def main():
    data = cargar_datos()
    model = construir_modelo(data)
    resultado = resolver_modelo(model)
    imprimir_resultados(resultado)

if __name__ == "__main__":
    main()
