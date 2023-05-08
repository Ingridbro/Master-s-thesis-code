import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

"""This code is based on the MATLAB code Windhydtools by Magnus Korpås, 2004
    that can be found here https://www.ntnu.no/ansatte/magnus.korpas"""

def readdata(sim_type):
    if sim_type == 'single':
        # **** SINGLE SIMULATION ****
        # Read component parameters from excel

        df = pd.read_excel("../singlesim1_pv_.xlsx", sheet_name='Component parameters')
        CP = df.iloc[11:30, 4].values.flatten() #E13:30 in Excel

        df = pd.read_excel("../singlesim1_pv_.xlsx", sheet_name='Solar and loads')
        solar_loads = df.iloc[13:16, 4].values.flatten() #E15:17 in Excel
        CP[13] = solar_loads[1]
        CP[14] = solar_loads[2]
        CP[15] = solar_loads[0]

        df = pd.read_excel("../singlesim1_pv_.xlsx", sheet_name='Simulation parameters')
        SP = df.iloc[10:20, 3].values.flatten() #D11:20 in Excel

        df = pd.read_excel("../singlesim1_pv_.xlsx", sheet_name='Control strategy')
        CS = int(df.iloc[28:29, 3].values.flatten())  #D30 in Excel
        CS = [CS]

        B = np.zeros((7, 6))

        #*** ELECTROLYZER ***
        B[0, 0] = 1  # Only one electrolyzer type so far
        B[0, 1] = 0  # Always zero so far
        B[0, 2] = CP[1] # Electrolyzer rating (Pe_max) = 30 000kW
        B[0, 3] = CP[8] # Electrolyzer efficiency (eta_e) = 70%
        B[0, 4] = CP[10]  # LHV of hydrogen = 2.995
        #*** FUEL CELL ***
        B[1, 0] = 1  # Same as electrolyzer
        B[1, 1] = 0  # Same as electrolyzer
        B[1, 2] = CP[5] # Fuel cell rating (Pf_max = 3 000 kW)
        B[1, 3] = CP[9] # Fuel cell efficiency = 50%
        B[1, 4] = CP[10]  # LHV of hydrogen = 2.995
        
        VH_max = CP[2] # 50 000Nm3 - hydrogen storage capacity
        rat_VH_0 = CP[3] # 50% of capacity
        rat_VH_secur = CP[4] # 50% of capacity

        VH_0 = VH_max*rat_VH_0*0.01 # 25 000Nm3
        VH_secur = VH_max*rat_VH_secur*0.01 # 25 000Nm3

        #*** STORAGE ***
        B[2, 0] = 1 # Type
        B[2, 1] = 0 # Minimum storage level
        B[2, 2] = VH_max # Maximum storage level
        B[2, 3] = VH_0 # Initial storage level
        B[2, 4] = VH_secur # Lower security limit of hydrogen storage

        # *** RENEWABLE CONVERTER ***
        B[3,0] = 2 #Resource 2 is Solar PV
        B[3,2] = CP[0] # Solar PV rating
        B[3,3] = CP[15] #average wind speed - 7.055 m/s
        B[3,4] = CP[16] #Coefficient of variation for wind speed (initialized as 0)
        B[3,5] = CP[18] # Solar PV rating

        #*** GRID ***
        B[4,0] = 1 #same as electrolyzer
        B[4,1] = CP[6] # grid import capacity
        B[4,2] = CP[7] # grid export capacity
        B[4,3] = 0 #Power losses is not considered

        #*** ELECTRICAL LOAD ***
        B[5,0] = CP[13] #average electrical load
        B[5,1] = CP[17] #Coefficient of variation for electrical load (initialized as 0)

        return B, SP, CS
    
def PV_from_csv():
    rawdata = np.loadtxt("../../../PV/PVsyst7.0_Data/UserHourly/Base case + north terrain east-west.txt")
    P = rawdata[0:8760]
    P = np.array(P)

    # Return the production values as a numpy array
    return P
    
def renewconv(resource, Rating):
    # ******* Check input arguments *********
    # NR means "Negative Rating"
    NR = np.where(Rating < 0)[0]

    if NR.size > 0:
        raise ValueError("Negative values of Rating")

    # ********** Find power output *********
    if resource == 2:
        #Solar PV
        P = PV_from_csv()
    else:
        raise ValueError("Non-existing renewable resource.")

    # Check power output
    if np.any(P < 0):
        raise ValueError("P is negative.")
    elif np.size(P) == 0:
        raise ValueError("P is an empty matrix.")
        
    return P
    
def whtimeseries(SP, B,):
    # Read data from time series files and put in matrix P

    # Extract info for simulation period SP
    t_start = SP[0] - 1  # simulation period 1 = 1
    t_end = SP[1] - 1  # simulation period 2 = 8760
    P = np.zeros((t_end-t_start+1, 2))  # create matrix with 4 columns and 8760 rows

    # **** Creating timeseries for renewable converter ****
    Resource = B[3, 0] # Resource 1 is wind
    PVRating = B[3, 5] # PV rating

    rawdata = np.zeros((8760, 1))
    P_PV = np.zeros((t_end-t_start+1, 1))
    P_l = P_PV.copy()

    if Resource == 2:
        #Solar PV
        P_PV = PV_from_csv()
    else:
        raise ValueError('Non-existing renewable resource.')

    P_PV = renewconv(Resource, PVRating)

    # **** Creating timeseries for electrical load ****
    P_l_max = B[5, 0] #Pl_max
    rawdata = np.loadtxt("../../../PV/PVsyst7.0_Data/UserHourly/Electrical load (normalized).txt")
    P_l = P_l_max * rawdata[t_start:(t_end+1)]

    # Inserting time series data in P:
    P[:, 0] = P_PV # solar PV power series
    P[:, 1] = P_l # electrical load series

    return P

def enbal(x0, p, B, CS, dt):
    idt = 1/dt
    epsilon = 1e-3

    VH_0 = x0
    Ppv = p[0]
    Pl = p[1]
    FHl = p[2]
    
    Pe_max = B[0,2] #Electrolyzer rating (Pe_max = 3 000 kW)
    eta_e = B[0,3] #Electrolyzer efficiency = 70%
    spc_e_ref = B[0,4] #Reference value for power consumption - 2.9950
    spc_e = 100*spc_e_ref/eta_e #Specific power consumption
    ispc_e = 1/spc_e #Inverse of spc_e
    
    Pf_max = B[1,2] #Fuel cell rating (Pf_max = 3 000 kW)
    eta_f = B[1,3] #Fuel cell efficiency = 50%
    spg_f_ref = B[1,4] #Reference value for power generation - 2.9950
    spg_f = eta_f*spg_f_ref/100 #Specific power generation
    ispg_f = 1/spg_f #Inverse of spg_f
    
    VH_max = B[2,2]
    VH_secur = B[2,3]
    
    Pg_imp_max = B[4,1]
    Pg_exp_max = B[4,2]
    

    Pe = 0 # Initialize
    Pf = 0 # Initialize
    FHnp = 0 # Initialize
    FHe = 0 # Initialize
    FHf = 0 # Initialize
    # Hydrogen storage level
    VH = VH_0 + dt * (FHe - FHf - FHl)
    
    if CS[0] == 1:
            # ******** Self-supplied with electricity and hydrogen ********

        # Maximum electrolyzer power (cannot import above grid limit)
        Pe_max = min(Pe_max, Pg_imp_max + Ppv - Pl)
        Pe_max = max(Pe_max, 0)

        Pbal = Ppv - Pl
        if Pbal >= 0:
            # Run electrolyzer
            Pf = 0  # Do not run fuel cell
            FHf = 0  # Do not run fuel cell
            FHnp = 0  # Initialize

            Pe = min(Pe_max, Pbal)  # Limitation on max power
            FHe = ispc_e * Pe

            # Hydrogen storage level
            VH = VH_0 + dt * (FHe - FHf - FHl)

            if VH < (VH_secur - epsilon):  # Increase hydrogen production
                FHe = (VH_secur - VH_0) * idt + FHl
                Pe = min(Pe_max, spc_e * FHe)
                FHe = ispc_e * Pe

                # Update storage level
                VH = VH_0 + dt * (FHe - FHf - FHl)

            elif VH > (VH_max + epsilon):  # Decrease hydrogen production
                FHe = (VH_max - VH_0) * idt + FHl
                Pe = min(Pe_max, spc_e * FHe)
                FHe = ispc_e * Pe

                # Update storage level
                VH = VH_0 + dt * (FHe - FHf - FHl)
        else:
            # Run fuel cell
            Pe = 0  # Do not run electrolyzer
            FHe = 0  # Do not run electrolyzer
            FHnp = 0  # Do not run electrolyzer

            Pf = min(Pf_max, -Pbal)
            #Pf = max(0, -Pbal)
            #Pf = min(Pf_max, Pf)
            FHf = ispg_f * Pf

            # Hydrogen storage level
            VH = VH_0 + dt * (FHe - FHf - FHl)

            if VH < (VH_secur - epsilon):  # Decrease fuel cell power
                FHf = (VH_0 - VH_secur) * idt - FHl
                Pf = max(0, spg_f * FHf)
                FHf = ispg_f * Pf

                # Update storage level
                VH = VH_0 + dt * (FHe - FHf - FHl)

                if VH < (VH_secur - epsilon):  # Increase H2 production
                    FHe = (VH_secur - VH_0) * idt + FHl
                    Pe = min(Pe_max, spc_e * FHe)
                    FHe = ispc_e * Pe

                    # Update storage level
                    VH = VH_0 + dt * (FHe - FHf - FHl)
          
    # Check hydrogen storage
    if VH < (0-epsilon):
        # Hydrogen load is not supplied
        FHnp = (0 - VH)*idt
        VH = 0

    # Updated power balance and hydrogen balance
    Pbal = Ppv-Pl+Pf-Pe
    FHout = FHl - FHnp

    Pns = 0
    if Pbal > 0:
        # Export to grid and/or power dumping
        Pd = max(0, Pbal-Pg_exp_max)
        Pg = min(Pg_exp_max, Pbal)
    else:
        # Import from grid or local balance achieved
        Pg = Pbal
        Pd = 0
        
        if -Pg > (Pg_imp_max + epsilon):
            # Electrical energy not supplied
            Pns = -Pg - Pg_imp_max
            Pg = -Pg_imp_max

    if (Pe > (0 + epsilon)) and (Pf > (0 + epsilon)):
        raise Exception('Cannot operate fuel cell and electrolyzer at the same time!')

    if (Pd > (0 + epsilon)) and (Pf > (0 + epsilon)):
        raise Exception('Cannot dump power when operating fuel cell!')

    if (FHnp > (0 + epsilon)) and (Pf > (0 + epsilon)):
        raise Exception('Cannot dump hydrogen when operating fuel cell!')

    if (VH < (VH_secur - epsilon)) and (Pf > (0 + epsilon)):
        raise Exception('Cannot operate fuel cell if VH < VH_secur!')

    Pcheck = Ppv + Pf - Pd - Pe - Pl - Pg + Pns
    if (Pcheck > epsilon) or (Pcheck < -epsilon):
        raise Exception('Power balance not fulfilled!')

    VHcheck = ispc_e*Pe*dt - ispg_f*Pf*dt - FHout*dt + VH_0 - VH

    if (VHcheck > epsilon) or (VHcheck < -epsilon):
        raise Exception('Hydrogen balance not fulfilled!')
    
    x = [VH]
    u = [Pe, Pf, FHout]
    w = [Pd, Pg, FHnp, Pns]

    return x, u, w

def simloop(P, SP, B, CS):
    # Extract simulation parameters from SP-vector.
    T = int(SP[2])  # Number of timesteps
    dt = SP[3]

    # Initialize X, U and W
    X = np.zeros((T, 1)) # state variables (hydrogen storage level VH)
    U = np.zeros((T, 3)) # control variables (Pe, Pf, Hout)
    W = np.zeros((T, 4)) # dependent variables (Pd, Pg, FHnp, Pnp)

    # Initial storage level
    x0 = B[2, 3]

    for t in range(T):
        # Parameters for timestep t.
        p = P[t, :]

        # Operate the system in timestep t.
        x, u, w = enbal(x0, p, B, CS, dt)

        # Store variables for timestep t in matrix.
        X[t, :] = x
        U[t, :] = u
        W[t, :] = w

        # Initial value for state variables at timestep t+1
        x0 = x[0]

    return X, U, W

def mainresults(P, X, U, W, SP, B, CS):
    # Self-supplied with electricity and hydrogen
    P_w = P[:, 0]
    P_load = P[:, 1]
    #H_load = P[:, 2]

    VH = X[:, 0]

    P_ely = U[:, 0]
    P_fc = U[:, 1]
    H_out = U[:, 2]

    P_dump = W[:, 0]
    P_g = W[:, 1]
    H_def = W[:, 2]
    P_np = W[:, 3]

    # Find imported energy
    P_imp = np.zeros(P_w.shape)
    P_exp = np.zeros(P_w.shape)

    IMP = np.where(P_g < 0)[0]
    EXP = np.where(P_g >= 0)[0]

    P_imp[IMP] = -P_g[IMP]
    P_exp[EXP] = P_g[EXP]

    P_eg = np.zeros(P_w.shape)
    P_lg = np.zeros(P_w.shape)

    switch_val = CS[0]
    if switch_val in [1, 3, 4]:
        # Electrical load highest priority
        IMP1 = np.where((P_g < 0) & (P_w < P_load))[0]

        P_eg[IMP1] = P_ely[IMP1]
        P_lg[IMP1] = P_load[IMP1] - P_w[IMP1] + P_dump[IMP1] - P_fc[IMP1] - P_np[IMP1]

        IMP2 = np.where((P_g < 0) & (P_w >= P_load))[0]
        P_eg[IMP2] = -P_g[IMP2]

    elif switch_val == 2:
        # Hydrogen production highest pri
        IMP1 = np.where(P_w < P_ely)[0]
        P_eg[IMP1] = P_ely[IMP1] - P_w[IMP1]
        P_lg[IMP1] = P_load[IMP1] - P_np[IMP1]

        IMP2 = np.where((P_w >= P_ely) & (P_g < 0))[0]
        P_lg[IMP2] = -P_g[IMP2]

    dt = SP[3]

    # Mean hydrogen level
    VH_mean = np.mean(VH)  # Her er det feil for simulering over flere år?
    VH_0 = B[2, 3]
    VH_end = VH[-1]

    # Energy E
    E_w = sum(P_w) * dt
    E_load = sum(P_load) * dt
    E_ely = sum(P_ely) * dt
    E_fc = sum(P_fc) * dt
    E_dump = sum(P_dump) * dt
    E_g = sum(P_g) * dt
    E_ns = sum(P_np) * dt

    E_imp = sum(P_imp) * dt
    E_exp = sum(P_exp) * dt

    E_eg = sum(P_eg) * dt
    E_lg = sum(P_lg) * dt

    E_w_net = E_w - E_dump

    # Fuel cell power to grid
    FG = [i for i in range(len(P_fc)) if P_fc[i] > P_load[i]]
    E_fg = sum([P_fc[i] - P_load[i] for i in FG])

    # Wind and fuel cell power to load
    E_wfl = E_load - E_ns - E_lg

    # Hydrogen
    #VH_load = sum(H_load) * dt
    VH_out = sum(H_out) * dt
    VH_def = sum(H_def) * dt

    # Hydrogen produced from grid power
    eta_e = B[0, 3]  # in percentage
    spc_e_ref = B[0, 4]  # Reference value for power consumption
    spc_e = 100 * spc_e_ref / eta_e  # Specific power consumption
    VH_eg = E_eg / spc_e

    # Total hydrogen production
    VH_e = E_ely / spc_e

    # Hydrogen consumed in fuel cell
    eta_f = B[1, 3]  # in percentage
    spg_f_ref = B[1, 4]  # Reference value for power generation
    spg_f = eta_f * spg_f_ref / 100  # Specific power generation
    VH_f = E_fc / spg_f

    # Store main results in array and send to excel
    ResArray = [E_w, E_load, E_ely, E_fc, E_dump, E_g, E_imp, E_exp, E_eg, E_lg, E_w_net, E_fg, E_wfl, E_ns, VH_mean, VH_0, VH_end, VH_out, VH_def, VH_eg, VH_e, VH_f]
    ResArray = np.array(ResArray)
    ResArray = ResArray.reshape(-1, 1)

    return ResArray