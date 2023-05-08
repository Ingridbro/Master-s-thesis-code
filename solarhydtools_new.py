import numpy as np
import pandas as pd
from pyomo.environ import *
import matplotlib.pyplot as plt

"""This code is based on the MATLAB code Windhydtools by Magnus Korpås, 2004
    that can be found here https://www.ntnu.no/ansatte/magnus.korpas"""

def readdata(sim_type):
    if sim_type == 'single':
        # **** SINGLE SIMULATION ****
        # Read component parameters from excel

        # CP = all sizing and efficiency parameters, 12 parameters
        df = pd.read_excel("../singlesim1_pv_.xlsx", sheet_name='Component parameters')
        CP = df.iloc[11:30, 4].values.flatten() #E13:30 in Excel
        #print("CP: " + str(CP) + "\n")

        # Read solar and loads from excel
        df = pd.read_excel("../singlesim1_pv_.xlsx", sheet_name='Solar and loads')
        solar_loads = df.iloc[13:16, 4].values.flatten() #E15:17 in Excel
        #print("solar_loads" + str(solar_loads) + "\n")

        CP[13] = solar_loads[1] # Pl_max - max load (kWh)
        CP[14] = solar_loads[2] # FHL_avg - average hydrogen load (Nm3/h)
        CP[15] = solar_loads[0] # PV_avg - average solar production (kW)
        #print("CP: " + str(CP) + "\n")

        #Read input/output from wind turbine
        #InValues = A25:A55 i Solar and loads
        InValues = df.iloc[23:54,0].values.flatten() #A25:A55 in Excel
        #print("InValues" + str(InValues) + "\n")
        OutValues = df.iloc[23:54,3].values.flatten() #D25:D55 in Excel
        #print("OutValues" + str(OutValues) + "\n")
        InOutPair = np.column_stack((InValues, OutValues))
        #print("InOutPair" + str(InOutPair) + "\n")

        # Extracting text files for solar power production, normalized electrical load data file and hydrogen load data file
        TS = df.iloc[8:11, 1].values.flatten() #B10:12 in Excel
        print("TS: " + str(TS) + "\n")

        # Extracting normalized solar PV power data file
        TS_pnorm = df.iloc[6:8, 1].values.flatten() #B8:9 in Excel
        #print("TS_pnorm: " + str(TS_pnorm) + "\n")

        # Read simulation parameters from excel
        df = pd.read_excel("../singlesim1_pv_.xlsx", sheet_name='Simulation parameters')
        SP = df.iloc[10:20, 3].values.flatten() #D11:20 in Excel
        #print("SP: " + str(SP) + "\n")

        # Read control strategy
        df = pd.read_excel("../singlesim1_pv_.xlsx", sheet_name='Control strategy')
        CS = int(df.iloc[28:29, 3].values.flatten())  #D30 in Excel
        CS = [CS]
        #print("CS: " + str(CS) + "\n")

        return CP, TS, TS_pnorm, SP, CS, InOutPair

def createparam(CP):
    # Create component parameter matrix B
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
    B[2, 1] = VH_secur # Minimum storage level
    B[2, 2] = VH_max # Maximum storage level
    B[2, 3] = VH_0 # Initial storage level
    B[2, 4] = VH_secur # Lower security limit of hydrogen storage

    # *** RENEWABLE CONVERTER ***
    B[3,0] = 2 #Resource 2 is Solar PV
    B[3,1] = 3 #Wind turbine type
    B[3,2] = CP[0] # Wind rating - 20 000 kW
    B[3,3] = CP[15] #average wind speed - 7.055 m/s
    B[3,4] = CP[16] #Coefficient of variation for wind speed (initialized as 0)
    B[3,5] = CP[18] # Solar PV rating - 20 000 kW

    #*** GRID ***
    B[4,0] = 1 #same as electrolyzer
    B[4,1] = CP[6] #46 000kW - grid import capacity
    B[4,2] = CP[7] #46 000kW - grid export capacity
    B[4,3] = 0 #Power losses is not considered

    #*** ELECTRICAL LOAD ***
    B[5,0] = CP[13] #average electrical load - 11687.138 kW
    B[5,1] = CP[17] #Coefficient of variation for electrical load (initialized as 0)

    #*** HYDROGEN LOAD ***
    B[6,0] = CP[14] #average hydrogen load - 1008.89 Nm3/h

    return B

def PV_from_csv():
    rawdata = np.loadtxt("../../../PV/PVsyst7.0_Data/UserHourly/Base case + north terrain east-west.txt")
    P = rawdata[0:8760]
    P = np.array(P)
    print("P: ", P)

    # Return the production values as a numpy array
    return P


def renewconv(InFlow, resource, Rating, InOutPair):
    # ******* Check input arguments *********
    if InFlow.ndim > 2:
        raise ValueError("The dimension of InFlow is larger than 2.")

    if np.size(Rating) > 1:
        raise ValueError("The number of elements in Rating exceeds 1.")

    if InFlow.size == 0:
        raise ValueError("InFlow is an empty matrix.")

    if np.size(Rating) == 0:
        raise ValueError("Rating is an empty matrix.")

    # NIF means "Negative InFlow"
    NIF = np.where(InFlow < 0)[0]

    if NIF.size > 0:
        InFlow[NIF] = 0
        print("Warning: Negative values of InFlow")

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


def whtimeseries(SP, B, TS, TS_pnorm, InOutPair):
    # Read data from time series files and put in matrix P
    # P[:,0]: solar PV power series
    # P[:,1]: electrical load series
    # P[:,2]: hydrogen load series - set to 0
    # P[:,3]: wind speed series - set to 0

    # Extract info for simulation period SP
    t_start = SP[0] - 1  # simulation period 1 = 1
    t_end = SP[1] - 1  # simulation period 2 = 8760
    P = np.zeros((t_end-t_start+1, 4))  # create matrix with 4 columns and 8760 rows

    # **** Creating timeseries for renewable converter ****
    Resource = B[3, 0] # Resource 1 is wind
    Rating = B[3, 2] # Wind rating
    PVRating = B[3, 5] # PV rating

    rawdata = np.zeros((8760, 1))
    InFlow = np.zeros((t_end-t_start+1, 1))
    P_w = InFlow.copy()
    P_l = P_w.copy()
    P_PV = P_w.copy()
    H_l = P_w.copy()

    if Resource == 2:
        #Solar PV
        P_PV = PV_from_csv()
    else:
        raise ValueError('Non-existing renewable resource.')

    P_w = renewconv(InFlow, Resource, Rating, InOutPair)

    # **** Creating timeseries for electrical load ****
    P_l_max = B[5, 0] #Pl_max
    rawdata = np.loadtxt("../../../PV/PVsyst7.0_Data/UserHourly/Electrical load (normalized).txt")
    P_l = P_l_max * rawdata[t_start:(t_end+1)]

    # **** Creating timeseries for hydrogen demand ****
    H_l_avg = B[6, 0]
    rawdata = np.loadtxt("../h2_l_medsn_timenfor.txt")
    H_l = H_l_avg * rawdata[t_start:(t_end+1)]

    # Inserting time series data in P:
    P[:, 0] = P_PV
    P[:, 1] = P_l
    P[:, 2] = H_l
    #P[:,3] = InFlow
    P[:,3] = InFlow.ravel()

    #np.set_printoptions(threshold=np.inf) # Set threshold to infinity to print the whole array
    #print("P: ", P)

    return P

def enbal(x0, p, B, CS, dt):
    idt = 1/dt
    epsilon = 1e-3

    VH_0 = x0
    Pw = p[0]
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
        Pe_max = min(Pe_max, Pg_imp_max + Pw - Pl)
        Pe_max = max(Pe_max, 0)

        Pbal = Pw - Pl
        #print("Pbal: " + str(Pbal))
        #print("Pw: " + str(Pw))
        #print("Pl: " + str(Pl))
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

                if VH < (VH_secur - epsilon):  # Increase hydrogen production
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
    Pbal = Pw-Pl+Pf-Pe
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

    Pcheck = Pw + Pf - Pd - Pe - Pl - Pg + Pns
    if (Pcheck > epsilon) or (Pcheck < -epsilon):
        raise Exception('Power balance not fulfilled!')

    VHcheck = ispc_e*Pe*dt - ispg_f*Pf*dt - FHout*dt + VH_0 - VH
    #print("ispc_e: " + str(ispc_e) + "\n")
    #print("Pe: " + str(Pe) + "\n")
    #print("dt: " + str(dt) + "\n")
    #print("ispg_f: " + str(ispg_f)+ "\n")
    #print("Pf: " + str(Pf) + "\n")
    #print("FHout: " + str(FHout) + "\n")
    #print("FHnp:" + str(FHnp) + "\n")
    #print("FHl: "+ str(FHl) + "\n")
    #print("VH_0: " + str(VH_0) + "\n")
    #print("VH: " + str(VH) + "\n")
    #print("VHcheck: " + str(VHcheck) + "\n")
    # VH skal være null!! og ikke 2500
    if (VHcheck > epsilon) or (VHcheck < -epsilon):
        raise Exception('Hydrogen balance not fulfilled!')
    
    x = [VH]
    u = [Pe, Pf, FHout]
    w = [Pd, Pg, FHnp, Pns]

    return x, u, w

def simloop(P, SP, B, CS):
    """
    Function simloop runs the simulator for all timesteps
    X - state variables (hydrogen storage level VH)
    U - control variables (Pe, Pf, Hout)
    W - dependent variables (Pd, Pg, FHnp, Pnp)

    Parameters:
    P  - matrix of size (T, 6) containing the system inputs for each time step.
    SP - vector of size (1, 4) containing simulation parameters.
    B  - matrix of size (6, 4) containing system constants.
    CS - matrix of size (6, 6) containing system constants.

    Returns:
    X - state variables (hydrogen storage level VH)
    U - control variables (Pe, Pf, Hout)
    W - dependent variables (Pd, Pg, FHnp, Pnp)
    """
    
    # Extract simulation parameters from SP-vector.
    T = int(SP[2])  # Number of timesteps
    dt = SP[3]

    # Initialize X, U and W
    X = np.zeros((T, 1))
    U = np.zeros((T, 3))
    W = np.zeros((T, 4))

    # Initial storage level
    x0 = B[2, 3]

    for t in range(T):
        # Parameters for timestep t.
        # print("iteration number: ", t, " out of ", range(T))
        p = P[t, :]
        #print("p: ", p)

        # Operate the system in timestep t.
        x, u, w = enbal(x0, p, B, CS, dt)
        #print("x: ", x)
        #print("u: ", u)
        #print("w: ", w)

        # Store variables for timestep t in matrix.
        X[t, :] = x
        U[t, :] = u
        W[t, :] = w

        #print("X: ", X)
        #print("U: ", U)
        #print("W", W)

        # Initial value for state variables at timestep t+1
        x0 = x[0]
        #print("x0: ", x0)

        # print(100 * t / T)

    return X, U, W

def mainresults(P, X, U, W, SP, B, CS):
    # P - time-varying parameters (P_w, P_l, H_l, wspeed)
    # X - state variables (hydrogen storage level VH)
    # U - control variables (Pe, Pf, Hout)
    # W - dependent variables (Pd, Pg, FHnp, Pnp)
    # SP: simulation parameters
    # B:  component parameters
    # CS: control strategy
    # TS: Timeseries for wind and loads

    # Self-supplied with electricity and hydrogen
    P_w = P[:, 0]
    P_load = P[:, 1]
    H_load = P[:, 2]

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
    print("IMP: ", IMP)
    print("EXP: ", EXP)

    P_imp[IMP] = -P_g[IMP]
    P_exp[EXP] = P_g[EXP]

    P_eg = np.zeros(P_w.shape)
    P_lg = np.zeros(P_w.shape)

    switch_val = CS[0]
    if switch_val in [1, 3, 4]:
        # Electrical load highest pri
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

    # dt=SP(4)
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
    VH_load = sum(H_load) * dt
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
    ResArray = [E_w, E_load, E_ely, E_fc, E_dump, E_g, E_imp, E_exp, E_eg, E_lg, E_w_net, E_fg, E_wfl, E_ns, VH_mean, VH_0, VH_end, VH_load, VH_out, VH_def, VH_eg, VH_e, VH_f]
    ResArray = np.array(ResArray)
    ResArray = ResArray.reshape(-1, 1)

    return ResArray


def plotdata(P, X, U, W, SP):
    """
    Plots time-series data in WINDHYDSIM
    P - time-varying parameters (P_w, P_l, H_l, wspeed)
    X - state variables (hydrogen storage level VH)
    U - control variables (Pe, Pf, Hout)
    W - dependent variables (Pd, Pg, FHnp)
    SP: simulation parameters
    B:  component parameters
    CS: control strategy
    TS: Timeseries for wind and loads
    SP(5) > Plot time-series of electrical power?
    SP(6) > Plot time-series of hydrogen flows?
    SP(7) > Plot time-series of hydrogen storage level?
    SP(8) > Plot duration curves of electrical power?
    SP(9) > Plot duration curve of storage level?
    """
    excel_name = "../Results.xlsx"
    
    if not SP[4]:
        print("You have chosen not to plot.")
    else:
        print('...plotting time-series data in Python.')

        t_start = SP[0] #8760
        t_end = SP[1] #1
        
        t = np.arange(t_start, t_end+1) #creates numpy array with numbers from 1 to 8760
        
        if SP[5] == 1: # Plot time-series of electrical power? yes
            plt.figure()
            plt.subplot(3,1,1)  # Pw
            plt.plot(t, P[:,0])
            plt.ylabel('Solar power [kW]', fontsize=8)
            plt.subplot(3,1,2) # Pd
            plt.plot(t, W[:,0])
            plt.ylabel('Dumped solar power [kW]', fontsize=8)
            plt.subplot(3,1,3) # Pwd
            plt.plot(t, P[:,0]-W[:,0])
            plt.ylabel('Net solar power [kW]', fontsize=8)
            plt.xlabel('Time [hours]', fontsize=8)
            plt.suptitle('Time-series of electrical power', fontsize=10)
            
            plt.figure()
            plt.subplot(3,1,1) # Pl
            plt.plot(t, P[:,1])
            plt.ylabel('Electrical load [kW]', fontsize=8)
            plt.subplot(3,1,2) # Pf-Pe
            plt.plot(t, U[:,1]-U[:,0])
            plt.ylabel('Net H2 power [kW]', fontsize=8)
            plt.subplot(3,1,3) # Pg
            plt.plot(t, -W[:,1])
            plt.ylabel('Import from the grid [kW]', fontsize=8)
            plt.xlabel('Time [hours]', fontsize=8)
            plt.suptitle('Time-series of electrical power', fontsize=10)

        #if SP[6] == 1: #Plot time-series of hydrogen flows? yes
            #plt.figure()
            #plt.subplot(3,1,1) #FHl
            #plt.plot(t, P[:,2])
            #plt.ylabel('H2 load [Nm3/h]', fontsize=8)
            #plt.subplot(3,1,2) #FHnp
            #plt.plot(t,W[:,2])
            #plt.ylabel('H2 not supplied [Nm3/h]', fontsize=8)
            #plt.subplot(3,1,3)
            #plt.plot(t,U[:,2]) #FHout
            #plt.ylabel('H2 supplied [Nm3/h]')
            #plt.xlabel('Time [hours]', fontsize=8)
            #plt.suptitle('Hydrogen flows', fontsize=10)

        if SP[7] == 1: #Plot time-series of hydrogen storage level? yes
            plt.figure()
            plt.plot(t,X[:,0]) #VH
            plt.xlabel('Time [hours]', fontsize=8)
            plt.ylabel('hydrogen storage level [Nm3]', fontsize=8)
            plt.suptitle('Hydrogen storage level', fontsize=10)

        if SP[8] == 1: #Plot duration curves of electrical power? yes
            plt.figure()
            plt.subplot(3,1,1)
            plt.plot(-np.sort(-P[:,0])) #Pw
            plt.ylabel('Wind power [kW]', fontsize=8)
            plt.subplot(3,1,2)
            plt.plot(-np.sort(- ( P[:,0] - W[:,0] ) )) #Pw
            plt.ylabel('Net wind power [kW]', fontsize=8)
            plt.subplot(3,1,3)
            plt.plot(-np.sort(-P[:,1])) #Pl
            plt.xlabel('Time [hours]', fontsize=8)
            plt.ylabel('Load [kW]', fontsize=8)
            plt.suptitle('Duration Curves of Electrical Power', fontsize=10)

            plt.figure()
            plt.subplot(3,1,1)
            plt.plot(-np.sort(-U[:,0])) #Pe
            plt.ylabel('ELY power [kW]', fontsize=8)
            plt.subplot(3,1,2)
            plt.plot(-np.sort(-U[:,1])) #Pf
            plt.ylabel('Fuel cell power [kW]', fontsize=8)
            plt.subplot(3,1,3)
            plt.plot(-np.sort(W[:,1])) #Pg
            plt.ylabel('Import from grid [kW]', fontsize=8)
            plt.xlabel('Time [hours]', fontsize=8)
            plt.suptitle('Duration Curves of Electrical Power', fontsize=10)

        if SP[9] == 1: #Plot duration curve of storage level? yes
            plt.figure()
            plt.plot(-np.sort(-X[:,0])) #VH
            plt.xlabel('Time [hours]', fontsize=8)
            plt.ylabel('Hydrogen storage level [Nm3]', fontsize=8)
            plt.suptitle('Duration curve of storage level', fontsize=10)
            

    plt.show()

def results2excel(P, X, U, W, ResArray, sim_type):
    # Write time series results to specified excel file
    # P - time-varying parameters (P_w, P_l, H_l, wspeed)
    # X - state variables (hydrogen storage level VH)
    # U - control variables (Pe, Pf, Hout)
    # W - dependent variables (Pd, Pg, FHnp, Pnp)
    # ResArray - [E_w, E_load, E_ely, E_fc, E_dump, E_g, E_imp, E_exp, E_eg, E_lg, E_w_net, E_fg, E_wfl, E_ns, VH_mean, VH_0, VH_end, VH_load, VH_out, VH_def, VH_eg, VH_e, VH_f]

    ExcelName = "../Results.xlsx"
    if sim_type == 'single':
        # single year simulation
        AllRes = np.concatenate((P, X, U, W), axis=1)

        # Clear all existing timeseries results in excel sheet
        ClearRes = np.zeros((8760,11))
        with pd.ExcelWriter(ExcelName, engine='openpyxl', mode='w') as writer:
            # write the main results to a sheet named "Main results"
            pd.DataFrame(ResArray).to_excel(writer, sheet_name='Main results', startrow=5, startcol=1, header=False, index=False)

            #write energy balance results
            pd.DataFrame(ClearRes).to_excel(writer, sheet_name='Simulation results', startrow=14, startcol=1, header=False, index=False)
            df = pd.DataFrame(AllRes, columns=['P_w', 'P_l', 'H_l', 'wspeed', 'VH', 'Pe', 'Pf', 'Hout', 'Pd', 'Pg', 'FHnp', 'Pnp'])
            df.to_excel(writer, sheet_name='Simulation results', startrow=14, startcol=1, header=True, index=False)

def solarhydsim():
    CP, TS, TS_pnorm, SP, CS, InOutPair = readdata('single')

    print(".. reading input data from excel")
    B = createparam(CP)

    # Load timeseries
    P = whtimeseries(SP, B, TS, TS_pnorm, InOutPair)

    # Run simulator
    print("Starting the simulation.. the number of time steps is " + str(SP[2]))
    X, U, W = simloop(P,SP,B,CS)
    print("Simulation finished!")

    # Plotting data
    plotdata(P, X, U, W, SP)

    # Main results
    ResArray = mainresults(P, X, U, W, SP, B, CS)

    #Write the time series results to Excel
    results2excel(P, X, U, W, ResArray, 'single')

    print('WINDHYDSIM exits. Open the excel file to view results.')


solarhydsim()









