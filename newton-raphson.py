import numpy as np 
## If gives error that numpy is not installed, use the command 
## pip install numpy

## Give code running instructions

## Add commenting

## Look for other examples in the book if possible

## Given variables used for the toy example
Y = [[20-50j,-10+20j,-10+30j],
     [-10+20j,26-52j,-16+32j],
     [-10+30j,-16+32j,26-62j]]
V = [1.05,1,1.04]
theta = np.angle(Y) ## theta is the angle for each entry in Y matrix
delta = np.zeros(3) ## initially delta is all zero

def newton_raphson(Y,V,theta,delta,pv_index,p_sched,q_sched,p_curr,q_curr,delta_curr,v_curr):
    N = len(V) ## No. buses

    # angle matrix holds theta_ij - delta_i + delta_j
    angle = np.zeros([N,N])
    for i in range(len(angle[0])):
        for j in range(len(angle[1])):
            angle[i,j] = theta[i,j] - delta[i] + delta[j]

    ## np.absolute(Y) is the magnitude for each entry in Y matrix
            
    ## Y_cos = |Y|cos(theta)
    Y_cos = np.absolute(Y)*np.cos(angle)

    ## Y_sin = |Y|sin(theta)
    Y_sin = np.absolute(Y)*np.sin(angle)

    #### The following block calculates P and Q

    P = np.zeros(N)
    Q = np.zeros(N)

    for i in range(N):
        for j in range(N):
            P[i] += np.abs(V[i]) * np.abs(V[j]) * Y_cos[i][j] ## P = |V_i||V_j||Y_ij|cos(theta)
            Q[i] -= np.abs(V[i]) * np.abs(V[j]) * Y_sin[i][j] ## Q = -|V_i||V_j||Y_ij|sin(theta)

    ####

    print("P,Q")
    print(P)
    print(Q)

    ## We start building the jacobian matrix by building it in four pieces
    ## Jacobian matrix is built by merging the submatrices J1,J2,J3,J4
    ## Each of J1,J2,J3,J4 are built using the equatons in the Power System Analysis
    ## textbook, splitting each matrix into diagonal and off-diagonal elements

    ## J1 is the top left submatirx
    J1 = np.zeros([N,N])
    ## Diagonal
    for i in range(len(J1[0])):
        val = 0
        for j in range(len(J1[1])):
            if i != j:
                val += np.absolute(V[j])*Y_sin[i,j]
        J1[i,i] = np.absolute(V[i])*val
    ## Off diagonal
    for i in range(len(J1[0])):
        for j in range(len(J1[1])):
            if i != j:
                J1[i,j] = -1*np.absolute(V[i]*V[j])*Y_sin[i,j]


    ## J2 is the topright submatrix
    J2 = np.zeros([N,N])
    ## Diagonal
    for i in range(len(J1[0])):
        val = 0
        for j in range(len(J1[1])):
            if i != j:
                val += np.absolute(V[j])*Y_cos[i,j]
        J2[i,i] = 2*np.absolute(V[i]*Y[i][i])*np.cos(theta[i][i]) + val
    ## Off diagonal
    for i in range(len(J1[0])):
        for j in range(len(J1[1])):
            if i != j:
                J2[i,j] = np.absolute(V[i])*Y_cos[i,j]

    ## J3 is the bottom-left submatrix
    J3 = np.zeros([N,N])
    ## Diagonal
    for i in range(len(J1[0])):
        val = 0
        for j in range(len(J1[1])):
            if i != j:
                val += np.absolute(V[j])*Y_cos[i,j]
        J3[i,i] = np.absolute(V[i])*val
    ## Off diagonal
    for i in range(len(J1[0])):
        for j in range(len(J1[1])):
            if i != j:
                J3[i,j] = -1*np.absolute(V[i]*V[j])*Y_cos[i,j]

    ## J4 is the bottom-right submatrix
    J4 = np.zeros([N,N])
    ## Diagonal
    for i in range(len(J1[0])):
        val = 0
        for j in range(len(J1[1])):
            if i != j:
                val += np.absolute(V[j])*Y_sin[i,j]
        J4[i,i] = -2*np.absolute(V[i]*Y[i][i])*np.sin(theta[i][i]) - val
    ## Off diagonal
    for i in range(len(J1[0])):
        for j in range(len(J1[1])):
            if i != j:
                J4[i,j] = -1*np.absolute(V[i])*Y_sin[i,j]

    jacobian = np.block([[J1,J2],[J3,J4]]) ## merge the four submatrices to get J

    ## obtain the indices of the rows and columns which need to be removed from the jacobian
    pv_indexes_to_remove = []
    for idx in pv_index:
        pv_indexes_to_remove.append(N+idx)
    
    PQ = np.concatenate([P,Q],axis=0) ## PQ is the vector P and Q stacked vertically
    PQ = np.delete(PQ,[0, N] + pv_indexes_to_remove) ## Get rid of necessary PQ vector indices

    print("prejacobian")
    print(jacobian)
    jacobian = np.delete(jacobian, list(set([0, N] + pv_indexes_to_remove)), axis=1) ## column deletion
    jacobian = np.delete(jacobian, list(set([0, N] + pv_indexes_to_remove)), axis=0) ## row deletion
    print("jacobian")
    print(jacobian)
    p_res = np.array(p_sched) - np.array(p_curr) ## obtain P_res
    q_res = np.array(q_sched) - np.array(q_curr) ## obtain Q_res
    print("residuals",np.concatenate([p_res,q_res],axis=0)) ## stack together P_res and Q_res vertically
    x = np.linalg.solve(jacobian,np.concatenate([p_res,q_res],axis=0)) ## x = (Jacobian)^(-1) * [p_res,q_res] vertically stacked

    ## Printing these values while running to verify their values
    print('x :',x)
    print('delta_curr :',delta_curr)
    print('v_curr :',v_curr)

    ##
    updated_values = x + np.concatenate([delta_curr,v_curr],axis=0) ## New values of P and Q 

    ## The updated values consist of delta's and V's
    ## All buses apart from slack bus (index 0) need their delta's changed
    ## All buses apart from slack bus (index 0) and with indices in PV buses need their V's changed

    updated_deltas = updated_values[:-len(pv_index)]
    updated_voltages = updated_values[-len(pv_index):]
    delta[1:] = updated_deltas

    ## Updating V indices and delta indices which need to be updated
    v_index = 0
    pv_index_set = set(idx for idx in pv_index)
    for i in range(len(V)):
        if i != 0 and i not in pv_index_set:
            V[i] = updated_voltages[v_index]
            v_index += 1
    print("updated delta: ",delta)
    # print(updated_voltages)
    print("updated V: ",V)

    # holds theta_ij - delta_i + delta_j
    angle = np.zeros([N,N])
    for i in range(len(angle[0])):
        for j in range(len(angle[1])):
            angle[i,j] = theta[i,j] - delta[i] + delta[j]

    
    ## Recalculating all values which were calculated earlier but with the updated values
    Y_cos = np.absolute(Y)*np.cos(angle)

    Y_sin = np.absolute(Y)*np.sin(angle)

    #### The following block calculates P and Q

    P = np.zeros(N)
    Q = np.zeros(N)

    for i in range(N):
        for j in range(N):
            P[i] += np.abs(V[i]) * np.abs(V[j]) * Y_cos[i][j] ## P = |V_i||V_j||Y_ij|cos(theta)
            Q[i] -= np.abs(V[i]) * np.abs(V[j]) * Y_sin[i][j] ## Q = -|V_i||V_j||Y_ij|sin(theta)

    ####

    print("P,Q")
    print(P)
    print(Q)
    return V,P,Q,delta

def converged(list1,list2,tolerance):
    ## Function to check whether two vectors have converged in value by checkig both
    ## vectors element-wise

    ## len(list1) has to be len(list2)
    for i in range(len(list1)):
        if abs(list1[i] - list2[i]) > tolerance:
            return False
    return True


def newton_method(Y,V,pv_index,p_sched,q_sched,p_initial,q_initial,tolerance):
    '''
    Full function which gives the initial conditions for the system, and returns the optimized system
    '''
    theta = np.angle(Y)
    delta = np.zeros(3)
    active_p = [1,2]
    active_q = [1]
    p_curr = [-1.14,0.5616]
    q_curr = [-2.28]
    delta_curr = [0,0]
    v_curr = [1]
    V,P,Q,delta = newton_raphson(Y,V,theta,delta,pv_index,p_sched,q_sched,p_initial,q_initial,delta_curr,v_curr)
    p_curr = P[1:]
    q_curr = [Q[1]]
    delta_curr = delta[1:]
    v_curr = [V[1]]
    i = 0
    while not (converged(p_sched,p_curr,tolerance) and converged(q_sched,q_curr,tolerance)):
        i += 1
        print('iteration: ',i)
        V,P,Q,delta = newton_raphson(Y,V,theta,delta,pv_index,p_sched,q_sched,p_curr,q_curr,delta_curr,v_curr)
        p_curr = P[1:]
        q_curr = [Q[1]]
        delta_curr = delta[1:]
        v_curr = [V[1]]
        print('p_curr: ',p_curr)
        print('q_curr: ',q_curr)
    print(P)
    print(Q)

newton_method(Y,V,pv_index=[2],p_sched=[-4,2],q_sched=[-2.5],p_initial=[-1.14,0.5616],q_initial=[-2.28],tolerance=0.0001)
Y = [[2-20j,-1+10j,0,-1+10j,0],
     [-1+10j,3-30j,-1+10j,-1+10j,0],
     [0,-1+10j,2-20j,0,-1+10j],
     [-1+10j,-1+10j,0,3-30j,-1+10j],
     [0,0,-1+10j,-1+10j,2-20j]]
V = [1,1,1,1,1]

# newton_method(Y,V,pv_index=[0,1],p_sched=[0.883,0.0076,-1.7137,-1.7355],q_sched=[-(1-0.5983j),-(0.8-0.5496j)],p_initial=[1,0.1,0,0],q_initial=[-1+1j,-1],tolerance=0.0001)

## Uncomment the code below to run it and verify the calculations

# newton_raphson(Y,V,theta,delta,[2],p_sched=[-4,2],q_sched=[-2.5],p_curr=[-1.14,0.5616],q_curr=[-2.28],
#                delta_curr=[0,0],v_curr=[1])
# newton_raphson(Y,[1.05,0.9734513274336283,1.04],theta,
#                delta=[0,-0.0452628,-0.00771829],pv_index=[2],
#                p_sched=[-4,2],q_sched=[-2.5],p_curr=[-3.90078211,1.97828507],q_curr=[-2.44908602],
#                delta_curr=[-0.0452628,-0.00771829],v_curr=[0.97345133])
# newton_raphson(Y,[1.05,0.9716841511285939,1.04],theta,
#                delta=[0,-0.04705814900063786,-0.008703361367510957],pv_index=[2],
#                p_sched=[-4,2],q_sched=[-2.5],p_curr=[-3.99978342,1.99996179],q_curr=[-2.49985682],
#                delta_curr=[-0.04705814900063786,-0.008703361367510957],v_curr=[0.9716841511285939])