import numpy as np 

## Given variables
Y = [[20-50j,-10+20j,-10+30j],
     [-10+20j,26-52j,-16+32j],
     [-10+30j,-16+32j,26-62j]]
V = [1.05,1,1.04]
theta = np.angle(Y)
delta = np.zeros(3)

## Need to be calculated

def newton_raphson(Y,V,theta,delta,pv_index,p_sched,q_sched,p_curr,q_curr,delta_curr,v_curr):
    N = 3
    # holds theta_ij - delta_i + delta_j
    angle = np.zeros([3,3])
    for i in range(len(angle[0])):
        for j in range(len(angle[1])):
            angle[i,j] = theta[i,j] - delta[i] + delta[j]
    # print('angle1',angle)

    # print(np.absolute(Y))

    Y_cos = np.absolute(Y)*np.cos(angle)
    # print('cosangle',np.cos(angle))
    # print(np.absolute(Y))
    VtY_cos = np.matmul(np.transpose(np.absolute(V)),np.transpose(Y_cos))
    # P = np.absolute(V)*VtY_cos Look into this one
    # print('vty_cos1',VtY_cos)

    Y_sin = np.absolute(Y)*np.sin(angle)
    VtY_sin = np.matmul(np.transpose(np.absolute(V)),Y_sin)
    # Q = -1*np.absolute(V)*VtY_sin Look into this one

    sum_VtY_cos = sum(VtY_cos)
    P = np.absolute(V)*sum_VtY_cos

    sum_VtY_sin = sum(VtY_sin)
    Q = -1*np.absolute(V)*sum_VtY_sin

    ####

    P = np.zeros(3)
    Q = np.zeros(3)

    for i in range(3):
        for j in range(3):
            P[i] += np.abs(V[i]) * np.abs(V[j]) * Y_cos[i][j]
            Q[i] -= np.abs(V[i]) * np.abs(V[j]) * Y_sin[i][j]

    ####

    print("P,Q")
    print(P)
    print(Q)

    J1 = np.zeros([3,3])
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


    J2 = np.zeros([3,3])
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

    J3 = np.zeros([3,3])
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

    J4 = np.zeros([3,3])
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

    jacobian = np.block([[J1,J2],[J3,J4]])
    # print(jacobian.shape)
    # print(jacobian)
    pv_indexes_to_remove = []
    for idx in pv_index:
        pv_indexes_to_remove.append(N+idx)
    
    PQ = np.concatenate([P,Q],axis=0)
    # print(PQ)
    PQ = np.delete(PQ,[0, N] + pv_indexes_to_remove)
    # print(PQ)

    jacobian = np.delete(jacobian, [0, N] + pv_indexes_to_remove, axis=1) ## column
    jacobian = np.delete(jacobian, [0, N] + pv_indexes_to_remove, axis=0) ## row
    # print(jacobian.shape)
    print("jacobian")
    print(jacobian)
    p_res = np.array(p_sched) - np.array(p_curr)
    q_res = np.array(q_sched) - np.array(q_curr)
    print("residuals",np.concatenate([p_res,q_res],axis=0))
    x = np.linalg.solve(jacobian,np.concatenate([p_res,q_res],axis=0))
    updated_values = x + np.concatenate([delta_curr,v_curr],axis=0)
    # print(updated_values)
    ## update deltas and Vs and then update p_curr and q_curr
    # return updated_values

    ## The updated values consist of delta's and V's
    ## All buses apart from slack bus (index 0) need their delta's changed
    ## All buses apart from slack bus (index 0) and with indices in PV buses need their V's changed

    updated_deltas = updated_values[:-len(pv_index)]
    updated_voltages = updated_values[-len(pv_index):]
    delta[1:] = updated_deltas

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
    angle = np.zeros([3,3])
    for i in range(len(angle[0])):
        for j in range(len(angle[1])):
            angle[i,j] = theta[i,j] - delta[i] + delta[j]
    # print('angle2',angle)

    # print(np.absolute(Y))

    Y_cos = np.absolute(Y)*np.cos(angle)
    # print(Y_cos)
    VtY_cos = np.matmul(np.transpose(np.absolute(V)),Y_cos)
    # P = np.absolute(V)*VtY_cos Look into this one
    # print('vty_cos2',VtY_cos)

    Y_sin = np.absolute(Y)*np.sin(angle)
    VtY_sin = np.matmul(np.transpose(np.absolute(V)),Y_sin)
    # Q = -1*np.absolute(V)*VtY_sin Look into this one

    sum_VtY_cos = sum(VtY_cos)
    P = np.absolute(V)*sum_VtY_cos

    sum_VtY_sin = sum(VtY_sin)
    Q = -1*np.absolute(V)*sum_VtY_sin

    ####

    P = np.zeros(3)
    Q = np.zeros(3)

    for i in range(3):
        for j in range(3):
            P[i] += np.abs(V[i]) * np.abs(V[j]) * Y_cos[i][j]
            Q[i] -= np.abs(V[i]) * np.abs(V[j]) * Y_sin[i][j]

    ####

    print("P,Q")
    print(P)
    print(Q)
    return [
        V,
        P,
        Q,
        delta
    ]

# newton_raphson(Y,V,theta,delta,[2],p_sched=[-4,2],q_sched=[-2.5],p_curr=[-1.14,0.5616],q_curr=[-2.28],delta_curr=[0,0],v_curr=[1])
# newton_raphson(Y,[1.05,0.9734513274336283,1.04],theta,
#                delta=[0,-0.0452628,-0.00771829],pv_index=[2],
#                p_sched=[-4,2],q_sched=[-2.5],p_curr=[-3.90078211,1.97828507],q_curr=[-2.44908602],
#                delta_curr=[-0.0452628,-0.00771829],v_curr=[0.97345133])
newton_raphson(Y,[1.05,0.9716841511285939,1.04],theta,
               delta=[0,-0.04705814900063786,-0.008703361367510957],pv_index=[2],
               p_sched=[-4,2],q_sched=[-2.5],p_curr=[-3.99978342,1.99996179],q_curr=[-2.49985682],
               delta_curr=[-0.04705814900063786,-0.008703361367510957],v_curr=[0.9716841511285939])
## Look at p_curr,q_curr