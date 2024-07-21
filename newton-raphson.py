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
    N = len(V)
    # holds theta_ij - delta_i + delta_j
    angle = np.zeros([N,N])
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

    P = np.zeros(N)
    Q = np.zeros(N)

    for i in range(N):
        for j in range(N):
            P[i] += np.abs(V[i]) * np.abs(V[j]) * Y_cos[i][j]
            Q[i] -= np.abs(V[i]) * np.abs(V[j]) * Y_sin[i][j]

    ####

    print("P,Q")
    print(P)
    print(Q)

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

    # delta_curr = delta[1:]
    # v_curr = []
    # for i in range(N):
    #     if i not in ([0] + pv_index):
    #         v_curr.append(i)

    jacobian = np.delete(jacobian, [0, N] + pv_indexes_to_remove, axis=1) ## column
    jacobian = np.delete(jacobian, [0, N] + pv_indexes_to_remove, axis=0) ## row
    # print(jacobian.shape)
    print("jacobian")
    print(jacobian)
    p_res = np.array(p_sched) - np.array(p_curr)
    q_res = np.array(q_sched) - np.array(q_curr)
    print("residuals",np.concatenate([p_res,q_res],axis=0))
    x = np.linalg.solve(jacobian,np.concatenate([p_res,q_res],axis=0))
    print('x :',x)
    print('delta_curr :',delta_curr)
    print('v_curr :',v_curr)
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
    angle = np.zeros([N,N])
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

    P = np.zeros(N)
    Q = np.zeros(N)

    for i in range(N):
        for j in range(N):
            P[i] += np.abs(V[i]) * np.abs(V[j]) * Y_cos[i][j]
            Q[i] -= np.abs(V[i]) * np.abs(V[j]) * Y_sin[i][j]

    ####

    print("P,Q")
    print(P)
    print(Q)
    return V,P,Q,delta

def converged(list1,list2,tolerance):
    ## len(list1) has to be len(list2)
    for i in range(len(list1)):
        if abs(list1[i] - list2[i]) > tolerance:
            return False
    return True


def newton_method(Y,V,pv_index,p_sched,q_sched,p_initial,q_initial,tolerance):
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
## Look at p_curr,q_curr

## v_curr - remove index 0 and indices in pv_index from V
## delta_curr = delta[1:] as the first bus is the only one whos delta wouldn't be changing

## Need to look at p_curr and q_curr