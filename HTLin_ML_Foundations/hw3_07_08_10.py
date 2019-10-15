# -*- coding: utf-8 -*-

import math
import numpy as np

def err(u, v):
    return math.exp(u)+math.exp(2*v)+math.exp(u*v)+u*u-2*u*v+2*v*v-3*u-2*v

def grad1(u, v):
    grad_u = math.exp(u)+v*math.exp(u*v)+2*u-2*v-3
    grad_v = 2*math.exp(2*v)+u*math.exp(u*v)-2*u+4*v-2
    return grad_u, grad_v

def grad2(u, v):
    grad_u2 = math.exp(u)+v*v*math.exp(u*v)+2
    grad_uv = math.exp(u*v)+u*v*math.exp(u*v)-2
    grad_v2 = 4*math.exp(2*v)+u*u*math.exp(u*v)+4
    return grad_u2, grad_uv, grad_v2

def newton_dir(u, v):
    grad_u1, grad_v1 = grad1(u, v)
    grad_u2, grad_uv, grad_v2 = grad2(u, v)
    grad2_e = np.ones((2, 2))
    grad2_e[0][0] = grad_u2
    grad2_e[0][1] = grad_uv
    grad2_e[1][0] = grad_uv
    grad2_e[1][1] = grad_v2
    inv_grad2e = np.linalg.inv(grad2_e)
    grad_e = np.ones((2, 1))
    grad_e[0] = grad_u1
    grad_e[1] = grad_v1
    nt_dir = -np.matmul(inv_grad2e, grad_e)
    return nt_dir[0], nt_dir[1], grad_u1, grad_v1, grad_u2, grad_uv, grad_v2

def hat_e2(u, v):
    
    return buu*(delta_u**2)+bvv*(delta_v**2)+buv*(delta_u*delta_v)+bu*delta_u+bv*delta_v+b

def main():
    u = 0
    v = 0

    # Q7
#    eta = 0.01
#    num_iter = 5
#    for ii in range(num_iter):
#        grad_u, grad_v = grad1(u, v)
#        u = u - eta*grad_u
#        v = v - eta*grad_v
#        print("[{}] (u, v) = ({}, {})".format(ii+1, u, v))
#    e_uv = err(u, v)
#    print("E = {}".format(e_uv))

    # Q8
#    grad_u1, grad_v1 = grad1(u, v)
#    grad_u2, grad_uv, grad_v2 = grad2(u, v)
#    buu = grad_u2/math.factorial(2)
#    bvv = grad_v2/math.factorial(2)
#    buv = grad_uv/(math.factorial(1)*math.factorial(1))
#    bu = grad_u1/math.factorial(1)
#    bv = grad_v1/math.factorial(1)
#    b = err(0, 0)
#    print("(buu, bvv, buv, bu, bv, b) = ({}, {}, {}, {}, {}, {})".format(buu, bvv, buv, bu, bv, b))

    # Q10
    num_iter = 5
    for ii in range(num_iter):
        delta_u, delta_v, grad_u1, grad_v1, grad_u2, grad_uv, grad_v2 = newton_dir(u, v)
        u += delta_u
        v += delta_v
    buu = grad_u2/math.factorial(2)
    bvv = grad_v2/math.factorial(2)
    buv = grad_uv/(math.factorial(1)*math.factorial(1))
    bu = grad_u1/math.factorial(1)
    bv = grad_v1/math.factorial(1)
    b = err(u, v)/math.factorial(0)
    hat_e2 = buu*(delta_u**2)+bvv*(delta_v**2)+buv*(delta_u*delta_v)+bu*delta_u+bv*delta_v+b
    print(hat_e2)
    return 

if __name__ == "__main__":
    main()