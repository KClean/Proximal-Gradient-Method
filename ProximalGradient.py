import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error
import timeit
# generate proble
# Generate sample data
np.random.seed(23)
n = 300
p = 300

#A = np.linalg.qr(np.random.randn(n, p))[0]
A = np.random.randn(n,p)

xopt = (np.random.randn(p) * np.random.binomial(1, 0.1, (p))).reshape((p,1))
# print(xopt)

# e3 = np.zeros((p,1))
# e3[3] = 1
# e7 = np.zeros((p,1))
# e7[7] = 1
# xopt = e3-e7

# b = A.dot(xopt)
b = A.dot(xopt) 
#+ np.random.randn(n,1)
lamda = 0
spectral_norm = np.linalg.norm(A.T @ A, ord=2)
print("Spectral Norm (2-Norm):", spectral_norm)
STEPSIZE  = 1/spectral_norm
EPSILON = pow(10,-13)
MAX_ITERATION = 4000
x0 = np.random.normal(size=p).reshape((p,1))
x0 = np.random.normal(size=p).reshape((p,1))
S = 0.1
ETA = 0.2
# print("x0 = ", x0)
# print("A = ", A)

# def skilearn_output():
# # Create a Lasso model
#     lasso_model = Lasso(alpha=0.1)  # Adjust the alpha parameter as needed

# # Fit the model to the data
#     lasso_model.fit(A, b)

# # Obtain the optimal coefficients
#     optimal_coefficients = lasso_model.coef_
#     return optimal_coefficients

def check_size(x):
    if not(np.size(x,0)==np.size(A,1) and np.size(A,0) == np.size(b,0) and 
    np.size(x,1)== np.size(b,1) == 1):
        print("Size of problem do not match")
        print(np.size(x,0)==np.size(A,1) , np.size(A,0) == np.size(b,0) , 
    np.size(x,1)== np.size(b,1) == 1)
        print(x.shape,  np.size(b,1))
        exit()

def f(x):
    check_size(x)
    Ax_b = A.dot(x) - b
    return 0.5*(Ax_b.T.dot(Ax_b))

def gradient_f(x):
    check_size(x)
    return A.T.dot(A.dot(x) - b)

def subgradient_F(x):
    return gradient_f(x) + lamda*(np.sign(x))
    # + np.array([np.random.uniform(low=-1, high=1) if xi == 0 else 0 for xi in x]).reshape((p,1))
def g(x):
    return lamda*np.sum(np.abs(x))

def F(x):
    return f(x) + g(x)

def prox(x,stepsize):
    if not(np.size(x,1)==1):
        exit()
    return np.sign(x)*np.maximum(np.zeros(np.shape(x)),np.abs(x)-stepsize*lamda)

def diminishing(stepsize, k):
    if k == 0: return stepsize
    return stepsize / np.sqrt(k)

def constant_steplength(x):
    return S / np.linalg.norm(subgradient_F(x))

def run_SubGradient(Fopt, stepsize_strategy):
    k = 0 # number of iterations
    stepsize = STEPSIZE
    x = x0.copy()
    x_old = x0.copy()
    gradient_mapping = []
    # if diminishing_stepsize == True: stepsize = S
    while(k < MAX_ITERATION):
        # if diminishing_stepsize == True:
        #     stepsize = diminishing(stepsize, k)
        if stepsize_strategy == True: stepsize = constant_steplength(x)
        x = x_old - stepsize * subgradient_F(x_old)
        if np.linalg.norm(x - x_old) < EPSILON: 
            break
        #print(k,abs(F(x) - fopt))
        #gradient_mapping.append((np.sum(F(x))))
        #gradient_mapping.append((np.sum((F(x) + Fopt)/Fopt)))
        gradient_mapping.append(np.log10(np.sum(F(x))))
        #gradient_mapping.append(np.sum(np.abs(F(x) - Fopt)))
        #gradient_mapping.append(np.linalg.norm((1 / stepsize)*(x - x_old), 2))
        
        x_old = x
        k += 1
    print("num steps of  subgradient : ",k)
    return x, gradient_mapping

def backtrack(x,s,eta):
    stepsize = s
    while True:
        prox_grad_x = prox(x - stepsize * gradient_f(x), stepsize)
        if f(prox_grad_x) > f(x) + gradient_f(x).T @ (prox_grad_x - x) + (1/(2*stepsize))*np.linalg.norm(prox_grad_x-x)**2:
            stepsize = eta * stepsize
        else:
            break
    return stepsize



def run_Proximal_Gradient(Fopt,backtracking):
    k = 0 # number of iterations
    stepsize = STEPSIZE
    x = x0.copy()
    x_old = x0.copy()
    gradient_mapping = []
    while(k < MAX_ITERATION):
        if backtracking == True: stepsize = backtrack(x, S, ETA)
        x = prox(x_old - stepsize * gradient_f(x_old), stepsize)
        # print("\n----------------\n","x before gradient ",x_old.T)
        # print("x after gradient ",(x_old - stepsize * gradient_f(x_old)).T)
        # print("x after prox ",x.T)
        #if k % 1 ==0:
        #    print(k,":",x_old,"\n----------\n",x_old - stepsize * gradient_f(x_old) ,"\n----------\n",x)
        #if k == 100:
        #    exit()
        #if k % 200 == 0: print("k =",k," ---- error = ",np.linalg.norm(x - x_old))
        if np.linalg.norm(x - x_old) < EPSILON: 
            break
        #gradient_mapping.append((np.sum(F(x))))
        #gradient_mapping.append((np.sum((F(x) + Fopt)/Fopt)))
        gradient_mapping.append(np.log10(np.sum(F(x))))
        #gradient_mapping.append(np.sum(np.abs(F(x) - Fopt)))
        #gradient_mapping.append(np.linalg.norm((1 / stepsize)*(x - x_old),2))

        x_old = x.copy()
        k += 1
    print("num step of proximal : ",k)
    return x, gradient_mapping

def run_FISTA(Fopt,backtracking):
    k = 0 # number of iterations
    stepsize = STEPSIZE
    x  = x0.copy()
    x_old = x0.copy()
    y = x.copy()
    t_old = 1
    gradient_mapping = []
    while(k < MAX_ITERATION):
        if backtracking == True: stepsize = backtrack(x, S, ETA)
        #print("-------",S,ETA,stepsize)
        x = prox(y - stepsize * gradient_f(y), stepsize)
        t = 0.5 + 0.5*np.sqrt(1+4*t_old*t_old)
        #print("Objective value: ",F(x),"----------", F(y - stepsize * gradient_f(y)))
        #print(k,":",x_old,"\n----------\n",y,"\n----------\n" ,y - stepsize * gradient_f(y),"\n----------\n",x)
        #if k == 50:break
        y = x + ((t_old-1)/(t))*(x - x_old)
        #if k % 1 ==0:
        #   print(k,":",x_old,"\n----------\n",y ,"\n----------\n",x)
        #if k == 100:
        #   exit()
        #if k % 20 == 0:print("x = ",x,"F(x) =",F(x),"Compare to Fopt",Fopt)
        if np.linalg.norm(x - x_old) < EPSILON: 
            break
        #gradient_mapping.append((np.sum(F(x))))
        #gradient_mapping.append((np.sum((F(x) + Fopt)/Fopt)))
        gradient_mapping.append(np.log10(np.sum(F(x))))
        #gradient_mapping.append(np.sum(np.abs(F(x) - Fopt)))
        # gradient_mapping.append(np.linalg.norm((1 / stepsize)*(x - x_old),2))

        x_old = x.copy()
        t_old = t
        k += 1
    print("num step of FISTA : ",k)
    return x, gradient_mapping

def construct_signal(x):
    plt.figure(2)
    plt.clf()        
    plt.subplot(211)    
    plt.plot(xopt)
    plt.title('Original x')
    plt.subplot(212)
    plt.plot(x)
    plt.title('Reconstructed x')
    plt.draw()

def test_model():
    stop = False
    Fopt = F(xopt)
    # print("here Fopt: ",Fopt, f(xopt), g(xopt),xopt)
    while stop == False:
        global lamda, EPSILON, MAX_ITERATION, STEPSIZE, S, ETA
        lamda = float(input("lamda : "))
        #STEPSIZE  = float(input("stepsize * Lf : "))/spectral_norm
        EPSILON = pow(10,-int(input(" - exponent of epsilon : ")))
        MAX_ITERATION = int(input("max iteration: "))
        backtracking = input("invoke backtracking (y/n):")
        linewidth = 2.5
        if backtracking == "y":
            backtracking = True
            S = float(input("S = "))
            ETA = float(input("Eta = "))
        else:
            backtracking == False
        Fopt = F(xopt)
        x0, ls0 = run_SubGradient(Fopt, False)
        x1, ls1 = run_Proximal_Gradient(Fopt, False)
        x2, ls2 = run_FISTA(Fopt, False)
        plt.plot(ls0, label = "SubGradient", linewidth=linewidth)
        plt.plot(ls1, label="ISTA", linewidth=linewidth)
        plt.plot(ls2, label = "FISTA", linewidth=linewidth)
        # Add a grid
        plt.grid(True)
        plt.xlabel('number of iterations')
        plt.ylabel('log(F(x))')
        plt.title(f"lambda = {lamda}", color = "red" )

        #plt.text(f"lambda = {lamda}" , verticalalignment='bottom', horizontalalignment='right')
        plt.legend(loc='best')
        plt.show()
        # print("-----------------------------------")
        # print(x2)
        # print("-----------------------------------")
        # print(x1)
        construct_signal(x1)
        plt.show()
        stop = bool(int(input("Stop or not (1 to stop - 0 to continue) : ")))

def test_model_compare_stepsizes():
    stop = False
    Fopt = F(xopt)
    # print("here Fopt: ",Fopt, f(xopt), g(xopt),xopt)
    while stop == False:
        global lamda, EPSILON, MAX_ITERATION, STEPSIZE, S, ETA
        lamda = float(input("lamda : "))
        #STEPSIZE  = float(input("stepsize * Lf : "))/spectral_norm
        EPSILON = pow(10,-int(input(" - exponent of epsilon : ")))
        MAX_ITERATION = int(input("max iteration: "))
        backtracking = input("invoke backtracking (y/n):")
        linewidth = 2.5
        if backtracking == "y":
            backtracking = True
            S = float(input("S = "))
            ETA = float(input("Eta = "))
        else:
            backtracking == False
        Fopt = F(xopt)
        #x0, ls0 = run_SubGradient(Fopt, False)
        x1, ls1 = run_Proximal_Gradient(Fopt, False)
        x2, ls2 = run_FISTA(Fopt, False)
        #plt.plot(ls0, label = "SubGradient", linewidth=linewidth)
        #plt.gca().lines[0].remove()
        plt.plot(ls1, label="ISTA Constant", linewidth=linewidth)
        plt.plot(ls2, label = "FISTA Constant", linewidth=linewidth)


#        x0, ls0 = run_SubGradient(Fopt, backtracking)
        x1, ls1 = run_Proximal_Gradient(Fopt, backtracking)
        x2, ls2 = run_FISTA(Fopt, backtracking)
#        plt.plot(ls0, label = "SubGradient", linewidth=linewidth)
#        plt.gca().lines[0].remove()
        plt.plot(ls1, label="ISTA Backtracking", linewidth=linewidth, color = "yellow")
        plt.plot(ls2, label = "FISTA Backtracking", linewidth=linewidth, color = "lightgreen")
        # Add a grid
        plt.grid(True)
        plt.xlabel('number of iterations')
        plt.ylabel('log(F(x))')
        plt.title(f"lambda = {lamda}", color = "red" )

        #plt.text(f"lambda = {lamda}" , verticalalignment='bottom', horizontalalignment='right')
        plt.legend(loc='best')
        plt.show()
        # print("-----------------------------------")
        # print(x2)
        # print("-----------------------------------")
        # print(x1)
        construct_signal(x1)
        plt.show()
        elapsed_time_ista_constant = timeit.timeit(lambda: run_Proximal_Gradient(Fopt, False), number=1)
        elapsed_time_fista_constant = timeit.timeit(lambda: run_FISTA(Fopt, False), number=1)
        elapsed_time_ista_backtrack = timeit.timeit(lambda: run_Proximal_Gradient(Fopt, backtracking), number=1)
        elapsed_time_fista_backtrack = timeit.timeit(lambda: run_FISTA(Fopt, backtracking), number=1)
        print(elapsed_time_ista_constant)
        print(elapsed_time_fista_constant)
        print(elapsed_time_ista_backtrack)
        print(elapsed_time_fista_backtrack)

        stop = bool(int(input("Stop or not (1 to stop - 0 to continue) : ")))

def main():
    test_model_compare_stepsizes()

if __name__ == "__main__":
    main()