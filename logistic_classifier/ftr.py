import numpy as np
from numpy.typing import NDArray
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from scipy.optimize import fmin_l_bfgs_b
from logistic_classifier.code import get_x_tilde, logistic,predict,evaluate_basis_functions
from prettytable import PrettyTable
from logistic_classifier.code import get_confusion_matrix, plot_data_internal,plot_predictive_distribution, compute_average_ll

LAMBDA2 = np.pi/8

def get_hessian(
        w:NDArray,
        X_tilde:NDArray, # Use Phi here if using RBFs
        sigma_0:float
):
    sigmoid_value = predict(X_tilde,w)
    v = sigmoid_value * (1- sigmoid_value) # Hadamard prod
    A =  X_tilde.T @ np.diag(v) @ X_tilde
    A = A + np.identity(A.shape[0]) / sigma_0 ** 2
    return A

def evaluate_objective(
        w:NDArray,
        X_tilde:NDArray,
        y:NDArray,
        sigma_0:float
):
    sigmoid = predict(X_tilde,w)
    return  1/ (2 * sigma_0**2) * np.dot(w,w) - np.dot(y, np.log(sigmoid) ) - np.dot( 1-y, np.log(1-sigmoid))

def get_objective(
        X_tilde:NDArray,
        y:NDArray,
        sigma_0:float
):
    return lambda w : evaluate_objective(w, X_tilde, y, sigma_0)

def evaluate_jacobian(
    w:NDArray,
    X_tilde: NDArray,
    y: NDArray,
    sigma_0: float
):
    sigmoid_value = predict(X_tilde,w)
    return w / sigma_0 ** 2 - (y - sigmoid_value).T @ X_tilde

def get_jacobian(
        X_tilde: NDArray,
        y: NDArray,
        sigma_0: float
):
    return lambda w : evaluate_jacobian(w,X_tilde,y,sigma_0)

def optimize(
        X_tilde: NDArray,
        y: NDArray,
): # L-BFGS-B algo to compute the MAP estimator for w  && Start optimization at origin
    J = get_objective(X_tilde,y,sigma_0)
    dJ = get_jacobian(X_tilde,y,sigma_0)
    w_hat, J_min, d = fmin_l_bfgs_b( func = J, x0 =np.random.randn(X_tilde.shape[1]), fprime = dJ)
    #table = PrettyTable()
    #table.title="Optimization"
    #table.field_names=["w_hat","J_min","warnflag", "grad", "nit"]
    #table.add_row([w_hat,J_min,d["warnflag"], d["grad"], d["nit"]])
    #print(table)
    return w_hat, J, dJ

def predict_laplace(
        x_tilde:NDArray,
        w_hat:NDArray,
        A_inverse: NDArray
):
    mu_a = np.dot(x_tilde,w_hat) #compute mu
    var_a = np.zeros_like(mu_a)
    for i in range(mu_a.size):
       var_a[i] =  np.dot(x_tilde[i,:], A_inverse @ x_tilde[i,:])
    return logistic(mu_a * np.power(1 + LAMBDA2 * var_a,-0.5))

def get_predictive_distribution(w_hat: NDArray, A_inverse: NDArray):
    return lambda x: predict_laplace(get_x_tilde(x),w_hat,A_inverse)

def model_evidence( J:float, M:int, logdetA:float, sigma_0:float):
    return - J - 0.5 * logdetA - M * np.log(sigma_0)

def ll_map(w, X_tilde_train, y_train, X_tilde_test, y_test):
    # Setup table
    table = PrettyTable()
    table.title="train/test log likelihood"
    table.field_names=[ "train loss", "test loss"]
    ll_train = compute_average_ll(X_tilde_train, y_train, w)
    ll_test  = compute_average_ll(X_tilde_test,   y_test, w)
    table.add_row([ll_train,ll_test])
    print(table)
    return ll_train, ll_test

def ll_laplace(w, A_inverse, X_tilde_train, y_train, X_tilde_test, y_test):
    # Setup table
    table = PrettyTable()
    table.title = "train/test log likelihood"
    table.field_names = ["train loss", "test loss"]
    output_prob_train= predict_laplace(X_tilde_train,w,A_inverse)
    ll_train = np.mean(y_train * np.log(output_prob_train) + (1 - y_train) * np.log(1.0 - output_prob_train))
    output_prob_test= predict_laplace(X_tilde_test,w,A_inverse)
    ll_test = np.mean(y_test * np.log(output_prob_test) + (1 - y_test) * np.log(1.0 - output_prob_test))
    table.add_row([ll_train,ll_test])
    print(table)
    return ll_train, ll_test

def plot_predictive_distribution_laplace(X, y, w, A_inverse, map_inputs = lambda x : x):
    xx, yy = plot_data_internal(X, y)
    ax = plt.gca()
    X_tilde = get_x_tilde(map_inputs(np.concatenate((xx.ravel().reshape((-1, 1)), yy.ravel().reshape((-1, 1))), 1)))
    Z = predict_laplace(X_tilde, w, A_inverse)
    Z = Z.reshape(xx.shape)
    cs2 = ax.contour(xx, yy, Z, cmap = 'RdBu', linewidths = 2)
    plt.clabel(cs2, fmt = '%2.1f', colors = 'k', fontsize = 14)
    plt.imshow(Z,interpolation="bilinear", origin="lower", cmap="RdBu", extent=(np.amin(xx), np.amax(xx), np.amin(yy), np.amax(yy)), zorder=0)
    plt.show()

def get_confusion_matrix_laplace(X,y,w,A_inverse, tau=0.5):
    X_tilde = get_x_tilde(X)
    pred_soft = predict_laplace(X_tilde,w, A_inverse)
    y_hat = (pred_soft>tau)
    TP = np.count_nonzero(y_hat[y==1])
    FN = y_hat[y==1].shape[0] - TP
    FP = np.count_nonzero(y_hat[y == 0])
    TN = y_hat[y == 0].shape[0] - FP

    return np.array(
        [[TN/(TN+FP),FP/(TN+FP)],
         [FN/(TP+FN),TP/(TP+FN)]]
    )

X = np.loadtxt('X.txt')
y = np.loadtxt('y.txt')

# We randomly permute the data
permutation = np.random.permutation(X.shape[ 0 ])
X = X[ permutation, : ]
y = y[ permutation ]
n_train = 800
X_train = X[ 0 : n_train, : ]
X_test = X[ n_train :, : ]
y_train = y[ 0 : n_train ]
y_test = y[ n_train : ]


def run(sigma_0:float, l:float):
    X_tilde_train = get_x_tilde(evaluate_basis_functions(l, X_train, X_train))
    X_tilde_test = get_x_tilde(evaluate_basis_functions(l, X_test, X_train))
    print(f"________________Variance = {round(sigma_0 ** 2, 2)} and l={round(l, 2)} __________________")
    w_hat, J, dJ = optimize(X_tilde_train, y_train)
    A = get_hessian(w_hat, X_tilde_train,sigma_0)
    A_inv = np.linalg.inv(A)
    logabsdet = np.linalg.slogdet(A)[1]
    # q = get_predictive_distribution(w_hat,A_inv)

    #Bayesian Inference
    ll_train, ll_test = ll_laplace( w_hat, A_inv,X_tilde_train, y_train, X_tilde_test, y_test)
    plot_predictive_distribution_laplace( X, y, w_hat, A_inv,lambda x : evaluate_basis_functions(l, x, X_train) )
    print("________ Laplace Confussion Matrix ________")
    mtx = get_confusion_matrix_laplace(evaluate_basis_functions(l, X_test, X_train),y_test,w_hat,A_inv)
    print(f"Confussion matrix: {mtx}")

    # MAP Inference
    plot_predictive_distribution(X, y, w_hat, lambda x : evaluate_basis_functions(l, x, X_train))
    ll_train, ll_test = ll_map(w_hat, X_tilde_train, y_train, X_tilde_test, y_test)
    print("________ MAP Confussion Matrix ________")
    mtx = get_confusion_matrix(evaluate_basis_functions(l, X_test, X_train),y_test,w_hat)
    print(f"Confussion matrix: {mtx}")
    return model_evidence(J(w_hat), w_hat.size, logabsdet, sigma_0)

if __name__=="__main__":
    start_s = 0.86
    end_s   = 0.86
    start_l = 0.51
    end_l = 0.51
    S0 = np.sqrt(np.flipud(np.linspace(start_s, end_s,num=1)))
    L = np.linspace(start_l, end_l,num=1)
    grid = np.zeros([S0.size,L.size])
    for i,sigma_0 in enumerate(S0):
        for j, l in enumerate(L):
            grid[i,j]= run(sigma_0, l)

    #Plot heatmap
    #plt.figure(figsize=(8, 8))
    #plt.imshow(grid, cmap='viridis', interpolation='catrom')
    #plt.colorbar(label='Log Evidence')
    #plt.xlabel('l ')
    #plt.ylabel('Prior Variance')
    #plt.xticks(np.arange(len(L)), np.round(L, 2))
    #plt.yticks(np.arange(len(S0)), np.round(np.flipud(np.linspace(start_s, end_s,num=10)), 2))
    #plt.title('Grid Search 3')
    
    #max_index = np.unravel_index(np.argmax(grid), grid.shape)
    #max_value = grid[max_index]
    #plt.scatter(max_index[1], max_index[0], color='red', label=f'Maximum: {max_value}', zorder=5)
    #plt.text(max_index[1], max_index[0], f'  L={L[max_index[1]]:.2f}, S0={S0[max_index[0]]:.2f}', ha='left', va='bottom', color='black')
    #plt.show()
