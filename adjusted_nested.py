# Python Libraries
import numpy as np
import utils as ut
from scipy.optimize import minimize
import pandas as pd


class AdjBaseModel:
    def __init__(self,):
        self.Nitr = 0
        # 0:car 1:pass 2:bus 3:train 4:walk 5:bike
        self.tree = [[0, 5], [1, 2, 3], [4]]       
        #self.tree = [[0], [1], [2], [3], [4], [5]]       

    # Initial Parameters
    def init_parameters(self,):
        np.random.seed(seed=123)
        k2, k3, k4, k5, k6 =  np.random.randn(5, 1).flatten()
        alpha, beta, theta, mu = np.random.randn(1 , 4).flatten()
        util_parms = np.array([k2, k3, k4, k5, k6, alpha, beta, theta, mu])

        # Get only lambda for those nest which has more than 1 ele in the node
        lmbda = np.random.uniform(size=sum(len(i) > 1 for i in self.tree))

        return np.concatenate((util_parms, lmbda))

    # Utility function
    def utility_function(self, alpha, beta, theta, mu, k):
        # Attributes used in the funtion
        cost, time, avail, tw, tg, _  = ut.get_attributes(df, model="base")

        # Utilities 
        V = alpha*cost + beta*time + theta*tw + mu*tg
        # Adding ASC
        V = np.add(V, k)
        # Utilities of mode where there is avialability
        V = np.multiply(V, avail)

        return V

    def adjusted_utility_function(self, V, lmbda):
        # Initialize the adjusted array
        V_adj = np.zeros(V.shape)

        # Itr for each node in the tree
        for i in range(len(self.tree)):
            node = self.tree[i]
            logsum = (lmbda[i] - 1) * np.log(np.nansum(np.exp(V[:, node]/lmbda[i]), axis=1) + + 1e-28) # This is just to avoid having zero in log
            # Itr for each mode in the node of tree
            for j in node:
                V_adj[:, j] = V[:, j]/lmbda[i] + logsum 
        return V_adj

    # Probability
    def probability_function(self, V):
        # Exponential of all utilities
        V_exp = np.exp(V)
        assert np.isinf(V_exp).any() == False, "There is a INF in the Exponential for Utilities"

        # Sum of Exponential Utilties for each ID
        V_exp_sum = np.nansum(V_exp, axis=1).reshape(-1, 1)

        # Probability of each mode for each ID
        Pr = np.divide(V_exp, V_exp_sum)
        assert np.equal(Pr, 0.0).any() == False, "Some of probability are calculated as Zero"
        assert np.isinf(Pr).any() == False, "Some of probability are INF"
        
        return Pr

    # Loglikelihood
    def loglikelihood(self, Pr):
        # Get probability of choosen mode
        _, _, _, _, _, mode_choosen  = ut.get_attributes(df, model="base")
        pr_of_choosen_mode = [Pr[i, mode_choosen[i]] for i in range(len(Pr))]

        # Log Likelihood
        ll = np.log(pr_of_choosen_mode)
        assert np.isinf(ll).any() == False, "There is a Log value that is INF"
        #assert np.equal(ll, 0.0).any() == False, "There is a Log value that is Zero"
        ll = -np.sum(ll)

        return ll

    def optimization_function(self, parameters):
        # Parameters
        k2, k3, k4, k5, k6, alpha, beta, theta, mu = parameters[:9]
        k = np.array([0.0, k2, k3, k4, k5, k6]).reshape(1, 6)
        lmbda = parameters[9:]

        # Now we want to have the consistant lamda as the tree. So we will push 1 in place where node is just one.
        if (np.array([len(node) for node in self.tree]) == 1).any():
            locs = np.where(np.array([len(node) for node in self.tree]) == 1)[0]
            for loc in locs:
                lmbda = np.insert(lmbda, loc, 1)

        # Utility
        V = self.utility_function(alpha, beta, theta, mu, k)
        V = V/100

        # Adjust Utilities
        V = self.adjusted_utility_function(V, lmbda)

        # Probability
        Pr = self.probability_function(V)
        # Loglikelihood
        ll = self.loglikelihood(Pr)

        # Print Summary
        if self.Nitr % 100 == 0:
                self.print_result(parameters, ll)
        self.Nitr += 1

        return ll

    def print_result(self, params, loglikelihood=None):  
        col = ['k2', 'k3', 'k4', 'k5', 'k6', 'alpha', 'beta', 'theta', 'mu']
        for i in range(sum(len(i) > 1 for i in self.tree)):
                col.append("Lmd_{}".format(i))
        if loglikelihood is not None:
            col.append("Log-Like")

        table = pd.DataFrame(columns=col)
        value = list(params)

        if loglikelihood is not None:
            value.append(loglikelihood)
        table.loc[self.Nitr] = value
        print(table.round(4))

    def optimization_routine(self,):
        print("\n-------- Parameter for Adj Base Model --------")
        result =  minimize(fun=self.optimization_function,
                           x0=self.init_parameters(),
                           method="BFGS",
                           options={"disp":True,
                                    "gtol":1e-03},
                          )
        return result

    def base_result(self, result):
        # Standard Deviation
        sd = np.sqrt(np.diagonal(result.hess_inv))
        # T-stat
        t_stat_util = np.divide(result.x[:9], sd[:9])
        t_stat_lmbd = np.divide(result.x[9:] - 1, sd[9:])

        # Likelihood at parameters at zero
        # Parameters
        k2, k3, k4, k5, k6, alpha, beta, theta, mu = np.zeros((1,9)).flatten()
        k = np.array([0.0, k2, k3, k4, k5, k6]).reshape(1, 6)
        lmbda = np.ones((1,len(self.tree))).flatten()
        V = self.utility_function(alpha, beta, theta, mu, k)
        V = V/100
        V = self.adjusted_utility_function(V, lmbda)
        Pr = self.probability_function(V)
        ll_zero = self.loglikelihood(Pr)

        # Loglikelihood ratio index
        ratio = 1 - (result.fun/ll_zero)

        # DataFrame for presentation
        print("\n Final parameters for ADJ BASE MODEL")
        index = ['k_pass', 'k_bus', 'k_train', 'k_walk', 'k_bike', 'alpha(M)', 'beta(T)', 'theta(TW)', 'mu(TG)']
        for i in range(sum(len(i) > 1 for i in self.tree)):
                index.append("Lmd_{}".format(i))
        col   = ['Optimal value', 'Standard Deviation', 'T-Statistics']
        table = pd.DataFrame(index=index, columns=col)
        table.loc[:, "Optimal value"] = result.x
        table.loc[:, "Standard Deviation"] = sd
        table.loc[:, "T-Statistics"] = np.concatenate((t_stat_util, t_stat_lmbd))
        print(table)
        print("\nOptimal LogLikelihood:\t", result.fun)
        print("Zero LogLikelihood:\t", ll_zero)
        print("Likelihood Ratio:\t", ratio)



if __name__ == "__main__":
    # Reading data
    filename = "data/modeData.csv"
    df = ut.read_csv(filename)
    # Base Model
    model = AdjBaseModel()

    # # Parameters
    # k2, k3, k4, k5, k6, alpha, beta, theta, mu = model.init_parameters()[:9]
    # k = np.array([0.0, k2, k3, k4, k5, k6]).reshape(1, 6)
    # lmbda = model.init_parameters()[9:]
    # print(model.init_parameters())

    # # Now we want to have the consistant lamda as the tree. So we will push 1 in place where node is just one.
    # if (np.array([len(node) for node in model.tree]) == 1).any():
    #     locs = np.where(np.array([len(node) for node in model.tree]) == 1)[0]
    #     for loc in locs:
    #         lmbda = np.insert(lmbda, loc, 1)
    # # Utility
    # V = model.utility_function(alpha, beta, theta, mu, k)
    # V = V/100  
    # # Adjust Utilities
    # V = model.adjusted_utility_function(V, lmbda)
    # # Probability
    # Pr = model.probability_function(V)
    # # Loglikelihood
    # ll = model.loglikelihood(Pr)
    # print(ll)

    # Optimization
    result = model.optimization_routine()
    model.base_result(result)