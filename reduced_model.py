# Python Libraries
import numpy as np
np.random.seed(seed=123)
import utils as ut
from scipy.optimize import minimize
import pandas as pd


class ReducedModel:
    def __init__(self,):
        self.Nitr = 0

    # Initial Parameters
    def init_parameters(self,):
        k2, k3, k4, k5, k6 =  np.random.randn(5, 1).flatten()
        alpha, beta = np.random.randn(1 , 2).flatten()
        return [k2, k3, k4, k5, k6, alpha, beta]

    # Utility function
    def utility_function(self, alpha, beta, k):
        # Attributes used in the funtion
        cost, time, avail, _  = ut.get_attributes(df, model="reduced")

        # Utilities 
        V = alpha*cost + beta*time
        # Adding ASC
        V = np.add(V, k)
        # Utilities of mode where there is avialability
        V = np.multiply(V, avail)

        return V

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
        assert np.equal(Pr, 0.0).any() == False, "Some of probability are INF"
        
        return Pr

    # Loglikelihood
    def loglikelihood(self, Pr):
        # Get probability of choosen mode
        _, _, _, mode_choosen  = ut.get_attributes(df, model="reduced")
        pr_of_choosen_mode = [Pr[i, mode_choosen[i]] for i in range(len(Pr))]

        # Log Likelihood
        ll = np.log(pr_of_choosen_mode)
        assert np.isinf(ll).any() == False, "There is a Log value that is INF"
        assert np.equal(ll, 0.0).any() == False, "There is a Log value that is Zero"
        ll = -np.sum(ll)

        return ll

    def optimization_function(self, parameters):
        # Parameters
        k2, k3, k4, k5, k6, alpha, beta = parameters
        k = np.array([0.0, k2, k3, k4, k5, k6]).reshape(1, 6)
        # Utility
        V = self.utility_function(alpha, beta, k)
        V = V/100
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
        col = ['k2', 'k3', 'k4', 'k5', 'k6', 'alpha', 'beta']
        if loglikelihood is not None:
            col.append("Log-Like")
        table = pd.DataFrame(columns=col)
        value = list(params)
        if loglikelihood is not None:
            value.append(loglikelihood)
        table.loc[self.Nitr] = value
        print(table.round(4))

    def optimization_routine(self,):
        print("\n-------- Parameter for Reduced Model --------")
        result =  minimize(fun=self.optimization_function,
                           x0=self.init_parameters(),
                           method="BFGS",
                           options={"disp":True,
                                    "gtol":1e-03},
                          )
        return result

    def reduced_result(self, result):
        # Standard Deviation
        sd = np.sqrt(np.diagonal(result.hess_inv))
        # T-stat
        t_stat = np.divide(result.x, sd)

        # Likelihood at parameters at zero
        k2, k3, k4, k5, k6, alpha, beta = np.zeros((1,7)).flatten()
        k = np.array([0.0, k2, k3, k4, k5, k6]).reshape(1, 6)
        V = model.utility_function(alpha, beta, k)
        Pr = model.probability_function(V)
        ll_zero = model.loglikelihood(Pr)

        # Loglikelihood ratio index
        ratio = 1 - (result.fun/ll_zero)

        # DataFrame for presentation
        print("\n Final parameters for REDUCED MODEL")
        index = ['k_pass', 'k_bus', 'k_train', 'k_walk', 'k_bike', 'alpha(M)', 'beta(T)']
        col   = ['Optimal value', 'Standard Deviation', 'T-Statistics']
        table = pd.DataFrame(index=index, columns=col)
        table.loc[:, "Optimal value"] = result.x
        table.loc[:, "Standard Deviation"] = sd
        table.loc[:, "T-Statistics"] = t_stat
        print(table)
        print("\nOptimal LogLikelihood:\t", result.fun)
        print("Zero LogLikelihood:\t", ll_zero)
        print("Likelihood Ratio:\t", ratio)



if __name__ == "__main__":
    # Reading data
    filename = "data/modeData.csv"
    df = ut.read_csv(filename)
    # Base Model
    model = ReducedModel()

    # # Init paramerets
    # k2, k3, k4, k5, k6, alpha, beta = model.init_parameters()
    # k = np.array([0.0, k2, k3, k4, k5, k6]).reshape(1, 6)
    # # Utility
    # V = model.utility_function(alpha, beta, k)
    # V = V/100
    # # Probability
    # Pr = model.probability_function(V)
    # # LL
    # print(model.loglikelihood(Pr))

    # Optimization
    result = model.optimization_routine()
    model.reduced_result(result)


    

    
