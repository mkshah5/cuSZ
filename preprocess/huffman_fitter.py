import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
# import statsmodels.api as sm
import scipy.stats as ss

def mse(A,B):
    return (np.square(A-B)).mean(axis=None)


QUANT_FILES = ['5e5', '25e5', '1e5', '75e6', '5e6', '25e6', '1e6']

for quant_file in QUANT_FILES:
    quant_codes_0 = np.fromfile('./quant_codes/hist-qtensor-'+quant_file+'.data', dtype=np.int32)
    # quant_codes_1 = np.fromfile('./quant_codes/hist-qtensor-1e5.data', dtype=np.int32)
    # quant_codes_2 = np.fromfile('./quant_codes/hist-qtensor-5e6.data', dtype=np.int32)
    # quant_codes_3 = np.fromfile('./quant_codes/hist-qtensor-1e6.data', dtype=np.int32)

    quant_codes_copy = quant_codes_0.copy()
    quant_codes_copy[quant_codes_copy == 0] = 1


    expanded_arr = np.zeros(np.sum(quant_codes_copy))

    ind = 0
    idx = 0


    ### Convert histogram to data points (as random variables)
    for num_quant in quant_codes_copy:
        for i in range(ind, ind+num_quant):
            expanded_arr[i] = idx
        idx+=1
        ind+=num_quant

    ### Let numpy and scipy handle histogram creation
    hist = np.histogram(expanded_arr, 1024)
    hist_dist = ss.rv_histogram(hist)

    ### Get some data points and fetch scipy distribution values
    x_test = np.linspace(0,1024,1024)
    test = hist_dist.pdf(x_test)

    results = []
    ### Run fitting, will take a while    
    #list_of_dists = ['alpha','anglit','arcsine','beta','betaprime','bradford','burr','burr12','cauchy','chi','chi2','cosine','dgamma','dweibull','erlang','expon','exponnorm','exponweib','exponpow','f','fatiguelife','fisk','foldcauchy','foldnorm','frechet_r','frechet_l','genlogistic','genpareto','gennorm','genexpon','genextreme','gausshyper','gamma','gengamma','genhalflogistic','gilbrat','gompertz','gumbel_r','gumbel_l','halfcauchy','halflogistic','halfnorm','halfgennorm','hypsecant','invgamma','invgauss','invweibull','johnsonsb','johnsonsu','kstwobign','laplace','levy','levy_l','logistic','loggamma','loglaplace','lognorm','lomax','maxwell','mielke','nakagami','ncx2','ncf','nct','norm','pareto','pearson3','powerlaw','powerlognorm','powernorm','rdist','reciprocal','rayleigh','rice','recipinvgauss','semicircular','t','triang','truncexpon','truncnorm','tukeylambda','uniform','vonmises','vonmises_line','wald','weibull_min','weibull_max']
    # list_of_dists = ['norm']
    list_of_dists = ['norm','chi2','laplace','cauchy','weibull_max', 'exponweib','gamma', 'maxwell','alpha']

    for i in list_of_dists:
        dist = getattr(ss, i)
        param = dist.fit(expanded_arr)
        a = ss.kstest(expanded_arr, i, args=param)
        
        mse_dist = mse(test, dist.pdf(x_test, *param))

        
        results.append((i,a[0],a[1], param, str(mse_dist)))
        del param
        
    outfile = open('mle_results_'+quant_file+'.txt','w')
    results.sort(key=lambda x:float(x[2]), reverse=True)
    for j in results:
        outfile.write("{}: statistic={}, pvalue={}, params={}, MSE={}".format(j[0], j[1], j[2], j[3],j[4]))

    outfile.close()