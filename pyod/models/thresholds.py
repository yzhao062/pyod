def AUCP(**kwargs):
    """AUCP class for Area Under Curve Precentage thresholder.

       Use the area under the curve to evaluate a non-parametric means
       to threshold scores generated by the decision_scores where outliers
       are set to any value beyond where the auc of the kde is less
       than the (mean + abs(mean-median)) percent of the total kde auc.

       Parameters
       ----------

       random_state : int, optional (default=1234)
            Random seed for the random number generators of the thresholders. Can also
            be set to None.
    """

    from pythresh.thresholds.aucp import AUCP as AUCP_thres
    return AUCP_thres(**kwargs)


def BOOT(**kwargs):
    """BOOT class for Bootstrapping thresholder.

       Use a boostrapping based method to find a non-parametric means
       to threshold scores generated by the decision_scores where outliers
       are set to any value beyond the mean of the confidence intervals.
   
       Parameters
       ----------
       
       random_state : int, optional (default=1234)
            Random seed for bootstrapping a confidence interval. Can also be set to None.

    """

    from pythresh.thresholds.boot import BOOT as BOOT_thres
    return BOOT_thres(**kwargs)


def CHAU(**kwargs):
    """CHAU class for Chauvenet's criterion thresholder.

       Use the Chauvenet's criterion to evaluate a non-parametric
       means to threshold scores generated by the decision_scores
       where outliers are set to any value below the Chauvenet's
       criterion.

       Parameters
       ----------

       method : {'mean', 'median', 'gmean'}, optional (default='mean')
            Calculate the area normal to distance using a scaler
       
            - 'mean':  Construct a scaler with the mean of the scores
            - 'median: Construct a scaler with the median of the scores
            - 'gmean': Construct a scaler with the geometric mean of the scores

       random_state : int, optional (default=1234)
            Random seed for the random number generators of the thresholders. Can also
            be set to None.
    """

    from pythresh.thresholds.chau import CHAU as CHAU_thres
    return CHAU_thres(**kwargs)


def CLF(**kwargs):
    """CLF class for Trained Classifier thresholder.

       Use the trained linear classifier to evaluate a non-parametric means
       to threshold scores generated by the decision_scores where outliers
       are set to any value beyond 0.

       Parameters
       ----------

       method : {'simple', 'complex'}, optional (default='complex')
            Type of linear model

            - 'simple':  Uses only the scores
            - 'complex': Uses the scores, log of the scores, and the scores' PDF

       random_state : int, optional (default=1234)
            Random seed for the random number generators of the thresholders. Can also
            be set to None.
    """

    from pythresh.thresholds.clf import CLF as CLF_thres
    return CLF_thres(**kwargs)


def CLUST(**kwargs):
    """CLUST class for clustering type thresholders.

       Use the clustering methods to evaluate a non-parametric means to
       threshold scores generated by the decision_scores where outliers
       are set to any value not labelled as part of the main cluster.

       Parameters
       ----------
        
       method : {'agg', 'birch', 'bang', 'bgm', 'bsas', 'dbscan', 'ema', 'kmeans', 'mbsas', 'mshift', 'optics', 'somsc', 'spec', 'xmeans'}, optional (default='spec')
            Clustering method
        
            - 'agg':    Agglomerative
            - 'birch':  Balanced Iterative Reducing and Clustering using Hierarchies
            - 'bang':   BANG
            - 'bgm':    Bayesian Gaussian Mixture
            - 'bsas':   Basic Sequential Algorithmic Scheme
            - 'dbscan': Density-based spatial clustering of applications with noise
            - 'ema':    Expectation-Maximization clustering algorithm for Gaussian Mixture Model
            - 'kmeans': K-means
            - 'mbsas':  Modified Basic Sequential Algorithmic Scheme
            - 'mshift': Mean shift
            - 'optics': Ordering Points To Identify Clustering Structure
            - 'somsc':  Self-organized feature map 
            - 'spec':   Clustering to a projection of the normalized Laplacian 
            - 'xmeans': X-means
            
       random_state : int, optional (default=1234)
            Random seed for the BayesianGaussianMixture clustering (method='bgm'). Can
            also be set to None.
    """

    from pythresh.thresholds.clust import CLUST as CLUST_thres
    return CLUST_thres(**kwargs)


def CPD(**kwargs):
    """CPD class for Change Point Detection thresholder.

       Use change point detection to find a non-parametric means
       to threshold scores generated by the decision_scores where outliers
       are set to any value beyond the detected change point.

       Parameters
       ----------

       method : {'Dynp', 'KernelCPD', 'Binseg', 'BottomUp'}, optional (default='Dynp')
            Method for change point detection

            - 'Dynp':      Dynamic programming (optimal minimum sum of errors per partition)
            - 'KernelCPD': RBF kernel function (optimal minimum sum of errors per partition)
            - 'Binseg':    Binary segmentation
            - 'BottomUp':  Bottom-up segmentation

       transform : {'cdf', 'kde'}, optional (default='cdf')
            Data transformation method prior to fit

            - 'cdf': Use the cumulative distribution function
            - 'kde': Use the kernel density estimation

       random_state : int, optional (default=1234)
            Random seed for the random number generators of the thresholders. Can also
            be set to None.
            
    """

    from pythresh.thresholds.cpd import CPD as CPD_thres
    return CPD_thres(**kwargs)


def DECOMP(**kwargs):
    """DECOMP class for Decomposition based thresholders.

       Use decomposition to evaluate a non-parametric means
       to threshold scores generated by the decision_scores where outliers
       are set to any value beyond the maximum of the decomposed
       matrix that results from decomposing the cumulative distribution
       function of the decision scores.
       
       Parameters
       ----------

       method : {'NMF', 'PCA', 'GRP', 'SRP'}, optional (default='PCA')
            Method to use for decomposition

            - 'NMF':  Non-Negative Matrix Factorization
            - 'PCA':  Principal Component Analysis
            - 'GRP':  Gaussian Random Projection
            - 'SRP':  Sparse Random Projection
            
       random_state : int, optional (default=1234)
            Random seed for the decomposition algorithm. Can also be set to None.

    """

    from pythresh.thresholds.decomp import DECOMP as DECOMP_thres
    return DECOMP_thres(**kwargs)


def DSN(**kwargs):
    """DSN class for Distance Shift from Normal thresholder.

       Use the distance shift from normal to evaluate a non-parametric means
       to threshold scores generated by the decision_scores where outliers
       are set to any value beyond the distance calculated by the selected
       metric.
       
       Parameters
       ----------

       metric : {'JS', 'WS', 'ENG', 'BHT', 'HLL', 'HI', 'LK', 'LP', 'MAH', 'TMT', 'RES', 'KS', 'INT', 'MMD'}, optional (default='MAH')
            Metric to use for distance computation
        
            - 'JS':  Jensen-Shannon distance
            - 'WS':  Wasserstein or Earth Movers distance
            - 'ENG': Energy distance
            - 'BHT': Bhattacharyya distance
            - 'HLL': Hellinger distance
            - 'HI':  Histogram intersection distance
            - 'LK':  Lukaszyk-Karmowski metric for normal distributions
            - 'LP':  Levy-Prokhorov metric
            - 'MAH': Mahalanobis distance
            - 'TMT': Tanimoto distance
            - 'RES': Studentized residual distance
            - 'KS':  Kolmogorov-Smirnov distance
            - 'INT': Weighted spline interpolated distance
            - 'MMD': Maximum Mean Discrepancy distance
            
       random_state : int, optional (default=1234)
            Random seed for the normal distribution. Can also be set to None.

    """

    from pythresh.thresholds.dsn import DSN as DSN_thres
    return DSN_thres(**kwargs)


def EB(**kwargs):
    """EB class for Elliptical Boundary thresholder.

       Use pseudo-random elliptical boundaries to evaluate a non-parametric means
       to threshold scores generated by the decision_scores where outliers
       are set to any value beyond a pseudo-random elliptical boundary set
       between inliers and outliers.

       Parameters
       ----------

       random_state : int, optional (default=1234)
            Random seed for the random number generators of the thresholders. Can also
            be set to None.
    """

    from pythresh.thresholds.eb import EB as EB_thres
    return EB_thres(**kwargs)


def FGD(**kwargs):
    """FGD class for Fixed Gradient Descent thresholder.

       Use the fixed gradient descent to evaluate a non-parametric means
       to threshold scores generated by the decision_scores where outliers
       are set to any value beyond where the first derivative of the kde
       with respect to the decision scores passes the mean of the first 
       and second inflection points.

       Parameters
       ----------

       random_state : int, optional (default=1234)
            Random seed for the random number generators of the thresholders. Can also
            be set to None.
    """

    from pythresh.thresholds.fgd import FGD as FGD_thres
    return FGD_thres(**kwargs)


def FILTER(**kwargs):
    """FILTER class for Filtering based thresholders.

       Use the filtering based methods to evaluate a non-parametric means
       to threshold scores generated by the decision_scores where outliers
       are set to any value beyond the maximum filter value.
       See :cite:`hashemi2019filter` for details.
       
       Parameters
       ----------

       method : {'gaussian', 'savgol', 'hilbert', 'wiener', 'medfilt', 'decimate','detrend', 'resample'}, optional (default='savgol')
            Method to filter the scores
       
            - 'gaussian': use a gaussian based filter
            - 'savgol':   use the savgol based filter
            - 'hilbert':  use the hilbert based filter
            - 'wiener':   use the wiener based filter
            - 'medfilt:   use a median based filter
            - 'decimate': use a decimate based filter
            - 'detrend':  use a detrend based filter
            - 'resample': use a resampling based filter
        

       sigma : int, optional (default='auto') 
            Variable specific to each filter type, default sets sigma to len(scores)*np.std(scores)
       
            - 'gaussian': standard deviation for Gaussian kernel
            - 'savgol':   savgol filter window size
            - 'hilbert':  number of Fourier components
            - 'medfilt:   kernel size
            - 'decimate': downsampling factor
            - 'detrend':  number of break points
            - 'resample': resampling window size

       random_state : int, optional (default=1234)
            Random seed for the random number generators of the thresholders. Can also
            be set to None.
    """

    from pythresh.thresholds.filter import FILTER as FILTER_thres
    return FILTER_thres(**kwargs)


def FWFM(**kwargs):
    """FWFM class for Full Width at Full Minimum thresholder.

       Use the full width at full minimum (aka base width) to evaluate
       a non-parametric means to threshold scores generated by the
       decision_scores where outliers are set to any value beyond the base
       width.

       Parameters
       ----------

       random_state : int, optional (default=1234)
            Random seed for the random number generators of the thresholders. Can also
            be set to None.
    """

    from pythresh.thresholds.fwfm import FWFM as FWFM_thres
    return FWFM_thres(**kwargs)


def GAMGMM(**kwargs):
    """GAMGMM class for gammaGMM thresholder.

       Use a Bayesian method for estimating the posterior distribution
       of the contamination factor (i.e., the proportion of anomalies)
       for a given unlabeled dataset. The threshold is set such
       that the proportion of predicted anomalies equals the
       contamination factor.

       Parameters
       ----------

       n_contaminations : int, optional (default=1000)
            number of samples to draw from the contamination posterior distribution

       n_draws : int, optional (default=50)
            number of samples simultaneously drawn from each DPGMM component

       p0 : float, optional (default=0.01)
            probability that no anomalies are in the data

       phigh : float, optional (default=0.01)
            probability that there are more than high_gamma anomalies

       high_gamma : float, optional (default=0.15)
            sensibly high number of anomalies that has low probability to occur

       gamma_lim : float, optional (default=0.5)
            Upper gamma/proportion of anomalies limit

       K : int, optional (default=100)
            number of components for DPGMM used to approximate the Dirichlet Process

       skip : bool, optional (default=False)
            skip optimal hyperparameter test (this may return a sub-optimal solution)

       steps : int, optional (default=100)
            number of iterations to test for optimal hyperparameters

       random_state : int, optional (default=1234)
            Random seed for the random number generators of the thresholders. Can also
            be set to None.

       verbose : bool, optional (default=False)
            20 iterations step printout of the DPGMM process

    """

    from pythresh.thresholds.gamgmm import GAMGMM as GAMGMM_thres
    return GAMGMM_thres(**kwargs)


def GESD(**kwargs):
    """GESD class for Generalized Extreme Studentized Deviate thresholder.

       Use the generalized extreme studentized deviate to evaluate a
       non-parametric means to threshold scores generated by the decision_scores
       where outliers are set to any less than the smallest detected outlier.
       
       Parameters
       ----------

       max_outliers : int, optional (default='auto')
            maximum number of outliers that the dataset may have. Default sets
            max_outliers to be half the size of the dataset

       alpha : float, optional (default=0.05)
            significance level

       random_state : int, optional (default=1234)
            Random seed for the random number generators of the thresholders. Can also
            be set to None.

    """

    from pythresh.thresholds.gesd import GESD as GESD_thres
    return GESD_thres(**kwargs)


def HIST(**kwargs):
    """HIST class for Histogram based thresholders.

       Use histograms methods as described in scikit-image.filters to
       evaluate a non-parametric means to threshold scores generated by
       the decision_scores where outliers are set by histogram generated
       thresholds depending on the selected methods. 
       
       Parameters
       ----------
       
       nbins : int, optional (default='auto')
            Number of bins to use in the histogram, default set to int(len(scores)**0.7)

       method : {'otsu', 'yen', 'isodata', 'li', 'minimum', 'triangle'}, optional (default='triangle')
            Histogram filtering based method
            
            - 'otsu':     OTSU's method for filtering
            - 'yen':      Yen's method for filtering
            - 'isodata':  Ridler-Calvard or inter-means method for filtering
            - 'li':       Li's iterative Minimum Cross Entropy method for filtering
            - 'minimum':  Minimum between two maxima via smoothing method for filtering
            - 'triangle': Triangle algorithm method for filtering

       random_state : int, optional (default=1234)
            Random seed for the random number generators of the thresholders. Can also
            be set to None.
    """

    from pythresh.thresholds.hist import HIST as HIST_thres
    return HIST_thres(**kwargs)


def IQR(**kwargs):
    """IQR class for Inter-Qaurtile Region thresholder.

       Use the inter-quartile region to evaluate a non-parametric
       means to threshold scores generated by the decision_scores
       where outliers are set to any value beyond the third quartile
       plus 1.5 times the inter-quartile region.

       Parameters
       ----------

       random_state : int, optional (default=1234)
            Random seed for the random number generators of the thresholders. Can also
            be set to None.
    """

    from pythresh.thresholds.iqr import IQR as IQR_thres
    return IQR_thres(**kwargs)


def KARCH(**kwargs):
    """KARCH class for Riemannian Center of Mass thresholder.

       Use the Karcher mean (Riemannian Center of Mass) to evaluate a
       non-parametric means to threshold scores generated by the
       decision_scores where outliers are set to any value beyond the
       Karcher mean + one standard deviation of the decision_scores.
       
       Parameters
       ----------

       ndim : int, optional (default=2)
            Number of dimensions to construct the Euclidean manifold

       method : {'simple', 'complex'}, optional (default='complex')
            Method for computing the Karcher mean
       
            - 'simple':  Compute the Karcher mean using the 1D array of scores
            - 'complex': Compute the Karcher mean between a 2D array dot product of the scores and the sorted scores arrays

       random_state : int, optional (default=1234)
            Random seed for the random number generators of the thresholders. Can also
            be set to None.
    """

    from pythresh.thresholds.karch import KARCH as KARCH_thres
    return KARCH_thres(**kwargs)


def MAD(**kwargs):
    """MAD class for Median Absolute Deviation thresholder.

       Use the median absolute deviation to evaluate a non-parametric
       means to threshold scores generated by the decision_scores
       where outliers are set to any value beyond the mean plus the 
       median absolute deviation over the standard deviation.

       Parameters
       ----------

       factor : int, optional (default=1)
            The factor to multiply the MAD by to set the threshold.
            The default is 1.
            
       random_state : int, optional (default=1234)
            Random seed for the random number generators of the thresholders. Can also
            be set to None.
    """

    from pythresh.thresholds.mad import MAD as MAD_thres
    return MAD_thres(**kwargs)


def MCST(**kwargs):
    """MCST class for Monte Carlo Shapiro Tests thresholder.

       Use uniform random sampling and statstical testing to evaluate a
       non-parametric means to threshold scores generated by the decision_scores
       where outliers are set to any value beyond the minimum value left after
       iterative Shapiro-Wilk tests have occured. Note** accuracy decreases with
       array size. For good results the should be array<1000. However still this
       threshold method may fail at any array size.
       
       Parameters
       ----------

       random_state : int, optional (default=1234)
            Random seed for the uniform distribution. Can also be set to None.

    """

    from pythresh.thresholds.mcst import MCST as MCST_thres
    return MCST_thres(**kwargs)


def META(**kwargs):
    """META class for Meta-modelling thresholder.

       Use a trained meta-model to evaluate a non-parametric means
       to threshold scores generated by the decision_scores where outliers
       are set based on the trained meta-model classifier. 
       
       Parameters
       ----------

       method : {'LIN', 'GNB', 'GNBC', 'GNBM'}, optional (default='GNBM')
           select

           - 'LIN':  RidgeCV trained linear classifier meta-model on true labels
           - 'GNB':  Gaussian Naive Bayes trained classifier meta-model on true labels
           - 'GNBC': Gaussian Naive Bayes trained classifier meta-model on best contamination
           - 'GNBM': Gaussian Naive Bayes multivariate trained classifier meta-model

       random_state : int, optional (default=1234)
            Random seed for the random number generators of the thresholders. Can also
            be set to None.

    """

    from pythresh.thresholds.meta import META as META_thres
    return META_thres(**kwargs)

def MIXMOD(**kwargs):
    """MIXMOD class for the Normal & Non-Normal Mixture Models thresholder.

       Use normal & non-normal mixture models to find a non-parametric means
       to threshold scores generated by the decision_scores, where outliers
       are set to any value beyond the posterior probability threshold
       for equal posteriors of a two distribution mixture model.
       
       Parameters
       ----------

       method : str, optional (default='mean')
            Method to evaluate selecting the best fit mixture model. Default
            'mean' sets this as the closest mixture models to the mean of the posterior
            probability threshold for equal posteriors of a two distribution mixture model
            for all fits. Setting 'ks' uses the two-sample Kolmogorov-Smirnov test for
            goodness of fit.

       tol : float, optional (default=1e-5)
            Tolerance for convergence of the EM fit

       max_iter : int, optional (default=250)
            Max number of iterations to run EM during fit

       random_state : int, optional (default=1234)
            Random seed for the random number generators of the thresholders. Can also
            be set to None.

    """

    from pythresh.thresholds.mixmod import MIXMOD as MIXMOD_thres
    return MIXMOD_thres(**kwargs)


def MOLL(**kwargs):
    """MOLL class for Friedrichs' mollifier thresholder.

       Use the Friedrichs' mollifier to evaluate a non-parametric means
       to threshold scores generated by the decision_scores where outliers
       are set to any value beyond one minus the  maximum of the smoothed
       dataset via convolution.

       Parameters
       ----------

       random_state : int, optional (default=1234)
            Random seed for the uniform distribution. Can also be set to None.
    """

    from pythresh.thresholds.moll import MOLL as MOLL_thres
    return MOLL_thres(**kwargs)


def MTT(**kwargs):
    """MTT class for Modified Thompson Tau test thresholder.

       Use the modified Thompson Tau test to evaluate a non-parametric means
       to threshold scores generated by the decision_scores where outliers
       are set to any value beyond the smallest outlier detected by the test.
       
       Parameters
       ----------

       alpha : float, optional (default=0.01)
            Confidence level corresponding to the t-Student distribution map to sample

       random_state : int, optional (default=1234)
            Random seed for the random number generators of the thresholders. Can also
            be set to None.
    """
    from pythresh.thresholds.mtt import MTT as MTT_thres
    return MTT_thres(**kwargs)


def OCSVM(**kwargs):
    """OCSVM class for One-Class Support Vector Machine thresholder.

       Use a one-class svm to evaluate a non-parametric means
       to threshold scores generated by the decision_scores where outliers
       are determined by the one-class svm using a polynomial kernel
       with the polynomial degree either set or determined by regression
       internally.
       
       Parameters
       ----------
       
       model : {'poly', 'sgd'}, optional (default='sgd')
           OCSVM model to apply

           - 'poly':  Use a polynomial kernel with a regular OCSVM
           - 'sgd':   Used the Additive Chi2 kernel approximation with a SGDOneClassSVM

       degree : int, optional (default='auto')
           Polynomial degree to use for the one-class svm.
           Default 'auto' finds the optimal degree with linear regression

       gamma : float, optional (default='auto')
           Kernel coefficient for polynomial fit for the one-class svm.
           Default 'auto' uses 1 / n_features

       criterion : {'aic', 'bic'}, optional (default='bic')
           regression performance metric. AIC is the Akaike Information Criterion,
           and BIC is the Bayesian Information Criterion. This only applies
           when degree is set to 'auto'

       nu : float, optional (default='auto')
           An upper bound on the fraction of training errors and a lower bound
           of the fraction of support vectors. Default 'auto' sets nu as the ratio
           between the any point that is less than or equal to the median plus
           the absolute difference between the mean and geometric mean over the
           the number of points in the entire dataset 

       tol : float, optional (default=1e-3)
           The stopping criterion for the one-class svm
           
       random_state : int, optional (default=1234)
           Random seed for the SVM's data sampling. Can also be set to None.

    """

    from pythresh.thresholds.ocsvm import OCSVM as OCSVM_thres
    return OCSVM_thres(**kwargs)


def QMCD(**kwargs):
    """QMCD class for Quasi-Monte Carlo Discreprancy thresholder.

       Use the quasi-Monte Carlo discreprancy to evaluate a non-parametric means
       to threshold scores generated by the decision_scores where outliers
       are set to any value beyond and percentile or quantile of one minus the
       descreperancy (Note** A discrepancy quantifies the distance between the
       continuous uniform distribution on a hypercube and the discrete uniform
       distribution on distinct sample points).
       
       Parameters
       ----------

       method : {'CD', 'WD', 'MD', 'L2-star'}, optional (default='WD')
            Type of discrepancy
        
            - 'CD':      Centered Discrepancy
            - 'WD':      Wrap-around Discrepancy
            - 'MD':      Mix between CD/WD
            - 'L2-star': L2-star discrepancy

       lim : {'Q', 'P'}, optional (default='P')
            Filtering method to threshold scores using 1 - discrepancy

            - 'Q': Use quantile limiting
            - 'P': Use percentile limiting

       random_state : int, optional (default=1234)
            Random seed for the random number generators of the thresholders. Can also
            be set to None.

    """

    from pythresh.thresholds.qmcd import QMCD as QMCD_thres
    return QMCD_thres(**kwargs)


def REGR(**kwargs):
    """REGR class for Regression based thresholder.

       Use the regression to evaluate a non-parametric means
       to threshold scores generated by the decision_scores where outliers
       are set to any value beyond the y-intercept value of the linear fit.
       
       Parameters
       ----------

       method : {'siegel', 'theil'}, optional (default='siegel')
            Regression based method to calculate the y-intercept
            
            - 'siegel': implements a method for robust linear regression using repeated medians
            - 'theil':  implements a method for robust linear regression using paired values
            
       random_state : int, optional (default=1234)
            random seed for the normal distribution. Can also be set to None

    """

    from pythresh.thresholds.regr import REGR as REGR_thres
    return REGR_thres(**kwargs)


def VAE(**kwargs):
    """VAE class for Variational AutoEncoder thresholder.

       Use a VAE to evaluate a non-parametric means
       to threshold scores generated by the decision_scores where outliers
       are set to any value beyond the maximum minus the minimum of the
       reconstructed distribution probabilities after encoding.

       Parameters
       ----------

       verbose : bool, optional (default=False)
            display training progress

       device : str, optional (default='cpu')
            device for pytorch

       latent_dims : int, optional (default='auto')
            number of latent dimensions the encoder will map the scores to.
            Default 'auto' applies automatic dimensionality selection using
            a profile likelihood.

       random_state : int, optional (default=1234)
            random seed for the normal distribution. Can also be set to None

       epochs : int, optional (default=100)
            number of epochs to train the VAE

       batch_size : int, optional (default=64)
            batch size for the dataloader during training

       loss : str, optional (default='kl')
            Loss function during training

            - 'kl' : use the combined negative log likelihood and Kullback-Leibler divergence
            - 'mmd': use the combined negative log likelihood and maximum mean discrepancy

       Attributes
       ----------

       thresh_ : threshold value that separates inliers from outliers
    """

    from pythresh.thresholds.vae import VAE as VAE_thres
    return VAE_thres(**kwargs)


def WIND(**kwargs):
    """WIND class for topological Winding number thresholder.

       Use the topological winding number (with respect to the origin) to
       evaluate a non-parametric means to threshold scores generated by
       the decision_scores where outliers are set to any value beyond the
       mean intersection point calculated from the winding number.
       
       Parameters
       ----------

       random_state : int, optional (default=1234)
            Random seed for the normal distribution. Can also be set to None.

    """

    from pythresh.thresholds.wind import WIND as WIND_thres
    return WIND_thres(**kwargs)


def YJ(**kwargs):
    """YJ class for Yeo-Johnson transformation thresholder.

       Use the Yeo-Johnson transformation to evaluate
       a non-parametric means to threshold scores generated by the
       decision_scores where outliers are set to any value beyond the
       max value in the YJ transformed data.

       Parameters
       ----------

       random_state : int, optional (default=1234)
            Random seed for the random number generators of the thresholders. Can also
            be set to None.
    """

    from pythresh.thresholds.yj import YJ as YJ_thres
    return YJ_thres(**kwargs)


def ZSCORE(**kwargs):
    """ZSCORE class for ZSCORE thresholder.

       Use the zscore to evaluate a non-parametric means to threshold
       scores generated by the decision_scores where outliers are set
       to any value beyond a zscore of one.

       Parameters
       ----------

       factor : int, optional (default=1)
            The factor to multiply the zscore by to set the threshold.
            The default is 1.
            
       random_state : int, optional (default=1234)
            Random seed for the random number generators of the thresholders. Can also
            be set to None.
    """

    from pythresh.thresholds.zscore import ZSCORE as ZSCORE_thres
    return ZSCORE_thres(**kwargs)
