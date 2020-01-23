import numpy as np
from math import gamma
import scipy.stats as stat
import cvxpy as cp
from scipy.stats import bernoulli

########### Multivariate T-distribution helper ###############################

def multivariate_t_distribution(x, mu, Sigma, df=np.inf):
    '''
    Multivariate t-student density. Returns the density
    of the function at points specified by x.

    input:
        x = parameter (n-d numpy array; will be forced to 2d)
        mu = mean (d dimensional numpy array)
        Sigma = scale matrix (dxd numpy array)
        df = degrees of freedom
    '''

    x = np.atleast_2d(x) # requires x as 2d
    nD = Sigma.shape[0] # dimensionality

    numerator = gamma(1.0 * (nD + df) / 2.0)

    denominator = (
            gamma(1.0 * df / 2.0) * 
            np.power(df * np.pi, 1.0 * nD / 2.0) *  
            np.power(np.linalg.det(Sigma), 1.0 / 2.0) * 
            np.power(
                1.0 + (1.0 / df) *
                np.diagonal(
                    np.dot( np.dot(x - mu, np.linalg.inv(Sigma)), (x - mu).T)
                ), 
                1.0 * (nD + df) / 2.0
                )
            )
    return 1.0 * numerator / denominator 



def multivariate_t_rvs(mu, Sigma, df=np.inf, n=1):
    '''generate random variables of multivariate t distribution
    Parameters
    ----------
    mu : array_like
        mean of random variable, length determines dimension of random variable
    Sigma : array_like
        square array of covariance  matrix
    df : int or float
        degrees of freedom
    n : int
        number of observations, return random array will be (n, len(m))
    Returns
    -------
    rvs : ndarray, (n, len(m))
        each row is an independent draw of a multivariate t distributed
        random variable
    '''
    mu = np.asarray(mu)
    d = len(mu)
    if df == np.inf:
        x = 1.
    else:
        x = np.random.chisquare(df, n)/df
    z = np.random.multivariate_normal(np.zeros(d),Sigma,(n,))
    return mu + z/np.sqrt(x)[:,None]   # same output format as random.multivariate_normal

##############################################################################################


def generate_regression(n, alpha = 0, beta = 1, loc1=0.75, loc2=-0.75, scale1=0.75, scale2=1.25):
    """Generate univariate regression"""

    z = stat.norm.rvs(loc=0.5,scale=1,size=n)
    ber = stat.bernoulli.rvs(0.5, size=n)
    e1 = stat.norm.rvs(loc=loc1,scale=scale1,size=n) # scale -> std
    e2 = stat.norm.rvs(loc=loc2,scale=scale2,size=n) 

    e = np.zeros(n)
    for i in range(e.shape[0]):
        if ber[i]==1:
            e[i]=e1[i]
        else:
            e[i]=e2[i]

    y = alpha + beta*z + e
    
    return z, y



def newton_solver(G):
    """Solve for lambda using the matrix of moment conditions"""

    n = G.shape[0] # dimension du vecteur lambda à trouver
    lambd = cp.Variable(n) # variable à trouver
    objective = cp.Minimize(cp.log_sum_exp(lambd*G))
    constraints = []
    prob = cp.Problem(objective, constraints)
    # The optimal objective value is returned by `prob.solve()`.
    result = prob.solve(solver=cp.SCS)
    return(lambd.value)

def calculate_posterior(x,y,z,model="M1"):
    
    alpha,beta,v = x[0],x[1],x[2]
    e_theta = lambda alpha, beta: y - alpha - beta*z
    
    ### different G_func for different model ###
    if model == "M1": 
        G_func = lambda alpha, beta, v: np.vstack((e_theta(alpha,beta), e_theta(alpha,beta)*z, e_theta(alpha,beta)**3 - v))
    if model == "M2":
        G_func = lambda alpha, beta, v: np.vstack((e_theta(alpha,beta), e_theta(alpha,beta)*z, e_theta(alpha,beta)**3 ))
            
    G=G_func(alpha,beta,v)
    
    lambd_opt=newton_solver(G)

    #prior distribution
    d=3
    Sigma=0.5*np.eye(d)
    df=2.5
    mu=np.array([0,0,0])

    x = np.array([alpha,beta,v])
    log_pi_theta_prior=np.log(multivariate_t_distribution(x,mu,Sigma,df))
    try:
        log_p_x_theta=(np.sum(np.dot(lambd_opt,G)))-y.shape[0]*np.log(np.sum(np.exp(np.dot(lambd_opt,G))))
    except:
        return
    return ((log_pi_theta_prior+log_p_x_theta),log_pi_theta_prior,log_p_x_theta)


def approx_hessian(alpha,beta,v,y,z,h1=0.01,h2=0.01,model="M1"):
    '''Compute and approximation of the Hessian at the mode'''
    
    d=3
    H = np.zeros((d,d))
    x = np.array([alpha,beta,v])
    e = np.array([[1,0,0],[0,1,0],[0,0,1]])
    
    for i in range(d):
        for j in range(d):
            f12,_,_ = calculate_posterior(x+h1*e[i]+h2*e[j],y,z,model)
            f1,_,_ = calculate_posterior(x+h1*e[i],y,z,model)
            f2,_,_ = calculate_posterior(x+h2*e[j],y,z,model)
            f,_,_ = calculate_posterior(x,y,z,model)
            H[i,j] = (f12-f1-f2+f)/h1/h2
            
    H = (H+H.T)/2
    
    return H




def iteration(alpha_0,beta_0,v_0,y,z, mean,cov,model="M1"):
    '''Perform one iteration of Metroplis-Hasting'''
    
    #génération selon q Student -- loc=mode of logETEL function, scale = - inverse of the Hessian of the logETEL at the mode
    theta = multivariate_t_rvs(mean, cov, df=3, n=1)[0]
    alpha_1,beta_1,v_1=theta[0],theta[1],theta[2]
    q_new=multivariate_t_distribution(theta, mu=mean, Sigma=cov, df=3)
    
    theta_old=np.array([alpha_0,beta_0,v_0])
    q_old = multivariate_t_distribution(theta_old, mu=mean, Sigma=cov, df=3)
    
    #coeff probability
    log_coeff=calculate_posterior(theta,y,z,model)[0]-calculate_posterior(theta_old,y,z,model)[0]+np.log(q_old)-np.log(q_new)
    p=min(np.exp(log_coeff),1)
    
    #acceptation of new with probability p
    r = int(bernoulli.rvs(p, size=1))
    if r == 1:
        alpha_next, beta_next, v_next=alpha_1,beta_1,v_1
    else:
        alpha_next, beta_next, v_next=alpha_0,beta_0,v_0     
    return(alpha_next, beta_next, v_next)


def metropolis(t, burn, y,z, mean, cov):

    # Initialisation
    alpha_0,beta_0,v_0=0.5,0.5,0.5  
    res = np.zeros([t-burn,3])

    for i in range(t):
        try:
            alpha_0,beta_0,v_0=iteration(alpha_0,beta_0,v_0,y,z,mean,cov)
        except Exception as e:
            print(str(e))
            if i >=burn:
                res[i-burn] = np.array([np.nan,np.nan,np.nan])
            continue
        if i>=burn:# on grille les premières générations
            res[i-burn] = np.array([alpha_0,beta_0,v_0])
        if i%50==0:
            print('iteration number: ', i+1)
            print("iter: ", i+1, "coeffs: ",alpha_0,beta_0,v_0)

    return res




def model_hessian_model(model,y,z):
    t = 10
    #t=4
    alpha_s = np.linspace(-1,1,t)
    beta_s = np.linspace(0,2,t)
    v_s = np.linspace(-1,1,t)
    res = np.zeros((t,t,t))
    c=0

    for ix,i in enumerate(alpha_s):
        for jx,j in enumerate(beta_s):
            for lx,l in enumerate(v_s):
                x=np.array([i,j,l])
                try:
                  v,_,_=calculate_posterior(x,y,z,model)
                except:
                  continue
                res[ix,jx,lx] = v
                c+=1
                if c%50==0:
                    print(c)
    
    res[res==0]=-np.infty
    a,b,c = np.unravel_index(np.argmax(res, axis=None), res.shape)
    mean=np.array([alpha_s[a], beta_s[b],v_s[c]])
    cov = -inv(approx_hessian(mean[0],mean[1],mean[2],y,z,h1=0.01,h2=0.01,model=model)) 
    
    return(mean,cov)

def q(phi,mean,cov):
    return(multivariate_t_distribution(phi, mu=mean, Sigma=cov, df=3))


def log_coeff(phi,phi_tild,model,y,z,mean,cov):
    return(calculate_posterior(phi_tild,y,z,model)[0]-calculate_posterior(phi,y,z,model)[0]+np.log(q(phi,mean,cov))-np.log(q(phi_tild,mean,cov)))





def marginal(model_test,y,z):

#il ne manque plus que le troisieme terme de log mx 
  print(model_test)
  phi_tild,cov= model_hessian_model(model_test,y,z) #je colle aux notations du papier mais je prends la moyenne de log posterior pour phi_tild_l (dépend du modele)
  log_posterior, log_prior,log_px = calculate_posterior(phi_tild,y,z,model=model_test)
   
  #step2
  #numérateur il faut générer selon hastings et moyenniser
  nburn=1000
  niter1=1000
  total1=0
  alpha_0,beta_0,v_0=0.5,0.5,0.5 
  print("Espérance 1")
  for i in range(niter1+nburn):
      if(i%100==0 and i<999):
        print(i)
      try:
      #on prend pour moyenne de la loi de proposition phi_tild qui est le mode de la fonction BETEL
        alpha_0,beta_0,v_0=iteration(alpha_0,beta_0,v_0,y,z,phi_tild,cov,model=model_test)
      except:
        continue
      if i>=999:#on grille les premières générations
          if i%100==0:
            print(i)
          gen= np.array([alpha_0,beta_0,v_0])
          total1+=np.exp(log_coeff(gen,phi_tild,model_test,y,z,phi_tild,cov))*q(phi_tild,phi_tild,cov)

  E1=total1/niter1

  print(E1)

  print("Espérance 2")
  #dénominateur
  #génération plusieurs fois suivant q

  niter2=1000
  total2=0
  for i in range(niter2):
      if i%100==0:
        print(i)
      gen=multivariate_t_rvs(phi_tild, cov, df=3, n=1)[0]
      total2+=np.exp(log_coeff(phi_tild,gen,model_test,y,z,phi_tild,cov))
  E2=total2/niter2  
  print(E2)  
  #Et Finalement
  log_mx= log_prior+log_px-np.log(E1/E2)
  print(log_mx)
  return(log_mx)

