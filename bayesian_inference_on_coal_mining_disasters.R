disasters <- c(4,5,4,1,0,4,3,4,0,6,3,3,4,0,2,6,3,3,5,4,5,3,1,4,4,1,
               5,5,3,4,2,5,2,2,3,4,2,1,3,2,2,1,1,1,1,3,0,0,1,0,1,1,
               0,0,3,1,0,3,2,2,0,1,1,1,0,1,0,1,0,0,0,2,1,0,0,0,1,1,
               0,2,3,3,1,1,2,1,1,1,1,2,4,2,0,0,0,1,4,0,0,0,1,0,0,0,
               0,0,1,0,0,1,0,1)
my_alpha <- 4 #prior parameter 1 on lambda
my_beta <- 1 #prior parameter 2 on lambda
my_gamma <- 1 #prior parameter 1 on phi
my_delta <- 2 #prior parameter 2 on on phi

bcp <- function(theta_matrix, y, a, b, g, d){
  n <- length(y)
  k_prob <- rep(0, times=n)
  for (i in 2:nrow(theta_matrix)){
    lambda <- rgamma(1, a+sum(y[1:theta_matrix[(i-1),3]]), b+theta_matrix[(i-1),3])
    phi <- rgamma(1, g+sum(y[theta_matrix[(i-1),3]:n]), d+n-theta_matrix[(i-1),3])
    for (j in 1:n){
      k_prob[j] <- exp(j*(phi-lambda))*(lambda/phi)^sum(y[1:j])
    }
    k_prob <- k_prob/sum(k_prob)
    k <- sample(1:n, size = 1, prob = k_prob)
    theta_matrix[i,] <- c(lambda, phi, k)
  }
  return(theta_matrix)
}

tot_draws <- 2000
init_param_values <- c(4, 0.5, 55) #initial values of (lambda, phi, k)
#used prior means as initial values

init_theta_matrix <- matrix(0, nrow=tot.draws, ncol=3)
init_theta_matrix <- rbind(init_param_values, init_theta_matrix)

my_gibbs_samples <- bcp(init_theta_matrix, y=disasters, a=my_alpha, b=my_beta, g=my_gamma, d=my_delta)

#remove the first 1000 samples as burn-in

my_post <- my_gibbs_samples[-(1:1000),]
apply(my_post, 2, median)
cbind(apply(my_post, 2, quantile, probs=0.025), apply(my_post,2, quantile, probs=0.975))
