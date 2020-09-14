setwd("C:/Users/tungt/Dropbox/Machine Learning Summer 2019 - Tung Thai/Homework/Homework 4")
library(readr)
library(nnet)
iris_train <- read_csv("iris_train.csv")
iris_test <- read.csv("iris_test.csv")
#input x sepal length
x_add <- rep(1, nrow(iris_train))
x <- data.frame(iris_train$Sepal.Length)
x1 <- iris_train$Sepal.Length # use to graph
# output y sepal width
y <- data.frame(iris_train$Sepal.Width)
y1 <- iris_train$Sepal.Width # use to graph
alpha = 0.01
iter = 500
n = length(y1)
my.gd <- function(x, y, alpha, iter, n, t.hold) {
  m <- 0
  m_current <- 4
  c <- 0
  c_current <- -0.15
  yhat <- m_current*x + c_current
  MSE <- sum((y - yhat) ^ 2) / n
  converged = 0
  iterations = 0
  while(converged == F) {
    ## Implement the gradient descent algorithm
    m <- m - alpha * ((1 / n) * (sum((yhat - y) * x)))
    c <- c - alpha * ((1 / n) * (sum(yhat - y)))
    m_new <- m_current + alpha*m
    c_new <- c_current + alpha*c
    yhat <- m * x + c
    MSE_new <- sum((y - yhat) ^ 2) / n
    
    if(MSE - MSE_new <= t.hold) {
      converged = 1
    }
    
    iterations = iterations + 1
    if(iterations > iter) { 
      converged = 1
    }
  }
  theta <- c(m_new,c_new)
  return(theta)
}
theta <- my.gd(x1, y1, alpha, iter, n, t.hold = 10e-3)
print("Result by Gradient Descent")
print(t(theta))
pdf("LinearByGD.pdf")
plot(x1,y1, col=rgb(0.3,0.1,0.3,1), main='Linear regression by gradient descent')
abline(theta, col='red')
dev.off()



# By Stochatics Gradient Descent
# x will be n+1 dimensions since x0 is being appended to it
x.sgd <- data.frame(cbind(x_add,iris_train$Sepal.Length))
y.sgd <- data.frame(iris_train$Sepal.Width)
# partial derivative formular
# the gradient descent formula
pd_f <- function(x, y, theta) {
  # need to transpose both x and theta to enable matrix multiplication in R
  p.deriv <- (1/ nrow(y))* (t(x) %*% ((x %*% t(theta)) - y))
  
  # return transpose of the partial derivative
  return(t(p.deriv))
}
# define stochastic gradient descent algorithm
my.sgd <- function(x, y, alpha, n){
  
  # merge x and y
  x.y <- data.frame(cbind(y,x))
  
  # theta MUST be 2 columns and 3 columns for multi: must match width of x matrix
  # set matrix theta, and intial theta as 4/0
  theta <- matrix(c(4, 0), nrow = 1)
  
  # store values of theta for each iteration (history)
  theta.h <- matrix(NA, nrow = n, ncol = 2)
  
  # set seed value for random sampling
  set.seed(123)
  
  # updating theta each step
  for (i in 1:n) {
    # randomly sample 4 items from the combined xy data frame
    xysamp <- as.matrix( x.y[sample(nrow(x.y), 4, replace = TRUE), ] )
    
    # isolate 'x' component of random samples
    xsamp <- as.matrix(xysamp[,2:3])
    
    # isolate 'y' component of random samples
    ysamp <- as.matrix(xysamp[,1])
    
    # update theta using mini batches
    # theta <- theta - 0.001  * pd_f(xsamp, ysamp, theta) 
    theta <- theta - alpha  * pd_f(xsamp, ysamp, theta) 
    
    # save the theta values for iteration i to a matrix for future plotting
    theta.h[i,] <- theta
    
  } # end for loop
  return(theta.h)
}
result.sgd <- my.sgd(x.sgd, y.sgd, alpha = alpha, iter)
print("Result by Stochatics Gradient Descent")
print(result.sgd[iter,])
pdf("LinearBySGD.pdf")
plot(x1,y1, col=rgb(0.3,0.1,0.3,1), main='Linear regression by stochatics gradient descent')
abline(t(result.sgd[iter,]), col='green')
dev.off()



# check with lm()
res <- lm(y1~x1)
print("Result by lm function")
print(res)
pdf("LinearBylm.pdf")
plot(x1,y1, col=rgb(0.3,0.1,0.3,1), main='Linear regression by lm function')
abline(res, col='blue')
dev.off()



# compare:
pdf("CompareLinearRegression.pdf")
plot(x1,y1, col=rgb(0.3,0.1,0.3,1), main='Comparison')
abline(res, col='blue')
abline(t(result.sgd[iter,]), col='green')
abline(theta, col='red')
legend("topleft", legend=c("GD", "SGD", "nnet"),
       col=c("red", "green", "blue"), lty=1, cex=0.8)
dev.off()
