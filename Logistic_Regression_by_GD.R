setwd("C:/Users/tungt/Dropbox/Machine Learning Summer 2019 - Tung Thai/Homework/Homework 4")
library(readr)
library(nnet)
iris_train <- read_csv("iris_train.csv")
iris_test <- read.csv("iris_test.csv")
#Logistics Regression
# initialize dummy variable x0 = 1 for each row of predictor data
x_add <- rep(1, nrow(iris_train))
# x will be n+1 dimensions since x0 is being appended to it
x <- data.frame(cbind(x_add,iris_train[,2:3]))
y <- data.frame(iris_train$Species)
x1 <- iris_train$Sepal.Length
x2 <- iris_train$Sepal.Width
y1 <- iris_train$Species
n <- length(y1)
alpha = 0.001
iter = 500
# sigmond function
sigmoid_f <- function(z)
{
  sig <- 1/(1+exp(-z))
  return(sig)
}
# By Gradient Descent
my.gd <- function(x1 ,x2, y, alpha, iter, n, t.hold) {
  m1 <- 0
  m1_current <- -70
  m2 <- 0
  m2_current <- 27
  c <- 0
  c_current <- -24
  yhat <- m1_current*x1 + m2_current*x2 + c_current
  MSE <- sum((y - yhat) ^ 2) / n
  converged = 0
  iterations = 0
  while(converged == 0) {
    ## Implement the gradient descent algorithm
    m1 <- m1 - alpha * ((1 / n) * (sum((yhat - y) * x1)))
    m2 <- m2 - alpha * ((1 / n) * (sum((yhat - y) * x2)))
    c <- c - alpha * ((1 / n) * (sum(yhat - y)))
    m1_new <- m1_current + alpha*m1
    m2_new <- m2_current + alpha*m2
    c_new <- c_current + alpha*c
    yhat <- m1*x1 +m2*x2 + c
    MSE_new <- sum((y - yhat) ^ 2) / n
    
    if(MSE - MSE_new <= t.hold) {
      converged = 1
    }
    
    iterations = iterations + 1
    if(iterations > iter) { 
      converged = 1
    }
  }
  theta <- c(m1_new, m2_new, c_new)
  return(theta)
}
theta.gd <- my.gd(x1, x2, y1, alpha, iter, n, t.hold = 10e-3)
print(theta.gd)
#predict class
pred.gd <- list(nrow(iris_test))
for (i in 1: nrow(iris_test)){
  pred.gd[i] <- sigmoid_f(c(1,iris_test$Sepal.Length[i],iris_test$Sepal.Width[i]) %*% theta.gd)
}
pred.gd <- unlist(pred.gd)
for (i in 1: nrow(iris_test)){
  if (pred.gd[i] < 0.5){
    pred.gd[i] = 0
  }else{
    pred.gd[i] = 1
  }
}
t.gd <- table(pred.gd,iris_test$Species)
print(t.gd)
A1.gd <- t.gd[1,1]/sum(t.gd[1,])
A2.gd <- t.gd[2,2]/sum(t.gd[2,])
OA.gd <- (t.gd[1,1]+t.gd[2,2])/sum(t.gd)
OE.gd <- 1 - OA.gd
results.gd <- cbind(OA.gd,A1.gd,A2.gd,OE.gd)
intercept.gd=theta.gd[1]/(-theta.gd[3])
slope.gd=theta.gd[2]/(-theta.gd[3])
boundary.gd <- c(slope.gd,intercept.gd)
print("Decision Boundary by GD: Slope, Intercept")
print(boundary.gd)
print("The Accuracy Results GD:")
print(results.gd)
#graph
pdf("LogisticsByGD.pdf")
#plot the data in different color
plot(iris_train$Sepal.Length,iris_train$Sepal.Width,col=as.factor(iris_train$Species),xlab="Sepal Length",ylab="Sepal Width")
abline(a=intercept.gd,b=slope.gd, col = 'red')
dev.off


#By Stochatics Gradient Descent
pd_f <- function(x, y, theta) {
  
  # need to transpose both x and theta to enable matrix multiplication in R
  p.deriv <- (1/ nrow(y))* (t(x) %*% ((x %*% t(theta)) - y))
  
  # return transpose of the partial derivative
  return(t(p.deriv))
}
my.sgd <- function(x, y, alpha, n){
  
  # merge x and y to enable accurate random sampling
  x.y <- data.frame(cbind(y,x))
  
  # theta MUST be 3 columns: must match width of x matrix
  # set matrix theta, initial theta near the actual result, this can be change to 0,0,0
  theta <- matrix(c(-70, 27, -24), nrow = 1)
  
  # theta history
  theta.h <- matrix(NA, nrow = n, ncol = 3)
  
  # set seed value for random sampling
  set.seed(123)
  
  # now iterate using mini batches of randomly sampled  data, updating theta each step
  for (i in 1:n) {
    # randomly sample 4 items from the combined xy data frame
    xysamp <- as.matrix( x.y[sample(nrow(x.y), 4, replace = TRUE), ] )
    
    # isolate 'x' component of random samples
    xsamp <- as.matrix(xysamp[,2:4])
    
    # isolate 'y' component of random samples
    ysamp <- as.matrix(xysamp[,1])
    
    # theta <- theta - 0.001  * pd_f(xsamp, ysamp, theta) 
    theta <- theta - alpha  * pd_f(xsamp, ysamp, theta) 
    
    # save the theta values for iteration i to a matrix for future plotting
    theta.h[i,] <- theta
    
  } # end for loop
  return(theta.h)
}
theta.sgd <- my.sgd(x,y,alpha = alpha, iter)
theta_mul <- theta.sgd[iter,]
print(theta_mul)
#predict class
pred <- list(nrow(iris_test))
for (i in 1: nrow(iris_test)){
  pred[i] <- sigmoid_f(c(1,iris_test$Sepal.Length[i],iris_test$Sepal.Width[i]) %*% theta_mul)
}
pred <- unlist(pred)
for (i in 1: nrow(iris_test)){
  if (pred[i] < 0.5){
    pred[i] = 0
  }else{
    pred[i] = 1
  }
}
t <- table(pred,iris_test$Species)
print(t)
A1 <- t[1,1]/sum(t[1,])
A2 <- t[2,2]/sum(t[2,])
OA <- (t[1,1]+t[2,2])/sum(t)
OE <- 1 - OA
results <- cbind(OA,A1,A2,OE)
intercept.sgd=theta_mul[1]/(-theta_mul[3])
slope.sgd=theta_mul[2]/(-theta_mul[3])
boundary.sgd <- c(slope.sgd,intercept.sgd)
print("Decision Boundary by SGD: Slope, Intercept")
print(boundary.sgd)
print("The Accuracy Results SGD:")
print(results)
#graph
pdf("LogisticsBySGD.pdf")
#plot the data in different color
plot(iris_train$Sepal.Length,iris_train$Sepal.Width,col=as.factor(iris_train$Species),xlab="Sepal Length",ylab="Sepal Width")
abline(a=intercept.sgd,b=slope.sgd, col = 'green')
dev.off



# confirm with mul nnet
multi1 <- multinom(iris_train$Species ~.,data=iris_train)
B <- coef(multi1)
print(B)
interceptg=B[1]/(-B[3])
slopeg=B[2]/(-B[3])
data1 <- iris_train[which(iris_train[,1]==0),]
data2 <- iris_train[which(iris_train[,1]==1),]
pdf("LogisticByMul.pdf")
#plot the data in different color
plot(iris_train$Sepal.Length,iris_train$Sepal.Width,col=as.factor(iris_train$Species),xlab="Sepal Length",ylab="Sepal Width")
abline(a=interceptg,b=slopeg, col = 'blue')
dev.off()
actual.value <- iris_test$Species
iris_test <- iris_test[,-1]
predict_class <- predict(multi1, iris_test)
t_mul <- table(predict_class,actual.value)
print(t_mul)
A1_c <- t_mul[1,1]/sum(t_mul[1,])
A2_c <- t_mul[2,2]/sum(t_mul[2,])
OA_c <- (t_mul[1,1]+t_mul[2,2])/sum(t_mul)
OE_c <- 1 - OA_c
results_c <- cbind(OA_c,A1_c,A2_c,OE_c)
boundary.mul <- c(slopeg, interceptg)
print("Decision Boundary by Mul: Slope, Intercept")
print(boundary.mul)
print("The Accuracy Results with nnet:")
print(results_c)

pdf("CompareLogisticsRegression.pdf")
#plot the data in different color
plot(iris_train$Sepal.Length,iris_train$Sepal.Width,col=as.factor(iris_train$Species),xlab="Sepal Length",ylab="Sepal Width")
abline(a=interceptg,b=slopeg, col = 'blue')
abline(a=intercept_m,b=slope_m, col = 'green')
abline(a=intercept.gd,b=slope.gd, col = 'red')
legend("topleft", legend=c("GD", "SGD", "nnet"),
       col=c("red", "green", "blue"), lty=1, cex=0.8)
dev.off()
