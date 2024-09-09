function yhat = softmax(z)
%SOFTMAX This function takes in an input vector and calculates a new vector
%that sums up to 1 where every entry is positive and bigger input entries
%correspond to bigger output entries
yhat = exp(z);
yhat = yhat / sum(yhat);
end