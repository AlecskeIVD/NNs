function [dB1, dW1, dB2, dW2, dB3, dW3] = alt_calcGradient(inputimage, label, B1, W1, B2, W2, B3, W3)
sigma = @(x) 1 ./ (1+exp(-x));
x1 = sigma(W1 * inputimage + B1);
x2 = sigma(W2 * x1 + B2);
zout = sigma(W3 * x2 + B3);
yhat = softmax(zout);
k = label+1;

dB3 = yhat;
dB3(k) = yhat(k)-1;
dW3 = dB3 * x2';

dB2 = (W3' * dB3) .* x2 .* (1-x2);
dW2 = dB2 * x1';

dB1 = (W2' * dB2) .* x1 .* (1-x1);
dW1 = dB1 * inputimage';

end