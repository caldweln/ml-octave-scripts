function g = sigmoid(z)
%   g = SIGMOID(z) computes the sigmoid of z.
%   Where z may be a scalar, vector or matrix.

	g = 1 ./ (1 + exp( -1 .* z));

end
