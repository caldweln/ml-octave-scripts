function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%   theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

	m = length(y); % number of training examples
	J_history = zeros(num_iters, 1);

	for iter = 1:num_iters

		h = X * theta;
		err = h -y;
		dt =  X' * err * (1/m);

		theta = theta - ( alpha * dt );
	 
		J_history(iter) = costFunction(X, y, theta);

	end

end
