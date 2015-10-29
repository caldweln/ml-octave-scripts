function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

	% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
	% for our 2 layer neural network
	Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
					 hidden_layer_size, (input_layer_size + 1));

	Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
					 num_labels, (hidden_layer_size + 1));

	m = size(X, 1);

	J = 0;
	Theta1_grad = zeros(size(Theta1));
	Theta2_grad = zeros(size(Theta2));


	% computing cost J, forward propagation

	a1_1 = [ones(m,1) X];
	z2 = a1_1*Theta1';
	a2_1 = [ones(m,1) sigmoid(z2)];
	z3 = a2_1*Theta2';
	h =  sigmoid(z3);

	yVec = zeros(m,num_labels);
	for i = 1:m
		yVec(i,y(i)) = 1;
	end

	pos_cost = sum(sum(yVec .* log(h)));
	neg_cost = sum(sum((1 - yVec) .* log(1 - h)));

	J = (-1/m)*(pos_cost + neg_cost);

	% adding regularization to cost J

	Theta1(:,1) = 0;
	Theta2(:,1) = 0;
	param_sum = sum(sum(Theta1.*Theta1)) + sum(sum(Theta2.*Theta2));
	reg_cost = (lambda/(2*m))*param_sum;

	J = J + reg_cost;

	% computing gradient grad of neural network, back propagation

	Theta2_unbias = Theta2(:,2:end);

	g2 = sigmoidGradient(z2);

	d3 = h - yVec;
	d2 = (d3 * Theta2_unbias) .* g2;

	Delta1 = d2' * a1_1;
	Delta2 = d3' * a2_1;

	Theta1_grad = Delta1 / m;
	Theta2_grad = Delta2 / m;

	% adding regularization to gradient

	reg_grad1 = (lambda/m) * Theta1;
	reg_grad2 = (lambda/m) * Theta2;

	Theta1_grad = Theta1_grad + reg_grad1;
	Theta2_grad = Theta2_grad + reg_grad2;

	% Unroll gradients
	grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
