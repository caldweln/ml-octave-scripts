function [error_train, error_val] = ...
    learningCurve(X, y, Xval, yval, lambda)
%   [error_train, error_val] = ...
%       LEARNINGCURVE(X, y, Xval, yval, lambda) returns the train and
%       cross validation set errors for a learning curve. In particular, 
%       it returns two vectors of the same length - error_train and 
%       error_val. Then, error_train(i) contains the training error for
%       i examples (and similarly for error_val(i)).

	m = size(X, 1);
	error_train = zeros(m, 1);
	error_val   = zeros(m, 1);

	for i = 1:m,
	 
		J_train = 0;
		J_val = 0;

		for j = 1:50,
			sel = randperm(size(X,1));
			sel = sel(1:i);
			X_train = X(sel,:);
			y_train = y(sel,:);

			theta = trainLinearReg(X_train,y_train,lambda);

			J_train = J_train + (linearRegCostFunction(X_train,y_train,theta,0.01) / 50);
			J_val = J_val + (linearRegCostFunction(Xval,yval,theta,0.01) / 50);

		end;

		error_train(i) = J_train;
		error_val(i) = J_val;
	end;

end
