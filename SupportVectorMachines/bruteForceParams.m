function [C, sigma] = bruteForceParams(X, y, Xval, yval)
%   [C, sigma] = bruteForceParams(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

	% build steps {0.001, 0.003, 0.01, 0.03, ...}
	steps = [power(10,-2:2); 3*power(10,-2:2)];
	steps = steps(:);
	step_count = length(steps);

	err = 0;
	best_err = realmax();
	best_i = 1;
	best_j = 1;

	% brute force, check performance for every C/sigma combination
	for i=1:step_count
		C = steps(i)
		
		for j=1:step_count
		sigma = steps(j)

		model = svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));
		predictions = svmPredict(model, Xval);
		err = mean(double(predictions ~= yval))

			if(err<best_err)
				% new performance record
				best_i = i;
				best_j = j;
				best_err = err;
			end
		end
	end

	C = steps(best_i);
	sigma = steps(best_j);


end
