function [C, sigma] = bruteForceParams(X, y, Xval, yval)
%   [C, sigma] = bruteForceParams(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

	% build steps {0.01, 0.03, 0.1, 0.3, ...}
	steps = [power(10,-2:2); 3*power(10,-2:2)];
	steps = steps(:);
	step_count = length(steps);

	err = 0;
	best_err = realmax();
	best_ind = ones(2,1);

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
				best_ind = [i;j];
				best_err = err;
			end
		end
	end

	C = steps(best_ind(1,1));
	sigma = steps(best_ind(2,1));

end
