function [X_poly] = polyFeatures(X, p)
%   [X_poly] = POLYFEATURES(X, p) takes a data matrix X (size m x 1) and
%   maps each example into its polynomial features where
%   X_poly(i, :) = [X(i) X(i).^2 X(i).^3 ...  X(i).^p];
%

	X_poly = zeros(numel(X), p);

	X_poly(:,1) = X(:,1);
	for i = 2:p,

		X_poly(:,i) = X_poly(:,i-1).*X_poly(:,1);

	end;


end
