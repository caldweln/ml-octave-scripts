function idx = findClosestCentroids(X, centroids)
%   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
%   in idx for a dataset X where each row is a single example. idx = m x 1 
%   vector of centroid assignments (i.e. each entry in range [1..K])
%

	% Set K
	K = size(centroids, 1);
	m = size(X,1);

	idx = zeros(m, 1);

	for i=1:m

		j = sum(( centroids .-  X(i,:) ) .^2,2);
		[v idx(i)] = min(j);

	end

end

