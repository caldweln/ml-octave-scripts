function centroids = computeCentroids(X, idx, K)
%   centroids = COMPUTECENTROIDS(X, idx, K) returns the new centroids by 
%   computing the means of the data points assigned to each centroid. It is
%   given a dataset X where each row is a single data point, a vector
%   idx of centroid assignments (i.e. each entry in range [1..K]) for each
%   example, and K, the number of centroids.

	[m n] = size(X);

	centroids = zeros(K, n);

	for i=1:K
		% get specific subset of X assigned to old centroid
		X_sub = X((idx==i),:);
		% compute mean and assign as centroid
		centroids(i,:) = sum(X_sub,1) ./ (size(X_sub,1));

	end

end

