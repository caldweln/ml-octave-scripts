function centroids = kMeansInitCentroids(X, K)
%   centroids = KMEANSINITCENTROIDS(X, K) returns K initial centroids to be
%   used with the K-Means on the dataset X
%

	centroids = zeros(K, size(X, 2));

	randidx = randperm(size(X,1));
	centroids = X(randidx(1:K),:);

end

