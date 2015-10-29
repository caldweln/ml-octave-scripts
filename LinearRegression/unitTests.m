%%%%%%%%%%%%%%%%
% costFunction.m
%%%%%%%%%%%%%%%%

%!test
%! X = [ones(3,1) magic(3)];
%! y = [7; 6; 5];
%! theta = [0.1; 0.2; 0.3; 0.4];
%! lambda = 7;
%! [J grad] = costFunction(X, y, theta, lambda);
%! assert (J, [1.6917], 1e-04);
%! assert (grad, [-1.4000; -8.2667; -3.6333; -7.0000], 1e-04);

%%%%%%%%%%%%%%%%%%%
% gradientDescent.m
%%%%%%%%%%%%%%%%%%%

%!test
%! X = [1 5; 1 2; 1 4; 1 5];
%! y = [1; 6; 4; 2];
%! theta = [0; 0];
%! alpha = 0.01;
%! num_iters = 1000;
%! [theta J_hist] = gradientDescent(X, y, theta, alpha, num_iters);
%! assert (theta, [5.2148; -0.5733], 5e-05);
%! assert (J_hist(1), 5.9794, 3e-05);
%! assert (J_hist(1000), 0.85426, 5e-06);

%!test
%! X = [1 5; 1 2];
%! y = [1; 6];
%! theta = [0.5; 0.5];
%! alpha = 0.1;
%! num_iters = 10;
%! [theta J_hist] = gradientDescent(X, y, theta, alpha, num_iters);
%! assert (theta, [1.70986; 0.19229], 1e-05);
%! assert (J_hist(1:5), [5.8853; 5.7139; 5.5475; 5.3861; 5.2294], 1e-04);
%! assert (J_hist(6:10), [5.0773; 4.9295; 4.7861; 4.6469; 4.5117], 1e-04);

