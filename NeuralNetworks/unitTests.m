%%%%%%%%%%%%%%%%%%
% nnCostFunction.m
%%%%%%%%%%%%%%%%%%

%!test
%! il = 2;              % input layer
%! hl = 2;              % hidden layer
%! nl = 4;              % number of labels
%! nn = [ 1:18 ] / 10;  % nn_params
%! X = cos([1 2 ; 3 4 ; 5 6]);
%! y = [4; 2; 3];
%! lambda = 4;
%! [j g] = nnCostFunction(nn, il, hl, nl, X, y, lambda);
%! assert (j, [19.474], 1e-03);
%! assert (g, [0.76614; 0.97990; 0.37246; 0.49749; 0.64174; 0.74614; 0.88342; 0.56876; 0.58467; 0.59814; 1.92598; 1.94462; 1.98965; 2.17855; 2.47834; 2.50225; 2.52644; 2.72233], 1e-05);

%%%%%%%%%%%%%%%%%%%
% sigmoidGradient.m
%%%%%%%%%%%%%%%%%%%

%!test
%! z = [[-1 -2 -3] ; magic(3)];
%! g = sigmoidGradient(z);
%! assert(g, [1.9661e-001 1.0499e-001 4.5177e-002; 3.3524e-004 1.9661e-001 2.4665e-003; 4.5177e-002 6.6481e-003 9.1022e-004; 1.7663e-002 1.2338e-004 1.0499e-001], 1e-05);
