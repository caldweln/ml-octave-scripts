%%%%%%%%%%%
% sigmoid.m
%%%%%%%%%%%

%!test
%! assert (sigmoid(1200000), 1);
%! assert (sigmoid(-250000), 0);
%! assert (sigmoid(0), 0.5);
%! assert (sigmoid([4 5;5 6]), [0.9820 0.9933; 0.9933 0.9975], 1e-04);

%%%%%%%%%%%%%%%%
% costFunction.m
%%%%%%%%%%%%%%%%

%!test
%! X = [ones(3,1) magic(3)];
%! y = [1; 0; 1];
%! theta = [-2; -1; 1; 2];
%! lambda = 3;
%! [j g] = lrCostFunction(theta, X, y, lambda);
%! assert (j, [7.6832], 1e-04);
%! assert (g, [0.3172; -0.1277; 2.6481; 4.2379], 1e-04);

%%%%%%%%%%%%
% oneVsAll.m
%%%%%%%%%%%%


%!test
%! X = [magic(3) ; sin(1:3); cos(1:3)];
%! y = [1; 2; 2; 1; 3];
%! num_labels = 3;
%! lambda = 0.1;
%! [all_theta] = oneVsAll(X, y, num_labels, lambda);
%! assert(all_theta, [-0.559478 0.619220 -0.550361 -0.093502; -5.47430 -0.471565 1.26120 0.634930; 0.068368 -0.375582 -1.652262 -1.410138], 1e-04);


%%%%%%%%%%%%%%%%%%%
% predictOneVsAll.m
%%%%%%%%%%%%%%%%%%%


%!test
%! all_theta = [1 -6 3; -2 4 -3];
%! X = [1 7; 4 5; 7 8; 1 4];
%! assert(predictOneVsAll(all_theta, X), [1; 2; 2; 1]);
