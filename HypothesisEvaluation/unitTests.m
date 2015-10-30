%%%%%%%%%%%%%%%%%%
% learningCurve.m
%%%%%%%%%%%%%%%%%%

%!test
%! X = [1 2; 1 3; 1 4; 1 5];
%! y = [7;6;5;4];
%! Xval = [1 7; 1 -2;];
%! yval = [2; 12];
%! lambda = 7;
%! [error_train, error_val] = learningCurve(X, y, Xval, yval, lambda);
%! assert(error_train, [0.00000; 0.10889; 0.20165; 0.21267]);
%! assert(error_val, [12.5000; 11.1700; 8.3951; 5.4696]);

%%%%%%%%%%%%%%%%%%%
% validationCurve.m
%%%%%%%%%%%%%%%%%%%

%!test
%! X = [1 2 ; 1 3 ; 1 4 ; 1 5]
%! y = [7 6 5 4]'
%! Xval = [1 7 ; 1 -2]
%! yval = [2 12]'
%! [lambda_vec, error_train, error_val] = validationCurve(X, y, Xval, yval )
%! assert(lambda_vec,[0.00000; 0.00100; 0.00300; 0.01000; 0.03000; 0.10000; 0.30000; 1.00000; 3.00000; 10.00000]);
%! assert(error_train,[0.00000; 0.00000; 0.00000; 0.00000; 0.00002; 0.00024; 0.00200; 0.01736; 0.08789; 0.27778]);
%! assert(error_val,[0.25000; 0.25055; 0.25165; 0.25553; 0.26678; 0.30801; 0.43970; 1.00347; 2.77539; 6.80556]);

%%%%%%%%%%%%%%%%%%%
% polyFeatures.m
%%%%%%%%%%%%%%%%%%%

%!test
%! X = [1;2;3];
%! p = 4;
%! X_poly = polyFeatures([1:3]', 4);
%! assert(X_poly,[1 1 1 1; 2 4 8 16; 3 9 27 81]);

%%%%%%%%%%%%%%%%%%%%
% featureNormalize.m
%%%%%%%%%%%%%%%%%%%%

%!test
%! X = [ 8 1 6; 3 5 7; 4 9 2 ];
%! [X_norm, mu, sigma] = featureNormalize(X);
%! assert(X_norm, [1.1339 -1.0000 0.3780; -0.7559 0 0.7559; -0.3780 1.0000 -1.1339], 1e-04);
%! assert(mu, [5 5 5]);
%! assert(sigma, [2.6458 4.0000 2.6458], 1e-04);
