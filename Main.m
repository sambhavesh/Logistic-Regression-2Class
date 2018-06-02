
clear; close all; clc

%% Load Data
data = load('data.txt');
X = data(:, 2:101); y = data(:, 1);
data_test = load('datatest.txt');
X_test = data_test(:, 2:101);  y_test = data_test(:, 1);
%% ============ Compute Cost and Gradient ============
[m, n] = size(X);
[m1, n1] = size(X_test);
X = [ones(m, 1) X];
X_test = [ones(m1, 1) X_test];
initial_theta = zeros(n + 1, 1);

% Compute and display initial cost and gradient
[cost, grad] = costFunction(initial_theta, X, y);

fprintf('Cost at initial theta (zeros): %f\n', cost);
% fprintf('Gradient at initial theta (zeros): \n');
% fprintf(' %f \n', grad);



%% ============= Optimizing using fminunc  =============
options = optimset('GradObj', 'on', 'MaxIter', 400,'TolFun', 01e-10);
[theta, cost] = ...
	fminunc(@(t)(costFunction(t, X, y)), initial_theta, options);

fprintf('Cost at theta found by fminunc: %f\n', cost);
% fprintf('theta: \n');
% fprintf(' %f \n', theta);



%% ============== Predict and Accuracies ==============

p = predict(theta, X);
fprintf('Train Accuracy: %f\n', mean(double(p == y)) * 100);
ptest = predict(theta, X_test);
fprintf('Testing Accuracy: %f\n', mean(double(ptest == y_test)) * 100);

