%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% RECONSTRUCT1D_ADVANCED_EXAMPLE
%
% Example on how to use Reconstruct1D based on Persistence1D results. 
%
% This file shows how to use Persistence1D results to call Reconstruct1D 
% with different parameters.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% clear all;
close all;

% Add Reconstruct1D folder to Matlab's path
% addpath('..');

% Add Persistence1D matlab folder to path. 
setup_persistence1d();

% Turn on MOSEK optimzers for faster run times
turn_on_mosek();

% Read the data set
% data = [ 0 1 -1 0 1 0 0 ];
% data = [2,5,7,-12,-13,-7,10,18,6,8,7,4];
data = load('../../regularity_score.mat');
data = double(data.regularity_score);

% Convert input data to single precision to work with Persistence1D
single_precision_data = single(data);

% Run Persistence1D on the data
[minIndices, maxIndices, persistence, globalMinIndex, globalMinValue] = run_persistence1d(single_precision_data); 

% Set threshold for surviving features
threshold = 0.01;

% Filter Persistence1D paired extrema for relevant features
pairs = filter_features_by_persistence(minIndices, maxIndices, persistence, threshold);

% interpolated minima and maxima
mins = get_min_indices(pairs);
maxs = get_max_indices(pairs);

% Set the data weight. Choosing 0.0 constructs smoother function
data_weight = 0.01;

% Set the smoothness for the results. 
bi_smoothness = 'biharmonic';

% Call reconstruct1d_with_persistence_res. Calling this function directly avoids
% re-running Persistence1D for different reconstruction parameters
x_bi_smooth  = reconstruct1d_with_persistence_res( data, ...
													mins, ...
													maxs, ...
													globalMinIndex, ...
													bi_smoothness, ...
													data_weight);

													
% Call reconstruct1d_with_persistence_res again with different parameters
tri_smoothness = 'triharmonic';		
x_tri_smooth  = reconstruct1d_with_persistence_res( data, ...
													mins, ...
													maxs, ...
													globalMinIndex, ...
													tri_smoothness, ...
													data_weight);

													
% Plotting the results together for comparison
plot(data);
title(['Reconstruction with persistence threshold of ', num2str(threshold)]);
hold on;

% Plot the reconstructed functions
plot(x_bi_smooth,'cyan');
plot(x_tri_smooth,'magenta');

legend('data', 'C1 smooth reconstruction', 'C8 smooth reconstruction');

% turn off MOSEK optimizers to use MATLAB optimizers again
turn_off_mosek();

cd('../..');
writematrix(x_bi_smooth, 'regularity_score_with_reconstruct1d-x_bi_smooth.txt');
writematrix(x_tri_smooth, 'regularity_score_with_reconstruct1d-x_tri_smooth.txt');
