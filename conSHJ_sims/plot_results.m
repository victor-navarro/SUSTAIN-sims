%Shepard, Hovland and Jenkins sims.
cd('/Users/edwardwasserman/Google Drive/Wasserman''s Lab/MATLAB/SUSTAIN/')
clc
clear all
close all


data = [];
for t = 1:6
    load(sprintf('./data/T%d_1000_iterations_mean_08-Nov-2016.mat', t));
    %holder(:, 4) = repmat(t, size(holder, 1), 1);
    data = [data, h1(:, 3), ];
    %data = [data; holder];
end

x = figure;
plot(1-data, '-o')
legend({'T1', 'T2', 'T3', 'T4', 'T5', 'T6'})
xlabel('Learning block')
ylabel('Probability of error')
saveas(x, './plots/overall_07-Nov-2016.jpg');