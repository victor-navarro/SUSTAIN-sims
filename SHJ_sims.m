%Shepard, Hovland and Jenkins SUSTAIN sims.

cd('/Users/edwardwasserman/Google Drive/Wasserman''s Lab/MATLAB/SUSTAIN/')
clc
clear all
close all

%set custom string for filenames (only for saving purposes)
custom = '';

%choose to save the data or not
save_data = 1;

%load category structures from SHJ
load SHJ_cats

cat_type = 1; %select category type (1?6)
arr = randi(size(cats(cat_type).stim, 3)); %pick arrangement at random just because we can
raw = cats(cat_type).stim(:, :, arr); %get stimuli
o_set = zeros(8, 8); %preallocate original training set. Each stimulus has 3+1 dimensions and each can take two dimensional values.

%transform (or collapse) stimuli, into the input vectors used by SUSTAIN
for s = 1:8 %cycle through stims
    for d = 1:4 %cycle through dimensions
        if ~raw(d, s)
            o_set(s, (d*2)-1) = 1; %the first unit of each dimension sector represents the dimensional value 0
        else
            o_set(s, d*2) = 1;
        end
    end
end

%set attentional focus parameter
r = 9.01245; 

%set competition parameter
beta = 1.252233;

%set decision consistency parameter
d = 16.924073;

%set learning rate parameter
eta = 0.092327;

%set number of iterations
iterations = 10;

%set number of training blocks
blocks = 16;

%intialize holder for data
iter_data = [];

for iter = 1:iterations
    fprintf('Iteration: %d\n', iter)
    
    %repeat and shuffle training schedule
    t_set = repmat(o_set, blocks*2, 1);
    indices = zeros(size(t_set, 1), 1);
    
    for x = 1:blocks
        if x == 1
            indices(1:8) = randperm(8);
            indices(9:16) = randperm(8);
        else
            indices(16*(x-1)+1:16*(x-1)+16, 1) = randperm(16);
        end
        
    end
    do_set = repmat(o_set, 2, 1); %double o_set, to create the super blocks from nosofsky
    
    t_set = t_set(indices, :); %shuffle training set
    
    h = []; %initialize an empty hidden layer (cluster coordinates)
    
    %set attentional weights
    lambdas = [1 1 1];

    %initialize an empty matrix to save data from iteration
    sim_data = [];
    
    %do the first trial separately to avoid comparing on every trial
    in = t_set(1, 1:6); %select input vector
    h(1, :) = in;
    w = zeros(1, 2);
    [h_act, ds] = activateClusters(in, h, lambdas, r); %activate clusters, ds contains the dimensional distances (cols) for each cluster (rows)
    [h_o, i] = clusterCompetition(h_act, beta); %make clusters compete, i is the index of the winning cluster
    out = h_o'*w; %activate the queried output units
    ps = exp(repmat(d, size(out)).*out)./sum(exp(repmat(d, size(out)).*out)); %calculate the probability of making the response associated to each output unit
    ts = getT(out, t_set(1, 7:8)); %get teaching signal
    h(i, :) = h(i, :) + eta*(in-h(i, :)); %adjust the position of the winner
    lambdas = lambdas + eta*exp(-lambdas.*ds(i, :)).*(1-lambdas.*ds(i, :)); %adjust the attentional weights, based on the winner
    w(i, :) = w(i, :) + eta*(ts - out)*h_o(i); %adjust weights of the winner
    sim_data(1, :) = [1, size(h, 1), ps(find(t_set(1, 7:8)))]; %save data
    t = 2;
    
    while t < size(t_set, 1)+1
        in = t_set(t, 1:6);
        [h_act, ds] = activateClusters(in, h, lambdas, r);
        [h_o, i] = clusterCompetition(h_act, beta);
        out = h_o'*w;
        ps = exp(repmat(d, size(out)).*out)./sum(exp(repmat(d, size(out)).*out));
        ts = getT(out, t_set(t, 7:8));
        [~, j] = max(out);
        [~, k] = max(ts);
        if j ~= k %reclute a new cluster if the maximally activated units do not correspond between output and teaching signal
            h(size(h, 1)+1, :) = in(1, :); %create cluster on the coordinates of the mispredicted input
            w(size(w, 1)+1, :) = zeros(1, 2); %initialize its weights to zero
            [h_act, ds] = activateClusters(in, h, lambdas, r);
            [h_o, i] = clusterCompetition(h_act, beta);
            out = h_o'*w;
            ts = getT(out, t_set(t, 7:8));
        end
        h(i, :) = h(i, :) + eta*(in-h(i, :));
        lambdas = lambdas + eta*exp(-lambdas.*ds(i, :)).*(1-lambdas.*ds(i, :));
        w(i, :) = w(i, :) + eta*(ts - out)*h_o(i);
        sim_data(t, :) = [t, size(h, 1), ps(k)];
        t = t+1; %move to next trial
    end
    
    iter_data = [iter_data; repmat(iter, size(sim_data, 1), 1), sim_data]; %save data
end

%END OF THE SIMULATION%

%get means
h1 = zeros(8, 3);
for t = 1:blocks
    h1(t, :) = [t, mean(iter_data(ceil(iter_data(:, 2)/16) == t, 3)), mean(iter_data(ceil(iter_data(:, 2)/16) == t, 4))];
end

%get counts
h2 = zeros(iterations, 2);
for it = 1:iterations
    h2(it, :) = [it, max(iter_data(iter_data(:, 1) == it, 3))];
end

%plot stuff
x = figure('Name', sprintf('Simulation results (%d iterations)', iterations), 'Visible', 'on');
subplot(2, 2, 1)
plotSHJ(raw, cat_type);
subplot(2, 2, 2)
plot(h1(:, 1), 1-h1(:, 3));
ylabel('Probability of error');
xlabel('Learning block')
subplot(2, 2, 3)
plot(h1(:, 1), h1(:, 2));
ylabel('Average number of clusters')
xlabel('Learning block')
subplot(2, 2, 4)
hist(h2(:, 2));
ylabel('Count');
xlabel('Clusters')

if(save_data)
    saveas(x, sprintf('./plots/%s(T%d)_%d_iterations_%s.jpg', custom, cat_type, iterations, date))

    %save data
    full_name = sprintf('./data/%sT%d_%d_iterations_full_%s.mat', custom, cat_type, iterations, date);
    mean_name = sprintf('./data/%sT%d_%d_iterations_mean_%s.mat', custom, cat_type, iterations, date);
    counts_name = sprintf('./data/%sT%d_%d_iterations_counts_%s.mat', custom, cat_type, iterations, date);

    save(full_name,'iter_data');
    save(mean_name,'h1');
    save(counts_name,'h2');
end

