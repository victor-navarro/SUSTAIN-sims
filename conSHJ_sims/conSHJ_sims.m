%Shepard, Hovland and Jenkins SUSTAIN sims.

%cd('/Users/edwardwasserman/Google Drive/Wasserman''s Lab/MATLAB/SUSTAIN/conSHJ_sims/')
%clc
%clear all
%close all

%set custom string for filenames (only for saving purposes)
custom = '';

%choose to save the data or not
save_data = 0;

%set number of iterations
iterations = 2;

%load category structures from SHJ
load SHJ_cats


o_set = [];
labs = eye(12);
for t = 1:6
    raw = cats(t).stim(:, :, 1); %get stimuli for the type
    stim_h = zeros(8, 25); %preallocate original training set. Each stimulus has 3+1 dimensions and each can take two dimensional values.
    %transform (or collapse) stimuli, into the input vectors used by SUSTAIN
    for s = 1:8 %cycle through stims
        for d = 1:3 %cycle through dimensions
            if ~raw(d, s)
                stim_h(s, (d*2)-1) = 1; %the first unit of each dimension sector represents the dimensional value 0
            else
                stim_h(s, d*2) = 1;
            end
        end
        %assign category output units
        if ~raw(4, s)
            stim_h(s, 13:24) = labs(2*(t-1)+1, :);
        else
            stim_h(s, 13:24) = labs(2*(t-1)+2, :);
        end
    end
    %put contexts
    stim_h(:, 6+t) = ones(8, 1);
    %put label
    stim_h(:, 25) = repmat(t, 8, 1);
    o_set = [o_set; stim_h];
end
o_set = [o_set; o_set]; %duplicate original training set to reach 96 trials

%Here's a quick description of the training matrix
%cols 1:6 refer to the dimensions of the target stimulus
%cols 7:12 refer to the background (six different backgrounds are used)
%cols 13:24 refer to the output assignation (12 different buttons are used,
%with pairs of buttons being exclusive to each type of category).
%col 25 labels the type. This is used for data-saving purposes

%set attentional focus parameter
r = 9.01245; 

%set competition parameter
beta = 1.252233;

%set decision consistency parameter
d = 16.924073;

%set learning rate parameter
eta = 0.092327;


%intialize holder for data
iter_data = [];

%set attentional parameter for context (default should be 1)
ctx_l = 1;

%set attentional parameter for all dimensions
o_lambdas = [1 1 1 ctx_l];



for iter = 1:iterations
    fprintf('Iteration: %d\n', iter)
    
    %repeat and shuffle training schedule
    indices = zeros(size(o_set, 1)*100, 1);
    
    for x = 1:sessions
        indices(96*(x-1)+1:96*(x-1)+96, 1) = randperm(96);
    end
    t_set = o_set(indices, :); %shuffle training set

    h = []; %initialize an empty hidden layer
    
    %copy original attentional weights
    lambdas = o_lambdas;

    %initialize an empty matrix to save data from iteration
    sim_data = [];
    
    %do the first trial separately to avoid comparing on every trial
    in = t_set(1, 1:12); %select input vector
    h(1, :) = in;
    c_type = t_set(1, 25); %get cat_type
    w = zeros(1, 12);
    [h_act, ds] = activateClusters(in, h, lambdas, r); %activate clusters, ds contains the dimensional distances (cols) for each cluster (rows)
    [h_o, i] = clusterCompetition(h_act, beta); %make clusters compete, i is the index of the winning cluster
    out = h_o'*w; %activate the queried output units
    choice_act = out((1:2)+(c_type-1)*2); %get the relevant choices
    ps = exp(repmat(d, 1, 2).*choice_act)./sum(exp(repmat(d, 1, 2).*choice_act)); %calculate the probability of making the response associated to each output unit
    ts = getT(out, t_set(1, 13:24)); %get teaching signal
    h(i, :) = h(i, :) + eta*(in-h(i, :)); %adjust the position of the winner
    lambdas = lambdas + eta*exp(-lambdas.*ds(i, :)).*(1-lambdas.*ds(i, :)); %adjust the attentional weights, based on the winner
    w(i, :) = w(i, :) + eta*(ts - out)*h_o(i); %adjust weights of the winner
    
    %get clusters per type, after training (based on weights, VERY
    %SLOW)
    clus = zeros(1, 6);
    for clu = 1:6
        clus(clu) = length(find(w(:, 2*(clu-1)+1:2*(clu-1)+2)));
    end
    
    sim_data(1, :) = [1, size(h, 1), ps(find(ts((1:2)+(c_type-1)*2))), c_type, clus]; %save data
    
    t = 2;
    
    while t < size(t_set, 1)+1
        in = t_set(t, 1:12);
        c_type = t_set(t, 25); %get cat_type
        [h_act, ds] = activateClusters(in, h, lambdas, r);
        [h_o, i] = clusterCompetition(h_act, beta);
        out = h_o'*w;
        choice_act = out((1:2)+(c_type-1)*2);
        ps = exp(repmat(d, 1, 2).*choice_act)./sum(exp(repmat(d, 1, 2).*choice_act)); %calculate the probability of making the response associated to each output unit
        ts = getT(out, t_set(t, 13:24));
        [~, j] = max(out);
        [~, k] = max(ts);
        if j ~= k %reclute a new cluster if the maximally activated units do not correspond between output and teaching signal
            h(size(h, 1)+1, :) = in(1, :); %create cluster on the coordinates of the mispredicted input
            w(size(w, 1)+1, :) = zeros(1, 12); %initialize its weights to zero
            [h_act, ds] = activateClusters(in, h, lambdas, r);
            [h_o, i] = clusterCompetition(h_act, beta);
            out = h_o'*w;
            ts = getT(out, t_set(t, 13:24));
        end
        h(i, :) = h(i, :) + eta*(in-h(i, :));
        lambdas = lambdas + eta*exp(-lambdas.*ds(i, :)).*(1-lambdas.*ds(i, :));
        w(i, :) = w(i, :) + eta*(ts - out)*h_o(i);
        clus = zeros(1, 6);
        for clu = 1:6
            clus(clu) = length(find(w(:, 2*(clu-1)+1:2*(clu-1)+2)));
        end
        sim_data(t, :) = [t, size(h, 1), ps(find(ts((1:2)+(c_type-1)*2))), c_type, clus];
        t = t+1; %move to next trial
    end
    
    
    iter_data = [iter_data; repmat(iter, size(sim_data, 1), 1), sim_data]; %save data
end

%END OF THE SIMULATION%

%get mean accuracy across sessions
h1 = zeros(6, sessions);
for t = 1:6
    for s = 1:sessions
        h1(t, s) = mean(iter_data(ceil(iter_data(:, 2)/96) == s & iter_data(:, 5) == t, 4));
    end
end

%get clusters across sessions
h2 = zeros(1, sessions);
for s = 1:sessions
    h2(s) = mean(iter_data(ceil(iter_data(:, 2)/96) == s, 3));
end

%get cluster counts
h3 = zeros(1, iterations);
for it = 1:iterations
    h3(it) = max(iter_data(iter_data(:, 1) == it, 3));
end

%get clusters per type, after training (based on weights); not very
%informative if done over trials
h4 = zeros(1, 6);
for t = 1:6
        h4(t) = mean(iter_data(iter_data(:, 5) == t, 5+t));
end

%plot stuff
x = figure('Name', sprintf('Simulation results (%d iterations)', iterations), 'Visible', 'on');
subplot(2, 2, 1)
plot(1-h1');
legend({'T1', 'T2', 'T3', 'T4', 'T5', 'T6'}, 'Location', 'NorthEast');
ylabel('Probability of error');
xlabel('Learning session')

subplot(2, 2, 2)
plot(h2)
ylabel('Number of clusters');
xlabel('Learning session')

subplot(2, 2, 3)
hist(h3);
ylabel('Count');
xlabel('Total clusters')

subplot(2, 2, 4)
bar(h4)
ylabel('Number of clusters');
xlabel('Task type')


if(save_data)
    %save plot
    saveas(x, sprintf('./plots/%sConcurrentSHJ_%d_iterations_%s.jpg', custom, iterations, date))

    %save data
    full_name = sprintf('./data/%sConcurrentSHJ_%d_iterations_full_%s.mat', custom, iterations, date);
    
    save(full_name,'iter_data');
end

