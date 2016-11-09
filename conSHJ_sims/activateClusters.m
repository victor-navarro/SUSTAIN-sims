function [h_act, ds] = activateClusters(in, h, lambdas, r)
    %in is a one dimensional vector describing the input [1 x n]
    %h is a matrix. each row is has the position of a cluster [c x m] 
    %lambdas is a vector with attentional parameters [1 x l]
    
    ins = repmat(in, size(h, 1), 1);
    %get cluster distance, per dimension (the dimensions and their length
    %are hardcoded, for now.
    ds = [.5*sum(abs(ins(:, 1:2)-h(:, 1:2)), 2), .5*sum(abs(ins(:, 3:4)-h(:, 3:4)), 2), .5*sum(abs(ins(:, 5:6)-h(:, 5:6)), 2), .5*sum(abs(ins(:, 7:12)-h(:, 7:12)), 2)];
    ls = repmat(lambdas, size(h, 1), 1);
    rs = repmat(r, size(ls));
    h_act = sum(ls.^rs.*exp(-ls.*ds), 2)./sum(ls.^rs, 2); %activate clusters