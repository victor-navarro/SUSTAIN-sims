function [h_o, i] = clusterCompetition(h_act, beta)
    %h_act is a vector of activated clusters [n x 1]
    %beta is a competition parameter
    
    %get position of the maximally activated cluster
    [~, i] = max(h_act);
    %make holder and apply lateral inhibition to the winning value
    h_o = zeros(size(h_act));
    h_o(i) = (h_act(i)^beta/sum(h_act.^repmat(beta, size(h_act, 1), 1)))*h_act(i);