function ts = getT(out, lab)
    %From the categorization learning group:
    %In ALCOVE, learning is driven by ?teacher? (t) values. 
    %The presence of a category label is represented by a 
    %teacher signal of 1; absence of a category label is 
    %typically represented by -1. The teacher is typically 
    %considered to be ?humble?. This means that if the 
    %output activation is more extreme than the +1/-1 
    %teaching value, then the ouput activation is used 
    %as the teaching signal.
    
    ts = out; %copy output layer activation
    ts(lab == 1 & ts < 1) = 1; %replace correct activations if necessary
    ts(lab == 0 & ~(ts < 0)) = zeros(1, length(ts(lab == 0 & ~(ts < 0)))); %same for incorrect activations.