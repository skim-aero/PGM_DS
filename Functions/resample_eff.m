function [newParticle_state,w] = resample_eff(w,oldParticle_state,threshold)
    N = length(w);

    % Compute effective sample size
    Neff = 1/sum(w.^2);
    
    if isempty(threshold)
        % Threshold for resampling, e.g., N/2
        threshold = N/2;
    end
        
    if Neff >= threshold
        newParticle_state = oldParticle_state;
    else
        newParticle_state = multinomial_resample(w,oldParticle_state);
        w = ones(1,N)/N;  % reset weights after resampling
    end
end

function newParticle_state = multinomial_resample(w,oldParticle_state)
    N = length(w);
    n = size(oldParticle_state, 1);
    newParticle_state = zeros(n,N);
    
    % Compute cumulative sum of weights
    cumulativeWeights = cumsum(w);

    % Perform multinomial resampling
    for i = 1:N
        u = rand; % Uniform random number between 0 and 1
        index = find(cumulativeWeights >= u, 1, 'first');
        if isempty(index)
            newParticle_state(:,i) = oldParticle_state(:,1);
        else
            newParticle_state(:,i) = oldParticle_state(:,index);
        end
    end
end
