function [AKKF] = AKKF_proposal(AKKF, n)

%% Estimation
AKKF.X_est(:,n) = AKKF.X_P(:,:,n) * AKKF.W_plus(:,n); %state mean
AKKF.X_C(:,:,n) =   AKKF.X_P(:,:,n) * AKKF.S_plus(:,:,n) * AKKF.X_P(:,:,n).'; %state covariance

%% Proposal partice draw
AKKF.X_P_proposal(:,:,n) = mgd(AKKF.N_P,length(AKKF.X_est(:,n)), AKKF.X_est(:,n), AKKF.X_C(:,:,n)).'; % Maybe here....

end

