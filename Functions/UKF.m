function [estState,x,P] = ...
         UKF(numStep,n,nm,meas,estState,x,P,numSigma,alpha,beta,lambda,...
             Q,R,F,H,G,funsys,funmeas,interm)
        % UKF

    % Initialisation
    xs = zeros(n,numSigma,numStep);
    zs = zeros(nm,numSigma,numStep);
    wm = zeros(numSigma);
    wc = zeros(numSigma);
    xn = zeros(n,1);
    Pn = zeros(n);
    zn = zeros(nm,1);
    Pzz = zeros(nm);
    Pxz = zeros(n,nm);

    % Sigma points
    wm(1) = lambda/(n+lambda);
    wc(1) = lambda/(n+lambda)+(1-alpha^2+beta);
    
    for j = 2:numSigma   
        wm(j) = 1/(2*(n+lambda));
        wc(j) = wm(j);
    end

    % Main loop
    for i = 2:numStep
        % Sigma points
        try
            sqrt_P = chol((n+lambda)*P,'lower');        % square root of scaled covariance
        catch
            P = nearestSPD(P)+1e-6*eye(n);              % if Cholesky fails, add regularisation and retry
            sqrt_P = chol((n+lambda)*P,'lower');        % square root of scaled covariance
        end

        xs(:,1,i) = x;

        for j = 1:n
            xs(:,j+1,i) = x+sqrt_P(:,j);
            xs(:,j+n+1,i) = x-sqrt_P(:,j);
        end

        % Prediction
        for j = 1:numSigma
            if ~isempty(F)
                xs(:,j,i) = F*xs(:,j,i);
            else
                xs(:,j,i) = funsys(xs(:,j,i),0,i-1,8);
            end
            xn = xn+wm(j)*xs(:,j,i);
        end
        
        for j = 1:numSigma
            Pn = Pn+wc(j)*(xs(:,j,i)-xn)*(xs(:,j,i)-xn)';
        end
        Pn = Pn+G*Q*G';
    
        % Update
        if ~isempty(interm)
            if rem(i,interm) == 0        
                for j = 1:numSigma
                    if ~isempty(H)
                        zs(:,j,i) = H*xs(:,j,i);
                    else
	                    zs(:,j,i) = funmeas(xs(:,j,i),0);
                    end
                    zn = zn+wm(j)*zs(:,j,i);
                end
        
                for j = 1:numSigma
                    Pzz = Pzz+wc(j)*(zs(:,j,i)-zn)*(zs(:,j,i)-zn)';
                end
                Pzz = Pzz+R*eye(nm);
            
                for j = 1:numSigma
                    Pxz = Pxz+wc(j)*(xs(:,j,i)-xn)*(zs(:,j,i)-zn)';
                end
            
                K = Pxz/Pzz;
                x = xn+K*(meas(:,i)-zn);
                P = Pn-K*Pzz*K';
            else
                P = Pn;
                x = xn;            
            end
        else
            for j = 1:numSigma
                if ~isempty(H)
                    zs(:,j,i) = H*xs(:,j,i);
                else
                    zs(:,j,i) = funmeas(xs(:,j,i),0);
                end
                zn = zn+wm(j)*zs(:,j,i);
            end
    
            for j = 1:numSigma
                Pzz = Pzz+wc(j)*(zs(:,j,i)-zn)*(zs(:,j,i)-zn)';
            end
            Pzz = Pzz+R*eye(nm);
        
            for j = 1:numSigma
                Pxz = Pxz+wc(j)*(xs(:,j,i)-xn)*(zs(:,j,i)-zn)';
            end
        
            K = Pxz/Pzz;
            x = xn+K*(meas(:,i)-zn);
            P = Pn-K*Pzz*K';
        end

        estState(:,i) = x;

        % Initialisation
        xn = zeros(n,1);
        Pn = zeros(n);
        zn = zeros(nm,1);
        Pzz = zeros(nm);
        Pxz = zeros(n,nm);
    end
end