function [estState,x,P] = ...
         UKF(numStep,n,nm,meas,estState,x,P,numSigma,alpha,beta,lambda,...
             Q,R,G,funsys,funmeas,interm)
        % UKF

    % Initialisation
    xs = zeros(n,numSigma,numStep);
    zs = zeros(nm,numSigma,numStep);
    wm = zeros(numSigma);
    wc = zeros(numSigma);
    xn = zeros(n,1);
    Pn = zeros(n);
    zn = zeros(nm,1);
    Pz = zeros(nm);
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
        sqrt_P = chol((n+lambda)*P,'lower'); % Square root of scaled covariance
        xs(:,1,i) = x;

        for j = 1:n
            xs(:,j+1,i) = x+sqrt_P(:,j);
            xs(:,j+n+1,i) = x-sqrt_P(:,j);
        end

        % Prediction
        for j = 1:numSigma
            xs(:,j,i) = funsys(xs(:,j,i),0,i-1);
            xn = xn+wm(j)*xs(:,j,i);
        end
        
        for j = 1:numSigma
            Pn = Pn+wc(j)*(xs(:,j,i)-xn)*(xs(:,j,i)-xn)';
        end
        Pn = Pn+G*Q*G';
    
        % Update
        if interm
            disp("Fuck?")
            if rem(i,20) == 0        
                for j = 1:numSigma
	                zs(:,j,i) = funmeas(xs(:,j,i),0);
                    zn = zn+wm(j)*zs(:,j,i);
                end
        
                for j = 1:numSigma
                    Pz = Pz+wc(j)*(zs(:,j,i)-zn)*(zs(:,j,i)-zn)';
                end
                Pz = Pz+R*eye(nm);
            
                for j = 1:numSigma
                    Pxz = Pxz+wc(j)*(xs(:,j,i)-xn)*(zs(:,j,i)-zn)';
                end
            
                K = Pxz/Pz;
                x = xn+K*(meas(:,i)-zn);
                P = Pn-K*Pz*K';
            else
                P = Pn;
                x = xn;            
            end
        else
            for j = 1:numSigma
	            zs(:,j,i) = funmeas(xs(:,j,i),0);
                zn = zn+wm(j)*zs(:,j,i);
            end
    
            for j = 1:numSigma
                Pz = Pz+wc(j)*(zs(:,j,i)-zn)*(zs(:,j,i)-zn)';
            end
            Pz = Pz+R*eye(nm);
        
            for j = 1:numSigma
                Pxz = Pxz+wc(j)*(xs(:,j,i)-xn)*(zs(:,j,i)-zn)';
            end
        
            K = Pxz/Pz;
            x = xn+K*(meas(:,i)-zn);
            P = Pn-K*Pz*K';
        end

        % Covariance correction tricks
        P = (P+P.')/2;
        
        % Check and correct eigenvalues if necessary
        while any(eig(P) <= 1e-8)
            P = P+0.1*eye(n);
        end

        estState(:,i) = x;

        % Initialisation
        xn = zeros(n,1);
        Pn = zeros(n);
        zn = zeros(nm,1);
        Pz = zeros(nm);
        Pxz = zeros(n,nm);
    end
end