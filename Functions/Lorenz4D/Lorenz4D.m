function X = Lorenz4D(tf,Q,initial,Param_F)
    % X (txJ) = lorenz4D(tf,noise,initial,forcing)
    % ensure final time is divisible by h
    
    %%% the Lorenz model is: (cyclical)
    % dX[j]/dt=(X[j+1]-X[j-2])*X[j-1]-X[j]+F
    J = 40;               %the number of variables
    h = 0.05;             %the time step
    
    %initialize
    t = 0:h:tf; 
    t = t';
    X = zeros(J,length(t));
    
    % the original conditions (steady state)
    % X(:,1) = F.*ones(J,1);
    X(:,1) = initial;
    
    % the perturbation in 1
    X(1,1) = Param_F + 10E-4;
    
    for i = 1:length(t) - 1 % for each time
        X(:,i+1) = X(:,i) + rk4(X(:,i),h,Param_F); % solved via RK4
    end
    
    % Noise = sqrt(Q)*randn(J,length(t));
    % 
    % X = X + Noise;
end

function deltay = rk4(Xold,h,F)
    % X[t+1] = rk4(X[t],step)
    k1 = f(Xold,F);
    k2 = f(Xold + 1/2.*h.*k1,F);
    k3 = f(Xold + 1/2.*h.*k2,F);
    k4 = f(Xold + h.*k3,F);
    deltay = 1/6*h*(k1 + 2*k2 + 2*k3 + k4);
end

function k = f(X,F)
    J = 40;
    k = zeros(J,1);
    
    % first the 3 problematic cases: 1,2,J
    k(1) = (X(2) - X(J - 1))*X(J) - X(1);
    k(2) = (X(3) - X(J))*X(1) - X(2);
    k(J) = (X(1) - X(J - 2))*X(J - 1) - X(J);
    
    % then the general case
    for j = 3:J - 1
        k(j) = (X(j + 1) - X(j - 2)).*X(j - 1) - X(j);
    end
    
    % add the F    
    k = k + F;
end