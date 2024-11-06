function [AKKF] = AKKF_update(meas, AKKF, R, n)

%% Data space
% for p = 1:AKKF.N_P  
%     AKKF.Z_P(:,p,n) = H*AKKF.X_P(:,p,n)+sqrt(R)*randn(length(H*AKKF.X_P(:,p,n)),1);
% end 

%% Kernel Space
if AKKF.kernel == 1 %Quadratic
    G_yy = (AKKF.Z_P(:,:,n).' * AKKF.Z_P(:,:,n)/AKKF.poly_para_b + AKKF.c * ones(AKKF.N_P, AKKF.N_P)).^2;
    g_y(:,n) = (AKKF.Z_P(:,:,n).' * meas(:,n)/AKKF.poly_para_b + AKKF.c * ones(AKKF.N_P, 1)).^2;
elseif  AKKF.kernel == 2  %Quartic
    G_yy = (AKKF.Z_P(:,:,n).' * AKKF.Z_P(:,:,n)/AKKF.poly_para_b + AKKF.c * ones(AKKF.N_P, AKKF.N_P)).^4;
    g_y(:,n) = (AKKF.Z_P(:,:,n).' * meas(:,n)/AKKF.poly_para_b + AKKF.c * ones(AKKF.N_P, 1)).^4;
else %Gaussian
    G_yy = exp(-(AKKF.Z_P(:,:,n).' - AKKF.Z_P(:,:,n)).^2/(2 * AKKF.Var_Gaussian))/(sqrt(2 * pi * AKKF.Var_Gaussian));
    g_y(:,n)  = exp(-(AKKF.Z_P(:,1,n).' - meas(:,n)).^2/(2 * AKKF.Var_Gaussian))/(sqrt(2 * pi * AKKF.Var_Gaussian));
%     for dim = 1:size(meas(:,n))
%         G_yy(:,:,dim) = exp(-(AKKF.Z_P(dim,:,n).' - AKKF.Z_P(dim,:,n)).^2/(2 * AKKF.Var_Gaussian))/(sqrt(2 * pi * AKKF.Var_Gaussian));
%     end
%     for p = 1:AKKF.N_P  
%         g_y(:,p)  = exp(-(AKKF.Z_P(:,p,n) - meas(:,n)).^2/(2 * AKKF.Var_Gaussian))/(sqrt(2 * pi * AKKF.Var_Gaussian));
%     end
end

Q_KKF = AKKF.S_minus(:,:,n) * inv(G_yy * AKKF.S_minus(:,:,n) + AKKF.lambda_KKF * eye(AKKF.N_P));
AKKF.W_plus(:,n) = AKKF.W_minus(:,n) + Q_KKF * (g_y(:,n) - G_yy * AKKF.W_minus(:,n));
AKKF.S_plus(:,:,n) = AKKF.S_minus(:,:,n) - Q_KKF * G_yy * AKKF.S_minus(:,:,n);

% Q_KKF = AKKF.S_minus(:,:,n) * inv(G_yy(:,:,1) * AKKF.S_minus(:,:,n) + AKKF.lambda_KKF * eye(AKKF.N_P));
% AKKF.W_plus(:,n) = AKKF.W_minus(:,n) + Q_KKF * (g_y(1,:)' - G_yy(:,:,1) * AKKF.W_minus(:,n)); %%%% ???????
% AKKF.S_plus(:,:,n) = AKKF.S_minus(:,:,n) - Q_KKF * G_yy(:,:,1) * AKKF.S_minus(:,:,n);

end
