function [estState,ensemble_var] = ...
         EnKF(numStep,n,nm,meas,numEnsembles,...
              estState,ensembles,ensemble_var,ensemble_mean,...
              Q,R,F,H,JH,funsys,funmeas,interm)
        % Ensemble Kalman filter

    % Main loop
    for i = 2:numStep
        % Prediction step
        if ~isempty(F)
            for e = 1:numEnsembles
                ensembles(:,e,i) = F*ensembles(:,e,i-1)+sqrt(Q)*randn(n);
            end
        else
            for e = 1:numEnsembles
                ensembles(:,e,i) = funsys(ensembles(:,e,i-1),sqrt(Q)*randn(1),i-1,8);
            end
        end

        ensemble_mean(:,i) = mean(ensembles(:,:,i),2);

        for j = 1:n
            for k = 1:n
                if j == k
                    ensemble_var(j,k,i) = var(ensembles(j,:,i));
                else
                    temp = cov(ensembles(j,:,i),ensembles(k,:,i));
                    ensemble_var(j,k,i) = temp(1,2);
                end
            end
        end

        % Update step
        if ~isempty(interm)
            if rem(i,interm) == 0
                [ensembles,ensemble_var] = update(i,nm,meas,numEnsembles,ensembles,ensemble_var,R,H,JH,funmeas);
            end
        else
            [ensembles,ensemble_var] = update(i,nm,meas,numEnsembles,ensembles,ensemble_var,R,H,JH,funmeas);
        end

        estState(:,i) = mean(ensembles(:,:,i),2);
    end
end

function [ensembles,ensemble_var] = update(i,nm,meas,numEnsembles,ensembles,ensemble_var,R,H,JH,funmeas)
    if isempty(funmeas) && isempty(JH)
        K = ensemble_var(:,:,i)*H'/(H*ensemble_var(:,:,i)*H'+R*eye(nm));

        for e = 1:numEnsembles
            pert = sqrt(R)*randn(1);
            residual = meas(:,i)+pert-H*ensembles(:,e,i);
            ensembles(:,e,i) = ensembles(:,e,i)+K*residual;
        end
    elseif ~isempty(funmeas) && isempty(JH)
        K = ensemble_var(:,:,i)*H'/(H*ensemble_var(:,:,i)*H'+R*eye(nm));
        
        for e = 1:numEnsembles
            pert = sqrt(R)*randn(1);
            residual = meas(:,i)+pert-funmeas(ensembles(:,e,i),0);
            ensembles(:,e,i) = ensembles(:,e,i)+K*residual;
        end
    elseif ~isempty(funmeas) && ~isempty(JH)
        for e = 1:numEnsembles
            H = JH(ensembles(:,e,i));
            K = ensemble_var(:,:,i)*H'/(H*ensemble_var(:,:,i)*H'+R*eye(nm));
               
            pert = sqrt(R)*randn(1);
            residual = meas(:,i)+pert-funmeas(ensembles(:,e,i),0);
            ensembles(:,e,i) = ensembles(:,e,i)+K*residual;    
        end
    else
        for e = 1:numEnsembles
            H = JH(ensembles(:,e,i));
            K = ensemble_var(:,:,i)*H'/(H*ensemble_var(:,:,i)*H'+R*eye(nm));
               
            pert = sqrt(R)*randn(1);
            residual = meas(:,i)+pert-H*ensembles(:,e,i);
            ensembles(:,e,i) = ensembles(:,e,i)+K*residual;  
        end
    end
end