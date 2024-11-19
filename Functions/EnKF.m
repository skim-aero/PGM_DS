function [estState,ensemble_var] = ...
         EnKF(numStep,n,nm,meas,numEnsembles,...
              estState,ensembles,ensemble_var,ensemble_mean,...
              Q,R,F,H,JH,funsys,funmeas,linear,interm)
        % Ensemble Kalman filter

    % Main loop
    for i = 2:numStep
        if linear
            for e = 1:numEnsembles
                % Prediction step
                ensembles(:,e,i) = F*ensembles(:,e,i-1)+sqrt(Q)*randn(n);
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
            K = ensemble_var(:,:,i)*H'/(H*ensemble_var(:,:,i)*H'+R*eye(nm));   

            for e = 1:numEnsembles
                pert = sqrt(R)*randn(1);
                residual = meas(:,i)+pert-H*ensembles(:,e,i);
                ensembles(:,e,i) = ensembles(:,e,i)+K*residual;
            end
        else
            if interm
                % Prediction step
                for e = 1:numEnsembles
                    ensembles(:,e,i) = funsys(ensembles(:,e,i-1),sqrt(Q)*randn(1),8);
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
                if rem(i,20) == 0
                    K = ensemble_var(:,:,i)*H'/(H*ensemble_var(:,:,i)*H'+R*eye(nm));
                    
                    for e = 1:numEnsembles
                        if ~isempty(JH)
                            H = JH(ensembles(:,e,i));
                            K = ensemble_var(:,:,i)*H'/(H*ensemble_var(:,:,i)*H'+R*eye(nm));
                        end
                           
                        pert = sqrt(R)*randn(1);
                        residual = meas(:,i)+pert-funmeas(ensembles(:,e,i),0);
                        ensembles(:,e,i) = ensembles(:,e,i)+K*residual;
                    end
                end
            else
                % Prediction step
                for e = 1:numEnsembles
                    ensembles(:,e,i) = funsys(ensembles(:,e,i-1),sqrt(Q)*randn(1),i-1);
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
                K = ensemble_var(:,:,i)*H'/(H*ensemble_var(:,:,i)*H'+R*eye(nm));

                for e = 1:numEnsembles
                    if ~isempty(JH)
                        H = JH(ensembles(:,e,i));
                        K = ensemble_var(:,:,i)*H'/(H*ensemble_var(:,:,i)*H'+R*eye(nm));
                    end
                       
                    pert = sqrt(R)*randn(1);
                    residual = meas(:,i)+pert-funmeas(ensembles(:,e,i),0);
                    ensembles(:,e,i) = ensembles(:,e,i)+K*residual;
                end
            end
        end

        estState(:,i) = mean(ensembles(:,:,i),2);
    end
end