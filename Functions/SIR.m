function [estState,particles,weights] = ...
         SIR(numStep,n,nm,meas,numParticles,estState,particles,weights,...
             Q,R,F,H,funsys,funmeas,numEffparticles,interm)
        % SIR Particle Filter for based on
        % Novel approach to nonlinear/non-Gaussian Bayesian state estimation
        % Gordon's paper

    % Main loop
    for i = 2:numStep
        % Prediction step
        if ~isempty(F)
            for p = 1:numEnsembles
                particles(:,p,i) = F*particles(:,p,i)+sqrt(Q)*randn(n);
            end
        else
            for p = 1:numParticles
                particles(:,p,i) = funsys(particles(:,p,i-1),sqrt(Q)*randn(1),i-1,8);
            end
        end

        % Update step
        if ~isempty(interm)    
            if rem(i,interm) == 0
                if ~isempty(H)
                    for p = 1:numParticles
                        residual = meas(:,i)-H*particles(:,p,i);
                        weights(p,i) = weights(p,i-1)*1/sqrt(norm(2*pi*R))*exp(-1/2*residual'/R*(residual));
                    end
                else
                    for p = 1:numParticles
                        residual = meas(:,i)-funmeas(particles(:,p,i),0);
                        weights(p,i) = weights(p,i-1)*1/sqrt(norm(2*pi*R))*exp(-1/2*residual'/R*(residual));
                    end
                end
            else
                weights(:,i) = 1/numParticles*ones(numParticles,1);
            end
        else
            if ~isempty(H)
                for p = 1:numParticles
                    residual = meas(:,i)-H*particles(:,p,i);
                    weights(p,i) = weights(p,i-1)*1/sqrt(norm(2*pi*R))*exp(-1/2*residual'/R*(residual));
                end
            else
                for p = 1:numParticles
                    residual = meas(:,i)-funmeas(particles(:,p,i),sqrt(R)*randn(nm));
                    weights(p,i) = weights(p,i-1)*1/sqrt(norm(2*pi*R))*exp(-1/2*residual'/R*(residual));
                end
            end
        end

        if all(weights(:,i) == 0)
            weights(:,i) = 1/numParticles;
        end

        weights(:,i) = weights(:,i)/sum(weights(:,i));  % normalisation

        % Resampling
        [particles(:,:,i),weights(:,i)] = resampleEff(weights(:,i),particles(:,:,i),numEffparticles);

        estState(:,i) = particles(:,:,i)*weights(:,i);
    end
end