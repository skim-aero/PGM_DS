function [estState,particles,weights] = ...
         SIR(numStep,n,nm,meas,numParticles,estState,particles,weights,...
             Q,R,F,H,funsys,funmeas,linear,numEffparticles,interm)
        % SIR Particle Filter for based on
        % Novel approach to nonlinear/non-Gaussian Bayesian state estimation
        % Gordon's paper

    % Main loop
    for i = 2:numStep
        if linear
            for p = 1:numParticles
                % Prediction step
                particles(:,p,i) = F*particles(:,p,i-1)+sqrt(Q)*randn(n);
                
                % Update step
                residual = meas(:,i)-H*particles(:,p,i)+sqrt(R)*randn(nm);
                weights(p,i) = weights(p,i-1)*1/sqrt(norm(2*pi*R))*exp(-1/2*residual'/R*(residual));
            end
        else
            if interm
                for p = 1:numParticles
                    % Prediction step
                    particles(:,p,i) = funsys(particles(:,p,i-1),sqrt(Q)*randn(1),8);
                    
                    % Update step
                    if rem(i,20) == 0
                        residual = meas(:,i)-funmeas(particles(:,p,i),0);
                        weights(p,i) = weights(p,i-1)*1/sqrt(norm(2*pi*R))*exp(-1/2*residual'/R*(residual));
                    else
                        weights(p,i) = 1/numParticles;
                    end
                end
            else
                for p = 1:numParticles
                    % Prediction step
                    particles(:,p,i) = funsys(particles(:,p,i-1),sqrt(Q)*randn(1),i-1);
                    
                    % Update step
                    residual = meas(:,i)-funmeas(particles(:,p,i),sqrt(R)*randn(nm));
                    weights(p,i) = weights(p,i-1)*1/sqrt(norm(2*pi*R))*exp(-1/2*residual'/R*(residual));
                end
            end
        end
        weights(:,i) = weights(:,i)/sum(weights(:,i));  % normalisation
        
        % Resampling
        [particles(:,:,i),weights(:,i)] = resample_eff(weights(:,i),particles(:,:,i),numEffparticles);
        
        estState(:,i) = mean(particles(:,:,i),2);
    end
end