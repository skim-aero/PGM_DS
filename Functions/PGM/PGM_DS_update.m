function [particles,estState,temp_particle_var,...
          cmean,ccovar,cweight] = ...
                       PGM_DS_update(i,n,nm,numStep,numParticles,...
                                    numMixture,numMixture_max,...
                                    ifupd,cweight,idx,meas,likelih,R,...
                                    particles,particle_clust,...
                                    particle_meas,particle_mean,...
                                    particle_var,estState,temp_particle_var,...
                                    ifut,numSigma,lambda,wc,wm,funmeas)

    residual = zeros(nm,numMixture_max);
    cmean = zeros(n,numMixture);
    ccovar = zeros(n,n,numMixture);

    if ifupd
        if ifut
            xs = zeros(n,numSigma,numMixture_max);
            zs = zeros(nm,numSigma,numMixture_max);

            for clst = 1:numMixture
                xn = zeros(n,1);
                zn = zeros(nm,1);
                Pn = zeros(n,n);
                Pxz = zeros(n,nm);
                Pzz = zeros(nm,nm);

                try
                    sqrt_P = chol((n+lambda)*particle_var(:,:,clst,i),'lower');         % square root of scaled covariance
                catch
                    particle_var(:,:,clst,i) = nearestSPD(particle_var(:,:,clst,i))+1e-6*eye(n);   % if Cholesky fails, add regularisation and retry
                    sqrt_P = chol((n+lambda)*particle_var(:,:,clst,i),'lower');         % square root of scaled covariance
                end

                % sqrt_P = chol((n+lambda)*particle_var(:,:,clst,i),'lower'); % Square root of scaled covariance
                xs(:,1,clst) = particle_mean(:,clst,i);

                % Sigma points
                for j = 1:n
                    xs(:,j+1,clst) = particle_mean(:,clst,i)+sqrt_P(:,j);
                    xs(:,j+n+1,clst) = particle_mean(:,clst,i)-sqrt_P(:,j);
                end

                for j = 1:numSigma
                    xn(:) = xn(:)+wm(j)*xs(:,j,clst);
                end

                for j = 1:numSigma
                    Pn(:,:) = Pn(:,:)+wc(j)*(xs(:,j,clst)-xn(:))*(xs(:,j,clst)-xn(:))';
                end

                % Measurements update
                for j = 1:numSigma
                    zs(:,j,clst) = funmeas(xs(:,j,clst),0);
                    zn(:) = zn(:)+wm(j)*zs(:,j);
                end

                for j = 1:numSigma
                    Pxz(:,:) = Pxz(:,:)+wc(j)*(xs(:,j,clst)-xn(:))*(zs(:,j,clst)-zn(:))';
                    Pzz(:,:) = Pzz(:,:)+wc(j)*(zs(:,j,clst)-zn(:))*(zs(:,j,clst)-zn(:))';
                end
                Pzz(:,:) = Pzz(:,:)+R*eye(nm)+1e-6*eye(nm);

                K = Pxz/Pzz;

                residual(:,clst) = meas(:,i)-zn(:);
                particle_mean(:,clst,i) = xn(:)+K(:,:)*residual(:,clst);
                particle_var(:,:,clst,i) = Pn(:,:)-K*Pzz(:,:)*K';

                cmean(:,clst) = particle_mean(:,clst,i);
                ccovar(:,:,clst) = particle_var(:,:,clst,i)';
                likelih(clst) = cweight(clst)*1/sqrt(norm(2*pi*det(R*eye(nm))))*exp(-1/2*residual(:,clst)'/R*eye(nm)*(residual(:,clst)));
            end
        else
            particle_exp = zeros(nm,numMixture_max,numStep);

            leng_old = 0;

            for clst = 1:numMixture
                leng = length(idx(idx == clst));

                Pxz = zeros(n,nm);
                Pzz = zeros(nm,nm);

                for j = 1:nm
                    particle_exp(j,clst,i) = mean(particle_meas(j,leng_old+1:leng_old+leng,i));
                end

                residual(:,clst) = meas(:,i)-particle_exp(:,clst,i);

                temp_meas = zeros(nm,leng);
                temp_mean = zeros(n,leng);

                for j = leng_old+1:leng_old+leng
                    temp_meas(:,j) = particle_meas(:,j,i)-particle_exp(:,clst,i);
                    temp_mean(:,j) = particle_clust(:,j,i)-particle_mean(:,clst,i);
                end

                Pxz(:,:) = temp_mean(:,:)*temp_meas(:,:)'/(leng-1);
                Pzz(:,:) = temp_meas(:,:)*temp_meas(:,:)'/(leng-1)+R*eye(nm)+1e-6*eye(nm);

                K = Pxz/Pzz;

                particle_mean(:,clst,i) = particle_mean(:,clst,i)+K*residual(:,clst);
                particle_var(:,:,clst,i) = particle_var(:,:,clst,i)-K*Pzz(:,:)*K';

                leng_old = leng_old+leng;
                cmean(:,clst) = particle_mean(:,clst,i);
                ccovar(:,:,clst) = particle_var(:,:,clst,i);
                likelih(clst) = cweight(clst)*1/sqrt(norm(2*pi*det(R*eye(nm))))*exp(-1/2*residual(:,clst)'/R*eye(nm)*(residual(:,clst)));
            end
        end

        % Save and mixture weight updated based on residual
        for clst = 1:numMixture
            cweight(clst) = likelih(clst)/sum(likelih);
        end

        cweight = normalize(cweight,"norm",1);

        if length(cweight) > 1
            for clst = 1:length(cweight)
                if sum(likelih) == 0 || isnan(cweight(clst)) || cweight(clst) == 0 %%%%% Need to figure out the better way
                    cweight = 1/numMixture*ones(numMixture,1);
                    break
                end
            end
        else
            if sum(likelih) == 0 || isnan(cweight) || cweight == 0 %%%%% Need to figure out the better way
                cweight = 1;
            end
        end

        cweight = normalize(cweight,"norm",1);

        for clst = 1:length(cweight)
            estState(:,i) = estState(:,i)+cweight(clst)*particle_mean(:,clst,i);

            % Covariance correction tricks
            ccovar(:,:,clst) = nearestSPD(ccovar(:,:,clst))+1e-6*eye(n);
        end

        % Sample new particles
        gm = gmdistribution(cmean',ccovar,cweight);

        temp = random(gm,numParticles);
        particles(:,:,i) = temp';
    else
        for clst = 1:length(cweight)
            cmean(:,clst) = particle_mean(:,clst,i);
            ccovar(:,:,clst) = particle_var(:,:,clst,i)';

            estState(:,i) = estState(:,i)+cweight(clst)*particle_mean(:,clst,i);
        end
    end

    for j = 1:n
        for k = 1:n
            if j == k
                temp_particle_var(j,k,i) = var(particles(j,:,i));
            else
                temp = cov(particles(j,:,i),particles(k,:,i));
                temp_particle_var(j,k,i) = temp(1,2);
            end
        end
    end
end