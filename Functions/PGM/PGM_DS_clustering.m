function [particle_clust,particle_mean,particle_var,...
          numMixture,cmean,ccovar,cweight,likelih,idx] = ...
               PGM_DS_clustering(i,n,numParticles,...
                              particles,particle_mean,particle_var,...
                              epsilon,minpts,numMixture_max,ifdbscan,iffirst)

    leng_old = 0;

    if ifdbscan
        idx = dbscan(particles(:,:,i)',epsilon,minpts);
    else
        numMixture = numMixture_max;
        [idx] = constrainedKMeans(particles(:,:,i)',numMixture,2,100);
    end

    [~,idx1] = sort(idx);
    idx = idx(idx1);
    particle_clust(:,:,i) = particles(:,idx1,i);

    numMixture = max(idx);

    cmean = zeros(n,numMixture);
    ccovar = zeros(n,n,numMixture);
    cweight = zeros(numMixture,1);
    likelih = zeros(numMixture,1);

    for clst = 1:numMixture
        leng = length(idx(idx == clst));
        cweight(clst) = leng/numParticles;

        for j = 1:n
            particle_mean(j,clst,i) = mean(particle_clust(j,leng_old+1:leng_old+leng,i));
        end

        for j = 1:n
            for k = 1:n
                if j == k
                    particle_var(j,k,clst,i) = var(particle_clust(j,leng_old+1:leng_old+leng,i));
                else
                    temp = cov(particle_clust(j,leng_old+1:leng_old+leng,i), ...
                                                   particle_clust(k,leng_old+1:leng_old+leng,i));
                    particle_var(j,k,clst,i) = temp(1,2);
                end
            end
        end

        leng_old = leng_old+leng;
        cmean(:,clst) = particle_mean(:,clst,i);
        ccovar(:,:,clst) = particle_var(:,:,clst,i)';
    end

    if iffirst

    elseif ifdbscan

    else
        LMES = getPDF_GMM(particle_clust(:,:,i), cweight, cmean(:,:)', ccovar(:,:,:));

        LMES_thres = LMES;
        particle_clust1(:,:) = particle_clust(:,:,i);
        particle_mean1(:,:) = particle_mean(:,:,i);
        particle_var1(:,:,:) = particle_var(:,:,:,i);
        cweight1 = cweight;
        idx2 = idx;
        numMixture_best = numMixture;

        while numMixture > 1
            numMixture = numMixture-1;
            leng_old = 0;

            cmean = zeros(n,numMixture);
            ccovar = zeros(n,n,numMixture);
            cweight = zeros(numMixture,1);

            [idx] = constrainedKMeans(particles(:,:,i)',numMixture,2,100);
            [~,idx1] = sort(idx);
            idx = idx(idx1);
            particle_clust(:,:,i) = particles(:,idx1,i);

            for clst = 1:numMixture
                leng = length(idx(idx == clst));
                cweight(clst) = leng/numParticles;

                for j = 1:n
                    particle_mean(j,clst,i) = mean(particle_clust(j,leng_old+1:leng_old+leng,i));
                end

                for j = 1:n
                    for k = 1:n
                        if j == k
                            particle_var(j,k,clst,i) = var(particle_clust(j,leng_old+1:leng_old+leng,i));
                        else
                            temp = cov(particle_clust(j,leng_old+1:leng_old+leng,i), ...
                                                           particle_clust(k,leng_old+1:leng_old+leng,i));
                            particle_var(j,k,clst,i) = temp(1,2);

                        end
                    end
                end

                leng_old = leng_old+leng;
                cmean(:,clst) = particle_mean(:,clst,i);
                ccovar(:,:,clst) = particle_var(:,:,clst,i)';
            end

            LMES = getPDF_GMM(particle_clust(:,:,i),cweight,cmean(:,:)',ccovar(:,:,:));

            if LMES >= LMES_thres
                numMixture_best = numMixture;
                LMES_thres = LMES;
                particle_clust1(:,:) = particle_clust(:,:,i);
                particle_mean1(:,:) = particle_mean(:,:,i);
                particle_var1(:,:,:) = particle_var(:,:,:,i);
                cweight1 = cweight;
                idx2 = idx;
            end 
        end

        numMixture = numMixture_best;
        particle_clust(:,:,i) = particle_clust1(:,:);
        particle_mean(:,:,i) = particle_mean1(:,:);
        particle_var(:,:,:,i) = particle_var1(:,:,:);

        cweight = cweight1;
        idx = idx2;
    end
end