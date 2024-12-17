function [particles,particle_clust,particle_mean,particle_var,...
          numMixture,cweight,idx] = ...
               PGM_DS_merging(i,n,numMixture,cweight,idx,merging_thres,...
                            particle_clust,particles,...
                            particle_mean,particle_var)

    notdone = true;

    while notdone
        idx_mer = [];
        for clst1 = 1:numMixture-1
            if ~ismember(clst1,idx_mer)
                for clst2 = clst1+1:numMixture
                    temp1 = particle_mean(:,clst1,i)-particle_mean(:,clst2,i);
                    temp2 = (particle_var(:,:,clst1,i)+particle_var(:,:,clst2,i))/2+1e-8*eye(n);

                    distance = sqrt(temp1'/temp2*temp1);

                    if distance < merging_thres
                        idx_mer = [idx_mer,clst2];
                        idx(idx == clst2) = clst1;
                    end
                end
            end
        end

        if isempty(idx_mer)
            notdone = false;

            % Single particle case filtering
            numMixture_temp = sum(unique(idx) ~= -1);
            numMixture = numMixture_temp;
        
            if numMixture_temp > 1
                for clst = 1:numMixture_temp
                    leng = length(idx(idx == clst));
            
                    if leng < 2
                        idx(idx == clst) = -1;
                        particle_mean(:,clst,:) = [];
                        particle_var(:,:,clst,:) = [];
                        cweight(clst) = [];
                        numMixture = numMixture-1;
                    end
                end
            end

        else
            % Initialise variables
            unique_values = unique(idx);
            new_idx = zeros(size(idx));

            % Assign consecutive integers to each unique value, except for -1
            counter = 1;
            for num = 1:numel(unique_values)
                if unique_values(num) ~= -1
                    indices = idx == unique_values(num);
                    if sum(indices) == 1
                        new_idx(indices) = -1;
                    else
                        new_idx(indices) = counter;
                        counter = counter+1;
                    end
                end
            end

            % Keep -1 unchanged
            new_idx(idx == -1) = -1;
            idx = new_idx;

            leng_old = 0;

            [~,idx1] = sort(idx);
            idx = idx(idx1);
            particle_clust(:,:,i) = particles(:,idx1,i);

            numMixture = sum(unique(idx) ~= -1);
            cweight = zeros(numMixture,1);

            for clst = 1:numMixture
                leng = length(idx(idx == clst));
                cweight(clst) = leng/length(particles);

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
            end
        end
    end

    particle_mean = particle_mean(:,1:numMixture,:);
    particle_var = particle_var(:,:,1:numMixture,:);
    cweight = normalize(cweight,"norm",1);
end