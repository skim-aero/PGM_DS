function [particles,particle_clust,particle_mean,particle_var,...
          numMixture,cweight,idx] = ...
               PGM_DS_merging(i,n,numMixture,cweight,idx,merging_thres,...
                            particle_clust,particles,...
                            particle_mean,particle_var)

    idx_mer = [];
    distance = zeros(numMixture,numMixture);
    isdone = 1;

    while isdone == 1
        for clst1 = 1:numMixture-1
            for clst2 = clst1+1:numMixture
                distance(clst1,clst2) = sqrt((particle_mean(:,clst1,i)-particle_mean(:,clst2,i))' ...
                                        /((particle_var(:,:,clst1,i)+particle_var(:,:,clst2,i))/2)* ...
                                        (particle_mean(:,clst1,i)-particle_mean(:,clst2,i)));

                if distance(clst1,clst2) < merging_thres
                    idx_mer = [idx_mer, clst2];
                    idx(idx == clst2) = clst1;
                    cweight(clst1) = cweight(clst1)+cweight(clst2);
                    cweight(clst2) = 0;
                end
            end
        end

        if isempty(idx_mer)
            isdone = 0;
        else
            cweight(cweight == 0) = [];
            numMixture = length(cweight);

            % Initialize variables
            unique_values = unique(idx);
            new_idx = zeros(size(idx));

            % Assign consecutive integers to each unique value, except for -1
            counter = 1;
            for num = 1:numel(unique_values)
                if unique_values(num) ~= -1
                    indices = idx == unique_values(num);
                    new_idx(indices) = counter;
                    counter = counter + 1;
                end
            end

            % Keep -1 unchanged
            new_idx(idx == -1) = -1;

            idx = new_idx;

            [~,idx1] = sort(idx);
            idx = idx(idx1);
            particle_clust(:,:,i) = particles(:,idx1,i);

            leng_old = 0;

            for clst = 1:numMixture
                leng = length(particle_clust(:,idx==clst,i));

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
            idx_mer = [];
            distance = zeros(numMixture,numMixture);
        end
    end

    cweight = normalize(cweight,"norm",1);
end