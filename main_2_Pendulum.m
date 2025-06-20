%% DBSCAN-based Particle Gaussian Mixture Filters
% Sukkeun Kim (s.kim.aero@gmail.com)

clc; clear; close all;
addpath(genpath('functions'));

warning('off','all')
warning

%% Simulation settings
figure_view = 1;            % 1: only when run with for (not parfor)
animation_view = 0;         % 1: show pdf animation / 0: no pdf animation
recordVid_view = 0;         % 1: record pdf animation / 0: no record
comparison_PGM_DS = 1;      % 1: compare with PGM_DS  / 0: no comparison
comparison_PGM_DU = 1;      % 1: compare with PGM_DU  / 0: no comparison
comparison_PGM1 = 1;        % 1: compare with PGM1  / 0: no comparison
comparison_PGM1_UT = 1;     % 1: compare with PGM1_UT  / 0: no comparison
comparison_AKKF = 1;        % 1: compare with AKKF / 0: no comparison
comparison_EnKF = 1;        % 1: compare with EnKF / 0: no comparison
comparison_UKF = 1;         % 1: compare with UKF / 0: no comparison
comparison_SIR = 1;         % 1: compare with SIR / 0: no comparison

numParticles = 100; % 100
numEnsembles = numParticles;
numMixturetemp = 10;
numMC = 100;

rng(22);

%% Define a scenario
% Time
dt = 0.05;
tf = 10;
t = 0:dt:tf;
numStep = length(t);

theta = 0:0.1:2*pi;
framerate = fix(numStep/tf);

% Parameters
n = 2;      % dimension of the state
nm = 1;     % dimension of the measurement 

L = 3;
g = 9.81;

Q = 10;
R = 0.02; % 0.02
merging_thres = chi2inv(0.995,n); % threshold for merging mixtures

trueinitial = [0.4*pi;0];
P0 = 0.4*eye(n); % 0.4*eye(n)

initial = [0;0]; % initial guess

% UT parameters
numSigma = 2*n+1;
alpha = 1;
beta = 2;
kappa = 0;
lambda = alpha^2*(n+kappa)-n;

% System/measurement equations
funsys = @(x,w,k,j) [wrapToPi(mod(x(1,:)+dt*x(2,:)+dt^2*(-g+w)*sin(x(1,:))/L,2*pi));
                     x(2,:)+dt*(-g+w)*sin(x(1,:))/L];
funmeas = @(x,v) abs(L*sin(x(1,:)))+v;
funpos = @(x) [L*sin(x(1,:));-L*cos(x(1,:))];
jacobianH = @(x) [L*cos(x(1)) 0];
F = [1 dt;0 1];
G = [dt^2/2;dt]/L;

%% For evaluation
% Error and RMSE
errorall = zeros(8,n,numStep,numMC);
errornorm = zeros(8,numStep,numMC);
rmse = zeros(8,numStep);
armse = zeros(8,1);

% Chi-square test
chisqall = zeros(8,numStep,numMC);
chinorm = zeros(8,numStep);
achi = zeros(8,1);

chithres = chi2inv(0.9999,n);
chicntall = zeros(8,numStep,numMC);
chicntfin = zeros(8,numStep,1);
chifrac = zeros(8,numStep,1);
chifracavg = zeros(8,1);

% Time-related
elpsdtime = zeros(8,numMC);
correctcnt = zeros(8,numMC);
cnttotal = zeros(8,1);
divergecnt = zeros(1,numMC);

% Cluster log
cmeanall = zeros(4,n,numMixturetemp,numStep,numMC);

%% Monte Carlo simulation
parfevalOnAll(@warning,0,'off','all'); % Temporary error off
parfor mc = 1:numMC
% for mc = 1:numMC 
    disptemp = ['Monete Carlo iteration: ',num2str(mc)];
    disp(disptemp)

    % Generate true states
    trueState = zeros(n,numStep);
    procNoise = sqrt(Q)*randn(n,numStep);
    trueState(:,1) = trueinitial+sqrt(P0)*randn(n,1);

    for i = 2:numStep
        trueState(:,i) = funsys(trueState(:,i-1),procNoise(i-1),[],[]);
    end

    truePos = funpos(trueState);
    
    % Generate measurements
    measNoise = sqrt(R)*randn(nm,numStep);
    meas = funmeas(trueState,measNoise);

    % For evaluation
    error = zeros(8,n,numStep);
    chisq = zeros(8,numStep); % chi-squared statistic storage
    elpst = zeros(8,1);

    crcnt = zeros(8,1);
    divcnt = zeros(1,1);

    cmeanmc = zeros(4,n,numMixturetemp,numStep);

    %% For plot
    if figure_view
        labels = {'PGM-DS','PGM-DU','PGM1','PGM1-UT','AKKF','EnKF','UKF','SIR'};
        markers = {'s','^','d','v','p','h','x','+'};
        colors  = {'r','g','#0072BD','#EDB120','m','c','#4DBEEE','#77AC30'};
        remap_columnwise = [1 5 2 6 3 7 4 8];
        marker_indices = 1:4:length(t);

        fig2 = figure; hold on; grid on;
        h2 = gobjects(1,8);
        h_true = plot(t, trueState(1,:), '--ok', ...
                      'DisplayName', 'True',...
                      'LineWidth',1,...
                      'MarkerSize',6,...
                      'MarkerIndices',marker_indices);
        xlim([0 tf]);
        xlabel('Time (sec)');
        ylabel('Estimated angle (rad)');

        fig3 = figure; hold on; grid on;
        h3 = gobjects(1,8);
        h_true_2 = plot(t, trueState(2,:), '--ok', ...
                      'DisplayName', 'True',...
                      'LineWidth',1,...
                      'MarkerSize',6,...
                      'MarkerIndices',marker_indices);
        xlim([0 tf]);
        xlabel('Time (sec)');
        ylabel('Estimated angular velocity (rad/sec)');
    
        fig4 = figure; hold on; grid on;
        h4 = gobjects(1,8);
        xlim([0 tf]);
        xlabel('Time (sec)');
        ylabel('|Estimation error| (rad)');

        % Pre-fill dummy handles for other algorithms
        for k = 1:8
            h2(k) = plot(NaN,NaN,'Marker',markers{k},...
                'Color',colors{k},'LineStyle','-',...
                'MarkerSize',6,'LineWidth',1,...
                'DisplayName',labels{k},'Visible','off');

            h3(k) = plot(NaN,NaN,'Marker',markers{k},...
                'Color',colors{k},'LineStyle','-',...
                'MarkerSize',6,'LineWidth',1,...
                'DisplayName',labels{k},'Visible','off');

            h4(k) = plot(NaN,NaN,'Marker',markers{k},...
                'Color',colors{k},'LineStyle','-',...
                'MarkerSize',6,'LineWidth',1,...
                'DisplayName',labels{k},'Visible','off');
        end
    end

    %% PGM-DS
    if comparison_PGM_DS
        %% Filter parameters
        fil_idx = 1;
        numMixture_max = n+1;   % number of Gaussian mixtures
        minpts = n+1;           % minimum number of neighbors for a core point n + 1
        epsilon = 5;            % distance to determine if a point is a core point 3
        
        %% Initialisation
        particles = zeros(n,numParticles,numStep);
        particle_mean = zeros(n,numMixture_max,numStep);
        particle_var = zeros(n,n,numMixture_max,numStep);
        particle_meas = zeros(nm,numParticles,numStep);
        
        estState = zeros(n,numStep);
        temp_particle_var = zeros(n,n,numStep);
    
        %% First time step
        particles(:,:,1) = initial+sqrt(P0)*randn(n,numParticles);
    
        for p = 1:numParticles
            particle_meas(:,p,1) = funmeas(particles(:,p,1),0);
        end 
    
        estState(:,1) = mean(particles(:,:,1),2);
        temp_particle_var(:,:,1) = P0;
    
        error(fil_idx,:,1) = abs(estState(:,1)-trueState(:,1));
        chisq(fil_idx,1) = error(fil_idx,:,1)*(temp_particle_var(:,:,1)\error(fil_idx,:,1)');
    
        % Clustering
        [~,particle_mean,particle_var,~,...
                cmean,~,cweight,~,~] = ...
                   PGM_DS_clustering(1,n,numParticles,...
                                  particles,particle_mean,particle_var,...
                                  epsilon,minpts,numMixture_max,1,1);
    
        for k = length(cweight)
            cmeanmc(fil_idx,:,k,1) = cmean(:,k);
        end
        
        particle_pos = funpos(particles(:,:,1));
    
        if animation_view
            fig99 = figure; hold on; axis equal; legend;
    
            estPos = funpos(estState(:,1));
            plot(L*cos(theta),L*sin(theta),'k:','HandleVisibility','off');
            fig_partPos = plot(particle_pos(1,:),particle_pos(2,:),'r.','DisplayName','Particles');
            fig_estPos = plot(estPos(1),estPos(2),'bo','DisplayName','Estimated','MarkerFaceColor','b');
            fig_truePos = plot(truePos(1,1),truePos(2,1),'ko','DisplayName','True','MarkerFaceColor','k');
            range = plot([0 truePos(1,1)],[0 truePos(2,1)],'k:','HandleVisibility','off');
            title('True and Estimated Positions')
    %         frame(1) = getframe(gcf);
        end
    
        %% Main loop
        tStart1 = tic; 
    
        for i = 2:numStep
            %% Prediction
            for p = 1:numParticles
                particles(:,p,i) = funsys(particles(:,p,i-1),sqrt(Q)*randn(1),[],[]);
            end
        
            %% Clustering
            [particle_clust,particle_mean,particle_var,numMixture,...
                        ~,~,cweight,likelih,idx] = ...
                           PGM_DS_clustering(i,n,numParticles,...
                                          particles,particle_mean,particle_var,...
                                          epsilon,minpts,numMixture_max,1,0);
    
            %% Merging
            [~,particle_clust,particle_mean,particle_var,...
                numMixture,cweight,idx] = ...
                           PGM_DS_merging(i,n,numMixture,cweight,idx,merging_thres,...
                                        particle_clust,particles,...
                                        particle_mean,particle_var);
    
            %%  Update
            for p = 1:numParticles
                particle_meas(:,p,i) = funmeas(particles(:,p,i),sqrt(R)*randn(nm,1));
            end
    
            [particles,estState,temp_particle_var,...
             cmean,~,cweight] = ...
                           PGM_DS_update(i,n,nm,numStep,numParticles,...
                                       numMixture,numMixture_max,...
                                       1,cweight,idx,meas,likelih,R,...
                                       particles,particle_clust,...
                                       particle_meas,particle_mean,...
                                       particle_var,estState,...
                                       temp_particle_var,0,0,0,0,0,0);
    
            %% For evaluation
            error(fil_idx,:,i) = abs(estState(:,i)-trueState(:,i));
            chisq(fil_idx,i) = error(fil_idx,:,i)*(temp_particle_var(:,:,i)\error(fil_idx,:,i)');
    
            for k = length(cweight)
                cmeanmc(fil_idx,:,k,i) = cmean(:,k);
            end
        
            particle_pos = funpos(particles(:,:,i));
    
            if animation_view
                delete(fig_truePos)
                delete(fig_partPos)
                delete(fig_estPos)
                delete(range)
                estPos = funpos(estState(:,i));
                fig_partPos = plot(particle_pos(1,:),particle_pos(2,:),'r.','DisplayName','Particles');
                fig_estPos = plot(estPos(1),estPos(2),'bo','DisplayName','Estimated','MarkerFaceColor','b');
                fig_truePos = plot(truePos(1,i),truePos(2,i),'ko','DisplayName','True','MarkerFaceColor','k');
                range = plot([0 truePos(1,i)],[0 truePos(2,i)],'k:','HandleVisibility','off');
                drawnow;
            end
        end
    
        elpst(fil_idx) = toc(tStart1);
        avgerr = mean(error(fil_idx,1,:));
    
        if avgerr < 0.5
            crcnt(fil_idx) = 1;
        end
    
        temperror = zeros(1,length(error(fil_idx,1,:)));
        temperror(:) = error(fil_idx,1,:);
    
        if figure_view
            figure(fig2);
            h2(fil_idx) = plot(t,estState(1,:),...
                'Color',colors{fil_idx},'Marker',markers{fil_idx},...
                'MarkerIndices',marker_indices,...
                'LineWidth',1,'MarkerSize',6,...
                'DisplayName',labels{fil_idx});

            figure(fig3);
            h3(fil_idx) = plot(t,estState(2,:),...
                'Color',colors{fil_idx},'Marker',markers{fil_idx},...
                'MarkerIndices',marker_indices,...
                'LineWidth',1,'MarkerSize',6,...
                'DisplayName',labels{fil_idx});
        
            figure(fig4);
            h4(fil_idx) = plot(t,temperror,...
                'Color',colors{fil_idx},'Marker',markers{fil_idx},...
                'MarkerIndices',marker_indices,...
                'LineWidth',1,'MarkerSize',6,...
                'DisplayName',labels{fil_idx});
        end
    
        if recordVid_view
            writeObj = VideoWriter('PGM-DS','MPEG-4');
            writeObj.FrameRate = framerate;
            open(writeObj);
            writeVideo(writeObj,frame);
            close(writeObj);
        end
    end

    %% PGM-DU
    if comparison_PGM_DU
        %% Filter parameters
        fil_idx = 2;
        numMixture_max = n+1;   % number of Gaussian mixtures
        minpts = n+1;           % minimum number of neighbors for a core point n + 1
        epsilon = 5;            % distance to determine if a point is a core point 3
        
        %% Initialisation
        particles = zeros(n,numParticles,numStep);
        particle_mean = zeros(n,numMixture_max,numStep);
        particle_var = zeros(n,n,numMixture_max,numStep);
        particle_meas = zeros(nm,numParticles,numStep);
    
        wm = zeros(numSigma,1);
        wc = zeros(numSigma,1);
    
        estState = zeros(n,numStep);
        temp_particle_var = zeros(n,n,numStep);
    
        %% First time step
        % particles(:,:,1) = [2*pi*rand(1,numParticles);initial+P0*randn(1,numParticles)];
        particles(:,:,1) = initial+sqrt(P0)*randn(n,numParticles);
    
        for p = 1:numParticles
            particle_meas(:,p,1) = funmeas(particles(:,p,1),0);
        end 
    
        wm(1,:) = lambda/(n+lambda);
        wc(1,:) = lambda/(n+lambda)+(1-alpha^2+beta);
    
        for j = 2:numSigma   
            wm(j,:) = 1/(2*(n+lambda));
            wc(j,:) = wm(j);
        end
        
        estState(:,1) = mean(particles(:,:,1),2);
        temp_particle_var(:,:,1) = P0;
    
        error(fil_idx,:,1) = abs(estState(:,1)-trueState(:,1));
        chisq(fil_idx,1) = error(fil_idx,:,1)*(temp_particle_var(:,:,1)\error(fil_idx,:,1)');
    
        % Clustering
        [~,particle_mean,particle_var,~,...
                cmean,~,cweight,~,~] = ...
                   PGM_DS_clustering(1,n,numParticles,...
                                  particles,particle_mean,particle_var,...
                                  epsilon,minpts,numMixture_max,1,1);
    
        for k = length(cweight)
            cmeanmc(fil_idx,:,k,1) = cmean(:,k);
        end
        
        particle_pos = funpos(particles(:,:,1));

        if animation_view
            fig99 = figure; hold on; axis equal; legend;
    
            estPos = funpos(estState(:,1));
            plot(L*cos(theta),L*sin(theta),'k:','HandleVisibility','off');
            fig_partPos = plot(particle_pos(1,:),particle_pos(2,:),'r.','DisplayName','Particles');
            fig_estPos = plot(estPos(1),estPos(2),'bo','DisplayName','Estimated','MarkerFaceColor','b');
            fig_truePos = plot(truePos(1,1),truePos(2,1),'ko','DisplayName','True','MarkerFaceColor','k');
            range = plot([0 truePos(1,1)],[0 truePos(2,1)],'k:','HandleVisibility','off');
            title('True and Estimated Positions')
        end
    
        %% Main loop
        tStart2 = tic; 
    
        for i = 2:numStep
            %% Prediction
            for p = 1:numParticles
                particles(:,p,i) = funsys(particles(:,p,i-1),sqrt(Q)*randn(1),[],[]);
            end
        
            %% Clustering
            [particle_clust,particle_mean,particle_var,numMixture,...
                        ~,~,cweight,likelih,idx] = ...
                           PGM_DS_clustering(i,n,numParticles,...
                                          particles,particle_mean,particle_var,...
                                          epsilon,minpts,numMixture_max,1,0);
    
            %% Merging
            [~,particle_clust,particle_mean,particle_var,...
                numMixture,cweight,idx] = ...
                           PGM_DS_merging(i,n,numMixture,cweight,idx,merging_thres,...
                                        particle_clust,particles,...
                                        particle_mean,particle_var);
    
            %% Update
            [particles,estState,temp_particle_var,...
             cmean,~,cweight] = ...
                           PGM_DS_update(i,n,nm,numStep,numParticles,...
                                       numMixture,numMixture_max,...
                                       1,cweight,idx,meas,likelih,R,...
                                       particles,particle_clust,...
                                       particle_meas,particle_mean,...
                                       particle_var,estState,...
                                       temp_particle_var,...
                                       1,numSigma,lambda,wc,wm,funmeas);
    
            %% For evaluation
            error(fil_idx,:,i) = abs(estState(:,i)-trueState(:,i));
            chisq(fil_idx,i) = error(fil_idx,:,i)*(temp_particle_var(:,:,i)\error(fil_idx,:,i)');
    
            for k = length(cweight)
                cmeanmc(fil_idx,:,k,i) = cmean(:,k);
            end  
    
            particle_pos = funpos(particles(:,:,i));
            
            if animation_view
                delete(fig_truePos)
                delete(fig_partPos)
                delete(fig_estPos)
                delete(range)
                estPos = funpos(estState(:,i));
                fig_partPos = plot(particle_pos(1,:),particle_pos(2,:),'r.','DisplayName','Particles');
                fig_estPos = plot(estPos(1),estPos(2),'bo','DisplayName','Estimated','MarkerFaceColor','b');
                fig_truePos = plot(truePos(1,i),truePos(2,i),'ko','DisplayName','True','MarkerFaceColor','k');
                range = plot([0 truePos(1,i)],[0 truePos(2,i)],'k:','HandleVisibility','off');
                drawnow;
            end
        end
    
        elpst(fil_idx) = toc(tStart2);
        avgerr = mean(error(fil_idx,1,:));
        
        if avgerr < 0.5
            crcnt(fil_idx) = 1;
        end
        
        temperror = zeros(1,length(error(fil_idx,1,:)));
        temperror(:) = error(fil_idx,1,:);
    
        if figure_view
            figure(fig2);
            h2(fil_idx) = plot(t,estState(1,:),...
                'Color',colors{fil_idx},'Marker',markers{fil_idx},...
                'MarkerIndices',marker_indices,...
                'LineWidth',1,'MarkerSize',6,...
                'DisplayName',labels{fil_idx});

            figure(fig3);
            h3(fil_idx) = plot(t,estState(2,:),...
                'Color',colors{fil_idx},'Marker',markers{fil_idx},...
                'MarkerIndices',marker_indices,...
                'LineWidth',1,'MarkerSize',6,...
                'DisplayName',labels{fil_idx});
        
            figure(fig4);
            h4(fil_idx) = plot(t,temperror,...
                'Color',colors{fil_idx},'Marker',markers{fil_idx},...
                'MarkerIndices',marker_indices,...
                'LineWidth',1,'MarkerSize',6,...
                'DisplayName',labels{fil_idx});
        end
    
        if recordVid_view
            writeObj = VideoWriter('PGM-DU','MPEG-4');
            writeObj.FrameRate = framerate;
            open(writeObj);
            writeVideo(writeObj,frame);
            close(writeObj);
        end
    end
  
    %% PGM1
    if comparison_PGM1
        %% Filter parameters
        fil_idx = 3;
        numMixture_max = 2;     % number of Gaussian mixtures
        
        %% Initialisation
        particles = zeros(n,numParticles,numStep);
        particle_mean = zeros(n,numMixture_max,numStep);
        particle_var = zeros(n,n,numMixture_max,numStep);
        particle_meas = zeros(nm,numParticles,numStep);
        
        estState = zeros(n,numStep);
        temp_particle_var = zeros(n,n,numStep);
    
        %% First time step
        particles(:,:,1) = initial+sqrt(P0)*randn(n,numParticles);
    
        for p = 1:numParticles
            particle_meas(:,p,1) = funmeas(particles(:,p,1),0);
        end 
    
        estState(:,1) = mean(particles(:,:,1),2);
        temp_particle_var(:,:,1) = P0;
    
        error(fil_idx,:,1) = abs(estState(:,1)-trueState(:,1));
        chisq(fil_idx,1) = error(fil_idx,:,1)*(temp_particle_var(:,:,1)\error(fil_idx,:,1)');
        
        % Clustering
        [~,particle_mean,particle_var,~,...
                    cmean,~,cweight,~,~] = ...
                       PGM_DS_clustering(1,n,numParticles,...
                                      particles,particle_mean,particle_var,...
                                      0,0,numMixture_max,0,1);
    
        for k = length(cweight)
            cmeanmc(fil_idx,:,k,1) = cmean(:,k);
        end
        
        particle_pos = funpos(particles(:,:,1));

        if animation_view
            fig99 = figure; hold on; axis equal; legend;
    
            estPos = funpos(estState(:,1));
            plot(L*cos(theta),L*sin(theta),'k:','HandleVisibility','off');
            fig_partPos = plot(particle_pos(1,:),particle_pos(2,:),'r.','DisplayName','Particles');
            fig_estPos = plot(estPos(1),estPos(2),'bo','DisplayName','Estimated','MarkerFaceColor','b');
            fig_truePos = plot(truePos(1,1),truePos(2,1),'ko','DisplayName','True','MarkerFaceColor','k');
            range = plot([0 truePos(1,1)],[0 truePos(2,1)],'k:','HandleVisibility','off');
            title('True and Estimated Positions')
        end
    
        %% Main loop
        tStart3 = tic; 
    
        for i = 2:numStep
            %% Prediction
            for p = 1:numParticles
                particles(:,p,i) = funsys(particles(:,p,i-1),sqrt(Q)*randn(1),[],[]);
            end
    
            %% Clustering
            [particle_clust,particle_mean,particle_var,numMixture,...
                        ~,~,cweight,likelih,idx] = ...
                           PGM_DS_clustering(i,n,numParticles,...
                                          particles,particle_mean,particle_var,...
                                          0,0,numMixture_max,0,0);
    
            %% Merging
            [~,particle_clust,particle_mean,particle_var,...
                numMixture,cweight,idx] = ...
                           PGM_DS_merging(i,n,numMixture,cweight,idx,merging_thres,...
                                        particle_clust,particles,...
                                        particle_mean,particle_var);
    
            %%  Update
            for p = 1:numParticles
                particle_meas(:,p,i) = funmeas(particles(:,p,i),sqrt(R)*randn(nm,1));
            end
    
            [particles,estState,temp_particle_var,...
             cmean,~,cweight] = ...
                           PGM_DS_update(i,n,nm,numStep,numParticles,...
                                       numMixture,numMixture_max,...
                                       1,cweight,idx,meas,likelih,R,...
                                       particles,particle_clust,...
                                       particle_meas,particle_mean,...
                                       particle_var,estState,...
                                       temp_particle_var,0,0,0,0,0,0);
            
            %% For evaluation
            error(fil_idx,:,i) = abs(estState(:,i)-trueState(:,i));
            chisq(fil_idx,i) = error(fil_idx,:,i)*(temp_particle_var(:,:,i)\error(fil_idx,:,i)');
    
            for k = length(cweight)
                cmeanmc(fil_idx,:,k,i) = cmean(:,k);
            end
        
            particle_pos = funpos(particles(:,:,i));
            if animation_view
                delete(fig_truePos)
                delete(fig_partPos)
                delete(fig_estPos)
                delete(range)
                estPos = funpos(estState(:,i));
                fig_partPos = plot(particle_pos(1,:),particle_pos(2,:),'r.','DisplayName','Particles');
                fig_estPos = plot(estPos(1),estPos(2),'bo','DisplayName','Estimated','MarkerFaceColor','b');
                fig_truePos = plot(truePos(1,i),truePos(2,i),'ko','DisplayName','True','MarkerFaceColor','k');
                range = plot([0 truePos(1,i)],[0 truePos(2,i)],'k:','HandleVisibility','off');
                drawnow;
            end
    
        end
    
        elpst(fil_idx) = toc(tStart3);
        avgerr = mean(error(fil_idx,1,:));
    
        if avgerr < 0.5
            crcnt(fil_idx) = 1;
        end
    
        temperror = zeros(1,length(error(fil_idx,1,:)));
        temperror(:) = error(fil_idx,1,:);
    
        if figure_view
            figure(fig2);
            h2(fil_idx) = plot(t,estState(1,:),...
                'Color',colors{fil_idx},'Marker',markers{fil_idx},...
                'MarkerIndices',marker_indices,...
                'LineWidth',1,'MarkerSize',6,...
                'DisplayName',labels{fil_idx});

            figure(fig3);
            h3(fil_idx) = plot(t,estState(2,:),...
                'Color',colors{fil_idx},'Marker',markers{fil_idx},...
                'MarkerIndices',marker_indices,...
                'LineWidth',1,'MarkerSize',6,...
                'DisplayName',labels{fil_idx});
        
            figure(fig4);
            h4(fil_idx) = plot(t,temperror,...
                'Color',colors{fil_idx},'Marker',markers{fil_idx},...
                'MarkerIndices',marker_indices,...
                'LineWidth',1,'MarkerSize',6,...
                'DisplayName',labels{fil_idx});
        end
        
        if recordVid_view
            writeObj = VideoWriter('PGM1','MPEG-4');
            writeObj.FrameRate = framerate;
            open(writeObj);
            writeVideo(writeObj,frame);
            close(writeObj);
        end
    end

    %% PGM-UT
    if comparison_PGM1_UT
        %% Filter parameters
        fil_idx = 4;
        numMixture_max = 2;     % number of Gaussian mixtures
        
        %% Initialisation
        particles = zeros(n,numParticles,numStep);
        particle_mean = zeros(n,numMixture_max,numStep);
        particle_var = zeros(n,n,numMixture_max,numStep);
        particle_meas = zeros(nm,numParticles,numStep);
    
        wm = zeros(numSigma,1);
        wc = zeros(numSigma,1);
    
        estState = zeros(n,numStep);
        temp_particle_var = zeros(n,n,numStep);
    
        %% First time step
        particles(:,:,1) = initial+sqrt(P0)*randn(n,numParticles);
    
        for p = 1:numParticles
            particle_meas(:,p,1) = funmeas(particles(:,p,1),0);
        end 
    
        wm(1,:) = lambda/(n+lambda);
        wc(1,:) = lambda/(n+lambda)+(1-alpha^2+beta);
    
        for j = 2:numSigma   
            wm(j,:) = 1/(2*(n+lambda));
            wc(j,:) = wm(j);
        end
        
        estState(:,1) = mean(particles(:,:,1),2);
        temp_particle_var(:,:,1) = P0;
    
        error(fil_idx,:,1) = abs(estState(:,1)-trueState(:,1));
        chisq(fil_idx,1) = error(fil_idx,:,1)*(temp_particle_var(:,:,1)\error(fil_idx,:,1)');
    
        % Clustering
        [~,particle_mean,particle_var,~,...
                    cmean,~,cweight,~,~] = ...
                       PGM_DS_clustering(1,n,numParticles,...
                                      particles,particle_mean,particle_var,...
                                      0,0,numMixture_max,0,1);
    
        for k = length(cweight)
            cmeanmc(fil_idx,:,k,1) = cmean(:,k);
        end
        
        particle_pos = funpos(particles(:,:,1));

        if animation_view
            fig99 = figure; hold on; axis equal; legend;
    
            estPos = funpos(estState(:,1));
            plot(L*cos(theta),L*sin(theta),'k:','HandleVisibility','off');
            fig_partPos = plot(particle_pos(1,:),particle_pos(2,:),'r.','DisplayName','Particles');
            fig_estPos = plot(estPos(1),estPos(2),'bo','DisplayName','Estimated','MarkerFaceColor','b');
            fig_truePos = plot(truePos(1,1),truePos(2,1),'ko','DisplayName','True','MarkerFaceColor','k');
            range = plot([0 truePos(1,1)],[0 truePos(2,1)],'k:','HandleVisibility','off');
            title('True and Estimated Positions')
        end
    
        %% Main loop
        tStart4 = tic; 
    
        for i = 2:numStep
            %% Prediction
            for p = 1:numParticles
                particles(:,p,i) = funsys(particles(:,p,i-1),sqrt(Q)*randn(1),[],[]);
            end
        
            %% Clustering
            [particle_clust,particle_mean,particle_var,numMixture,...
                        ~,~,cweight,likelih,idx] = ...
                           PGM_DS_clustering(i,n,numParticles,...
                                          particles,particle_mean,particle_var,...
                                          0,0,numMixture_max,0,0);
    
            %% Merging
            [~,particle_clust,particle_mean,particle_var,...
                numMixture,cweight,idx] = ...
                           PGM_DS_merging(i,n,numMixture,cweight,idx,merging_thres,...
                                        particle_clust,particles,...
                                        particle_mean,particle_var);
    
            %% Update
            [particles,estState,temp_particle_var,...
             cmean,~,cweight] = ...
                           PGM_DS_update(i,n,nm,numStep,numParticles,...
                                       numMixture,numMixture_max,...
                                       1,cweight,idx,meas,likelih,R,...
                                       particles,particle_clust,...
                                       particle_meas,particle_mean,...
                                       particle_var,estState,...
                                       temp_particle_var,...
                                       1,numSigma,lambda,wc,wm,funmeas);
    
            %% For evaluation
            error(fil_idx,:,i) = abs(estState(:,i)-trueState(:,i));
            chisq(fil_idx,i) = error(fil_idx,:,i)*(temp_particle_var(:,:,i)\error(fil_idx,:,i)');
    
            for k = length(cweight)
                cmeanmc(fil_idx,:,k,i) = cmean(:,k);
            end  
    
            particle_pos = funpos(particles(:,:,i));
            
            if animation_view
                delete(fig_truePos)
                delete(fig_partPos)
                delete(fig_estPos)
                delete(range)
                estPos = funpos(estState(:,i));
                fig_partPos = plot(particle_pos(1,:),particle_pos(2,:),'r.','DisplayName','Particles');
                fig_estPos = plot(estPos(1),estPos(2),'bo','DisplayName','Estimated','MarkerFaceColor','b');
                fig_truePos = plot(truePos(1,i),truePos(2,i),'ko','DisplayName','True','MarkerFaceColor','k');
                range = plot([0 truePos(1,i)],[0 truePos(2,i)],'k:','HandleVisibility','off');
                drawnow;
            end
        end
    
        elpst(fil_idx) = toc(tStart4);
        avgerr = mean(error(fil_idx,1,:));
        
        if avgerr < 0.5
            crcnt(fil_idx) = 1;
        end
        
        temperror = zeros(1,length(error(fil_idx,1,:)));
        temperror(:) = error(fil_idx,1,:);
    
        if figure_view
            figure(fig2);
            h2(fil_idx) = plot(t,estState(1,:),...
                'Color',colors{fil_idx},'Marker',markers{fil_idx},...
                'MarkerIndices',marker_indices,...
                'LineWidth',1,'MarkerSize',6,...
                'DisplayName',labels{fil_idx});

            figure(fig3);
            h3(fil_idx) = plot(t,estState(2,:),...
                'Color',colors{fil_idx},'Marker',markers{fil_idx},...
                'MarkerIndices',marker_indices,...
                'LineWidth',1,'MarkerSize',6,...
                'DisplayName',labels{fil_idx});
        
            figure(fig4);
            h4(fil_idx) = plot(t,temperror,...
                'Color',colors{fil_idx},'Marker',markers{fil_idx},...
                'MarkerIndices',marker_indices,...
                'LineWidth',1,'MarkerSize',6,...
                'DisplayName',labels{fil_idx});
        end
    
        if recordVid_view
            writeObj = VideoWriter('PGM1-UT','MPEG-4');
            writeObj.FrameRate = framerate;
            open(writeObj);
            writeVideo(writeObj,frame);
            close(writeObj);
        end
    end

    %% AKKF
    if comparison_AKKF
        %% Filter parameters
        fil_idx = 5;
        AKKF.kernel = 1;        % "Quadratic" "Quartic" "Gaussian"
        AKKF.N_P = numParticles;
    
        AKKF.c = 1;
        AKKF.lambda = 10^(-3);      % kernel normisation parameter
        AKKF.lambda_KKF = 10^(-3);  % kernel normisation parameter for Kernel Kalman filter gain calculation
        AKKF.poly_para_b = 10^(3);  % polynomial triack scale for the bearing tracking
        AKKF.Var_Gaussian = R;
    
        %% Initialisation
        estState = zeros(n,numStep); % storage for state estimates
        temp_particle_var = zeros(n,n,numStep);
    
        %% First time step
        AKKF.X_P(:,:,1) = initial+sqrt(P0)*randn(n,numParticles);
    
        % Data space
    %     AKKF.X_P(:,:,1) = mgd(AKKF.N_P,4,Tar.X(:,1),Sys.P0).'; % state particles
        
        % Kernel space
        AKKF.W_minus(:,1) = ones(AKKF.N_P,1)/AKKF.N_P;
        AKKF.S_minus(:,:,1) = eye(AKKF.N_P)/AKKF.N_P;
        
        AKKF.W_plus(:,1) = ones(AKKF.N_P,1)/AKKF.N_P;
        AKKF.S_plus(:,:,1) = eye(AKKF.N_P)/AKKF.N_P;
        
        AKKF.X_P_proposal(:,:,1) =  AKKF.X_P(:,:,1);
        
        AKKF.X_est(:,1) = AKKF.X_P(:,:,1) * AKKF.W_plus(:,1); %state mean
        AKKF.X_C(:,:,1) =   AKKF.X_P(:,:,1) * AKKF.S_plus(:,:,1) * AKKF.X_P(:,:,1).'; %state covariance
        
        for p = 1:numParticles
            AKKF.Z_P(:,p,1) = funmeas(AKKF.X_P(:,p,1),0);
        end 
        
        estState(:,1) = mean(AKKF.X_P(:,:,1),2);
        temp_particle_var(:,:,1) = P0;
    
        error(fil_idx,:,1) = abs(estState(:,1)-trueState(:,1));
        chisq(fil_idx,1) = error(fil_idx,:,1)*(temp_particle_var(:,:,1)\error(fil_idx,:,1)');
    
        particle_pos = funpos(AKKF.Z_P(:,:,1));
        
        if animation_view
            fig99 = figure; hold on; axis equal; legend;
    
            estPos = funpos(estState(:,1));
            plot(L*cos(theta),L*sin(theta),'k:','HandleVisibility','off');
            fig_partPos = plot(particle_pos(1,:),particle_pos(2,:),'r.','DisplayName','Particles');
            fig_estPos = plot(estPos(1),estPos(2),'bo','DisplayName','Estimated','MarkerFaceColor','b');
            fig_truePos = plot(truePos(1,1),truePos(2,1),'ko','DisplayName','True','MarkerFaceColor','k');
            range = plot([0 truePos(1,1)],[0 truePos(2,1)],'k:','HandleVisibility','off');
            title('True and Estimated Positions')
        end
    
        %% Main loop
        tStart5 = tic; 
    
        for i = 2:numStep
            %% Prediction
            for p = 1:numParticles
                 AKKF.X_P(:,p,i) = funsys(AKKF.X_P_proposal(:,p,i-1),sqrt(Q)*randn(1),[],[]);
            end
    
            [AKKF] = AKKF_predict(AKKF,i);
    
            %% Update
            for p = 1:numParticles
                AKKF.Z_P(:,p,i) = funmeas(AKKF.X_P(:,p,i),sqrt(R)*randn(nm,1));
            end
    
            [AKKF] = AKKF_update(meas,AKKF,R,i);
            [AKKF] = AKKF_proposal(AKKF,i);
    
            estState(:,i) = mean(AKKF.X_P(:,:,i),2);
    
            %% For evaluation
            for j = 1:n
                for k = 1:n
                    if j == k
                        temp_particle_var(j,k,i) = var(AKKF.X_P(j,:,i));
                    else
                        temp = cov(AKKF.X_P(j,:,i),AKKF.X_P(k,:,i));
                        temp_particle_var(j,k,i) = temp(1,2);
                    end
                end
            end
            
            error(fil_idx,:,i) = abs(estState(:,i)-trueState(:,i));
            chisq(fil_idx,i) = error(fil_idx,:,i)*(temp_particle_var(:,:,i)\error(fil_idx,:,i)');
    
            particle_pos = funpos(AKKF.X_P(:,:,i));
            
            if animation_view
                delete(fig_truePos)
                delete(fig_partPos)
                delete(fig_estPos)
                delete(range)
                estPos = funpos(estState(:,i));
                fig_partPos = plot(particle_pos(1,:),particle_pos(2,:),'r.','DisplayName','Particles');
                fig_estPos = plot(estPos(1),estPos(2),'bo','DisplayName','Estimated','MarkerFaceColor','b');
                fig_truePos = plot(truePos(1,i),truePos(2,i),'ko','DisplayName','True','MarkerFaceColor','k');
                range = plot([0 truePos(1,i)],[0 truePos(2,i)],'k:','HandleVisibility','off');
                drawnow;
            end
        end
    
        elpst(fil_idx) = toc(tStart5);
        avgerr = mean(error(fil_idx,1,:));
    
        if avgerr < 0.5
            crcnt(fil_idx) = 1;
        end
            
        temperror = zeros(1,length(error(fil_idx,1,:)));
        temperror(:) = error(fil_idx,1,:);
    
        if figure_view
            figure(fig2);
            h2(fil_idx) = plot(t,estState(1,:),...
                'Color',colors{fil_idx},'Marker',markers{fil_idx},...
                'MarkerIndices',marker_indices,...
                'LineWidth',1,'MarkerSize',6,...
                'DisplayName',labels{fil_idx});

            figure(fig3);
            h3(fil_idx) = plot(t,estState(2,:),...
                'Color',colors{fil_idx},'Marker',markers{fil_idx},...
                'MarkerIndices',marker_indices,...
                'LineWidth',1,'MarkerSize',6,...
                'DisplayName',labels{fil_idx});
        
            figure(fig4);
            h4(fil_idx) = plot(t,temperror,...
                'Color',colors{fil_idx},'Marker',markers{fil_idx},...
                'MarkerIndices',marker_indices,...
                'LineWidth',1,'MarkerSize',6,...
                'DisplayName',labels{fil_idx});
        end
    end

    %% EnKF
    if comparison_EnKF
        %% Filter parameters
        fil_idx = 6;

        %% Initialisation
        ensembles = zeros(n,numEnsembles,numStep);
        ensemble_mean = zeros(n,numStep);
        ensemble_var = zeros(n,n,numStep);
    
        estState = zeros(n,numStep); % storage for state estimates
        temp_particle_var = zeros(n,n,numStep);
    
        %% First time step
        ensembles(:,:,1) = initial+sqrt(P0)*randn(n,numEnsembles); 
        ensemble_var(:,:,1) = P0;
        
        estState(:,1) = mean(ensembles(:,:,1),2);
        temp_particle_var(:,:,1) = P0;
    
        error(fil_idx,:,1) = abs(estState(:,1)-trueState(:,1));
        chisq(fil_idx,1) = error(fil_idx,:,1)*(temp_particle_var(:,:,1)\error(fil_idx,:,1)');
    
        particle_pos = funpos(ensembles(:,:,1));
        
        if animation_view
            fig99 = figure; hold on; axis equal; legend;
    
            estPos = funpos(estState(:,1));
            plot(L*cos(theta),L*sin(theta),'k:','HandleVisibility','off');
            fig_partPos = plot(particle_pos(1,:),particle_pos(2,:),'r.','DisplayName','Particles');
            fig_estPos = plot(estPos(1),estPos(2),'bo','DisplayName','Estimated','MarkerFaceColor','b');
            fig_truePos = plot(truePos(1,1),truePos(2,1),'ko','DisplayName','True','MarkerFaceColor','k');
            range = plot([0 truePos(1,1)],[0 truePos(2,1)],'k:','HandleVisibility','off');
            title('True and Estimated Positions')
        end
        
        %% Main loop
        tStart6 = tic; 

        [estState,ensemble_var] = EnKF(numStep,n,nm,meas,numEnsembles,...
                                       estState,ensembles,ensemble_var,ensemble_mean,...
                                       Q,R,[],[],jacobianH,funsys,funmeas,[]);
    
        %% For evaluation
        for i = 2:numStep
            temp_particle_var(:,:,i) = ensemble_var(:,:,i);
    
            error(fil_idx,:,i) = abs(estState(:,i)-trueState(:,i));
            chisq(fil_idx,i) = error(fil_idx,:,i)*(temp_particle_var(:,:,i)\error(fil_idx,:,i)');
    
            particle_pos = funpos(ensembles(:,:,i));
           
            if animation_view
                delete(fig_truePos)
                delete(fig_partPos)
                delete(fig_estPos)
                delete(range)
                estPos = funpos(estState(:,i));
                fig_partPos = plot(particle_pos(1,:),particle_pos(2,:),'r.','DisplayName','Particles');
                fig_estPos = plot(estPos(1),estPos(2),'bo','DisplayName','Estimated','MarkerFaceColor','b');
                fig_truePos = plot(truePos(1,i),truePos(2,i),'ko','DisplayName','True','MarkerFaceColor','k');
                range = plot([0 truePos(1,i)],[0 truePos(2,i)],'k:','HandleVisibility','off');
                drawnow;
            end
        end
    
        elpst(fil_idx) = toc(tStart6);
        avgerr = mean(error(fil_idx,1,:));
    
        if avgerr < 0.5
            crcnt(fil_idx) = 1;
        end
    
        temperror = zeros(1,length(error(fil_idx,1,:)));
        temperror(:) = error(fil_idx,1,:);
    
        if figure_view
            figure(fig2);
            h2(fil_idx) = plot(t,estState(1,:),...
                'Color',colors{fil_idx},'Marker',markers{fil_idx},...
                'MarkerIndices',marker_indices,...
                'LineWidth',1,'MarkerSize',6,...
                'DisplayName',labels{fil_idx});

            figure(fig3);
            h3(fil_idx) = plot(t,estState(2,:),...
                'Color',colors{fil_idx},'Marker',markers{fil_idx},...
                'MarkerIndices',marker_indices,...
                'LineWidth',1,'MarkerSize',6,...
                'DisplayName',labels{fil_idx});
        
            figure(fig4);
            h4(fil_idx) = plot(t,temperror,...
                'Color',colors{fil_idx},'Marker',markers{fil_idx},...
                'MarkerIndices',marker_indices,...
                'LineWidth',1,'MarkerSize',6,...
                'DisplayName',labels{fil_idx});
        end
    end

    %% UKF
    if comparison_UKF
        %% Filter parameters
        fil_idx = 7;

        %% Initialisation
        estState = zeros(n,numStep);
        temp_particle_var = zeros(n,n,numStep);
    
        %% First time step
        x = initial;
        P = P0;
        
        estState(:,1) = x;
        temp_particle_var(:,:,1) = P;
    
        error(fil_idx,:,1) = abs(estState(:,1)-trueState(:,1));
        chisq(fil_idx,1) = error(fil_idx,:,1)*(temp_particle_var(:,:,1)\error(fil_idx,:,1)');
    
        %% Main loop
        tStart7 = tic; 

        [estState,x,P] = UKF(numStep,n,nm,meas,estState,x,P,...
                             numSigma,alpha,beta,lambda,...
                             Q,R,[],[],G,funsys,funmeas,[]);
    
        %% For evaluation
        for i = 2:numStep
            error(fil_idx,:,i) = abs(estState(:,i)-trueState(:,i));
            chisq(fil_idx,i) = error(fil_idx,:,i)*(P\error(fil_idx,:,i)');    
    
            if any(eig(P) > 1.0e+04)
                divcnt = 1;
                break
            end
        end
        
        elpst(fil_idx) = toc(tStart7);
        avgerr = mean(error(fil_idx,1,:));
    
        if avgerr < 0.5
            crcnt(fil_idx) = 1;
        end
    
        temperror = zeros(1,length(error(fil_idx,1,:)));
        temperror(:) = error(fil_idx,1,:);
    
        if figure_view
            figure(fig2);
            h2(fil_idx) = plot(t,estState(1,:),...
                'Color',colors{fil_idx},'Marker',markers{fil_idx},...
                'MarkerIndices',marker_indices,...
                'LineWidth',1,'MarkerSize',6,...
                'DisplayName',labels{fil_idx});

            figure(fig3);
            h3(fil_idx) = plot(t,estState(2,:),...
                'Color',colors{fil_idx},'Marker',markers{fil_idx},...
                'MarkerIndices',marker_indices,...
                'LineWidth',1,'MarkerSize',6,...
                'DisplayName',labels{fil_idx});
        
            figure(fig4);
            h4(fil_idx) = plot(t,temperror,...
                'Color',colors{fil_idx},'Marker',markers{fil_idx},...
                'MarkerIndices',marker_indices,...
                'LineWidth',1,'MarkerSize',6,...
                'DisplayName',labels{fil_idx});
        end
    end

    %% SIR
    if comparison_SIR
        %% Filter parameters
        fil_idx = 8;

        %% Initialisation
        particles = zeros(n,numParticles,numStep);
        weights = zeros(numParticles,numStep);
    
        estState = zeros(n,numStep); % storage for state estimates
        temp_particle_var = zeros(n,n,numStep);
    
        %% First time step
        particles(:,:,1) = initial+sqrt(P0)*randn(n,numParticles);    
        weights(:,1) = 1/numParticles*ones(numParticles,1);
    
        estState(:,1) = mean(particles(:,:,1),2);
        temp_particle_var(:,:,1) = P0;
    
        error(fil_idx,:,1) = abs(estState(:,1)-trueState(:,1));
        chisq(fil_idx,1) = error(fil_idx,:,1)*(temp_particle_var(:,:,1)\error(fil_idx,:,1)');
    
        particle_pos = funpos(particles(:,:,1));

        if animation_view
            fig99 = figure; hold on; axis equal; legend;
    
            estPos = funpos(estState(:,1));
            plot(L*cos(theta),L*sin(theta),'k:','HandleVisibility','off');
            fig_partPos = plot(particle_pos(1,:),particle_pos(2,:),'r.','DisplayName','Particles');
            fig_estPos = plot(estPos(1),estPos(2),'bo','DisplayName','Estimated','MarkerFaceColor','b');
            fig_truePos = plot(truePos(1,1),truePos(2,1),'ko','DisplayName','True','MarkerFaceColor','k');
            range = plot([0 truePos(1,1)],[0 truePos(2,1)],'k:','HandleVisibility','off');
            title('True and Estimated Positions')
        end
        
        %% Main loop
        tStart8 = tic; 
    
        [estState,particles,weights] = ...
                SIR(numStep,n,nm,meas,numParticles,estState,particles,weights,...
                    Q,R,[],[],funsys,funmeas,[],[]);
    
        %% For evaluation
        for i = 2:numStep
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
    
            error(fil_idx,:,i) = abs(estState(:,i)-trueState(:,i));
            chisq(fil_idx,i) = error(fil_idx,:,i)*(temp_particle_var(:,:,i)\error(fil_idx,:,i)');
    
            particle_pos = funpos(particles(:,:,i));
            if animation_view
                delete(fig_truePos)
                delete(fig_partPos)
                delete(fig_estPos)
                delete(range)
                estPos = funpos(estState(:,i));
                fig_partPos = plot(particle_pos(1,:),particle_pos(2,:),'r.','DisplayName','Particles');
                fig_estPos = plot(estPos(1),estPos(2),'bo','DisplayName','Estimated','MarkerFaceColor','b');
                fig_truePos = plot(truePos(1,i),truePos(2,i),'ko','DisplayName','True','MarkerFaceColor','k');
                range = plot([0 truePos(1,i)],[0 truePos(2,i)],'k:','HandleVisibility','off');
                drawnow;
            end
        end
    
        elpst(fil_idx) = toc(tStart8);
        avgerr = mean(error(fil_idx,:));
        
        if avgerr < 0.5
            crcnt(fil_idx) = 1;
        end
    
        temperror = zeros(1,length(error(fil_idx,1,:)));
        temperror(:) = error(fil_idx,1,:);
    
        if figure_view
            figure(fig2);
            h2(fil_idx) = plot(t,estState(1,:),...
                'Color',colors{fil_idx},'Marker',markers{fil_idx},...
                'MarkerIndices',marker_indices,...
                'LineWidth',1,'MarkerSize',6,...
                'DisplayName',labels{fil_idx});

            figure(fig3);
            h3(fil_idx) = plot(t,estState(2,:),...
                'Color',colors{fil_idx},'Marker',markers{fil_idx},...
                'MarkerIndices',marker_indices,...
                'LineWidth',1,'MarkerSize',6,...
                'DisplayName',labels{fil_idx});
        
            figure(fig4);
            h4(fil_idx) = plot(t,temperror,...
                'Color',colors{fil_idx},'Marker',markers{fil_idx},...
                'MarkerIndices',marker_indices,...
                'LineWidth',1,'MarkerSize',6,...
                'DisplayName',labels{fil_idx});
        end
    
        if recordVid_view
            writeObj = VideoWriter('SIR','MPEG-4');
            writeObj.FrameRate = framerate;
            open(writeObj);
            writeVideo(writeObj,frame);
            close(writeObj);
        end
    end

    %% End of parfor
    errorall(:,:,:,mc) = error;
    chisqall(:,:,mc) = chisq;
    elpsdtime(:,mc) = elpst;

    correctcnt(:,mc) = crcnt;
    divergecnt(:,mc) = divcnt;

    cmeanall(:,:,:,:,mc) = cmeanmc;

    if figure_view
        labels = {'PGM-DS','AKKF','PGM-DU','EnKF','PGM1','UKF','PGM1-UT','SIR'};

        figure(fig2);
        legend(h2(remap_columnwise),labels,'NumColumns',4,'Location','north');
        ylim([-3 3])
        
        ax_main = gca;
        ax_true = axes('Position',get(ax_main,'Position'),...
                       'Color','none',...
                       'XColor','none','YColor','none');
        lgd_true = legend(ax_true,h_true,'True','Location','southeast');
        set(lgd_true,'Box','off');

        figure(fig3);
        legend(h3(remap_columnwise),labels,'NumColumns',4,'Location','north');
        ylim([-4.5 4.5])
        
        ax_main = gca;
        ax_true = axes('Position',get(ax_main,'Position'),...
                       'Color','none',...
                       'XColor','none','YColor','none');
        lgd_true = legend(ax_true,h_true_2,'True','Location','southeast');
        set(lgd_true,'Box','off');
    
        figure(fig4);
        legend(h4(remap_columnwise),labels,'NumColumns',4,'Location','north');
        ylim([0 4.5])
    end
end

%% Evaluations
% For plot
labels = {'PGM-DS','AKKF','PGM-DU','EnKF','PGM1','UKF','PGM1-UT','SIR'};
markers = {'s','^','d','v','p','h','x','+'};
colors  = {'r','g','#0072BD','#EDB120','m','c','#4DBEEE','#77AC30'};
remap_columnwise = [1 5 2 6 3 7 4 8];
marker_indices = 1:1:length(t);

comparison_flags = [ ...
    comparison_PGM_DS,...
    comparison_PGM_DU,...
    comparison_PGM1,...
    comparison_PGM1_UT,...
    comparison_AKKF,...
    comparison_EnKF,...
    comparison_UKF,...
    comparison_SIR];

comparison_flags =  1 - comparison_flags;

% Root means squared error (RMSE)
for fil = 1:8
    for i = 1:numStep
        for mc = 1:numMC
            errornorm(fil,i,mc) = vecnorm(errorall(fil,:,i,mc));
        end
    end
end

for fil = 1:8
    for i = 1:numStep
        rmse(fil,i) = sqrt(sum(errornorm(fil,i,1:numMC).^2)/numMC);
    end
end

% Averaged RMSE
for fil = 1:8
    armse(fil) = sum(rmse(fil,:))/numStep;  % PGM-UT/PGM/UKF/SIR
    cnttotal(fil) = sum(correctcnt(fil,:));
end

divall = sum(divergecnt(:,:));

% Plot RMSE
figure; hold on; grid on;
h = gobjects(1,8);  % Preallocate

for k = 1:8
    if comparison_flags(k)
        h(k) = plot(NaN,NaN,...
            'Color',colors{k},...
            'Marker',markers{k},...
            'LineWidth',1,...
            'MarkerSize',6,...
            'DisplayName',labels{k},...
            'Visible','off');
    else
        h(k) = plot(t,rmse(k,:),...
            'Color',colors{k},...
            'Marker',markers{k},...
            'LineWidth',1,...
            'MarkerSize',6,...
            'DisplayName',labels{k},...
            'MarkerIndices',marker_indices);
    end
end

legend(h(remap_columnwise),labels,'NumColumns',4,'Location','north');

xlim([0 tf])
ylim([0 8.5])
xlabel('Time (sec)')
ylabel('RMSE')
hold off;

% Chi-Square
for fil = 1:8
    for i = 1:numStep
        chinorm(fil,i) = sum(chisqall(fil,i,:))/(n*numMC);
    end
end

for fil = 1:8
    achi(fil) = sum(chinorm(fil,:))/numStep;
end

% Counting Chi-squared
for fil = 1:8
    for i = 1:numStep
        for mc = 1:numMC
            if chisqall(fil,i,mc) > chithres
                chicntall(fil,i,mc) = 1;
            end
        end
    end
end

for fil = 1:8
    for i = 1:numStep
        chicntfin(fil,i) = sum(chicntall(fil,i,:) == 1);
        chifrac(fil,i) = chicntfin(fil,i)/numMC;
    end
end

for fil = 1:8
    chifracavg(fil) = mean(chifrac(fil,:),2);
end

% Plot Chisq percentage cases
figure; hold on; grid on;
h = gobjects(1,8);  % Preallocate

for k = 1:8
    if comparison_flags(k)
        h(k) = plot(NaN,NaN,...
            'Color',colors{k},...
            'Marker',markers{k},...
            'LineWidth',1,...
            'MarkerSize',6,...
            'DisplayName',labels{k},...
            'Visible','off');
    else
        h(k) = plot(t,chifrac(k,:),...
            'Color',colors{k},...
            'Marker',markers{k},...
            'LineWidth',1,...
            'MarkerSize',6,...
            'DisplayName',labels{k},...
            'MarkerIndices',marker_indices);
    end
end

legend(h(remap_columnwise),labels,'NumColumns',4,'Location','north');

xlim([0 tf])
ylim([0 1.15])
xlabel('Time (sec)')
ylabel('Cases exceeding 99.99% \chi^2 bound (%)');
ax = gca;

yticks = get(ax,'YTick');
yticklabels = arrayfun(@(y) sprintf('%.0f%%',y * 100),yticks,'UniformOutput',false);
set(ax,'YTickLabel',yticklabels);
hold off;

elpadtavg = mean(elpsdtime,2);

%% Save variables
save('2_Pendulum_MC')