function [estState,x,P] = ...
         EKF(numStep,n,nm,meas,estState,x,P,...
             Q,R,F,H,JF,JH,G,funsys,funmeas,interm)
        % EKF but if Flinear and Hlinear are both true, same as KF

    % Main loop
    for i = 2:numStep
        % Prediction step
        if isempty(funsys) && isempty(JF)
            x = F*x;
        elseif ~isempty(funsys) && isempty(JF)
            x = funsys(x,0,i-1,8);
        elseif ~isempty(funsys) && ~isempty(JF)
            F = JF(x);
            x = funsys(x,0,i-1,8);
        else
            F = JF(x);
            x = F*x;
        end

        P = F*P*F'+G*Q*G';

        % Update step
        if ~isempty(interm)
            if rem(i,interm) == 0
                if isempty(funmeas) && isempty(JH)
                    residual = meas(:,i)-H*x;
                elseif ~isempty(funmeas) && isempty(JH)
                    residual = meas(:,i)-funmeas(x,0);
                elseif ~isempty(funmeas) && ~isempty(JH)
                    H = JH(x);
                    residual = meas(:,i)-funmeas(x,0);
                else
                    H = JH(x);
                    residual = meas(:,i)-H*x;
                end

                K = P*H'/(H*P*H'+R*eye(nm));   
                x = x+K*residual;
                P = (eye(n)-K*H)*P;  
            end
        else
            if isempty(funmeas) && isempty(JH)
                residual = meas(:,i)-H*x;
            elseif ~isempty(funmeas) && isempty(JH)
                residual = meas(:,i)-funmeas(x,0);
            elseif ~isempty(funmeas) && ~isempty(JH)
                H = JH(x);
                residual = meas(:,i)-funmeas(x,0);
            else
                H = JH(x);
                residual = meas(:,i)-H*x;
            end

            K = P*H'/(H*P*H'+R*eye(nm));   
            x = x+K*residual;
            P = (eye(n)-K*H)*P;  
        end      

        estState(:,i) = x;
    end
end