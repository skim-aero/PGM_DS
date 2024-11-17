function [estState,x,P] = ...
         EKF(numStep,n,nm,meas,estState,x,P,...
             Q,R,F,H,JF,JH,G,funsys,funmeas,Flinear,Hlinear,interm)
        % EKF but if Flinear and Hlinear are both true, same as KF

    % Main loop
    for i = 2:numStep
        % Prediction step
        if Flinear
            x = F*x;
        else
            if ~isempty(funsys)
                F = JF(x);
                x = funsys(x,0,i-1);
            else
                F = JF(x);
                x = F*x;
            end
        end

        P = F*P*F'+G*Q*G';

        % Update step
        if interm
            if rem(i,20) == 0
                if Hlinear
                    residual = meas(:,i)-H*x;
                else
                    if ~isempty(funmeas)
                        H = JH(x);
                        residual = meas(:,i)-funmeas(x,0);
                    else
                        H = JH(x);
                        residual = meas(:,i)-H*x;
                    end
                end

                K = P*H'/(H*P*H'+R*eye(nm));   
                x = x+K*residual;
                P = (eye(n)-K*H)*P;  
            end
        else
            if Hlinear
                residual = meas(:,i)-H*x;
            else
                if ~isempty(funmeas)
                    H = JH(x);
                    residual = meas(:,i)-funmeas(x,0);
                else
                    H = JH(x);
                    residual = meas(:,i)-H*x;
                end
            end

            K = P*H'/(H*P*H'+R*eye(nm));   
            x = x+K*residual;
            P = (eye(n)-K*H)*P;  
        end      

        estState(:,i) = x;
    end
end