function F = Lorenz4DJ(X)
    % Lorenz 96 Jacobian

    J = length(X);
    F = zeros(J,J); % Initialise Jacobian matrix

    % Compute the Jacobian matrix
    for i = 1:J
        % Periodic boundary
        im1 = mod(i-2,J)+1;
        im2 = mod(i-3,J)+1;
        ip1 = mod(i,J)+1;

        % Derivativation
        F(i,im2) = -X(im1);        % d(k[i])/d(X[im2])
        F(i,im1) = X(ip1)-X(im2);% d(k[i])/d(X[im1])
        F(i,i) = -1;             % d(k[i])/d(X[i])
        F(i,ip1) = X(im1);         % d(k[i])/d(X[ip1])
    end
end
