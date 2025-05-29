% Equidistantly resample a motion trajectory xReal to length(xDemo) 
% 
% Author: Matteo Saveriano
%
% Copyright (c) 2021 Matteo Saveriano, Dept. of Computer Science and 
% Digital Science Center, University of Innsbruck, 6020, Innsbruck,
% Austria, https://iis.uibk.ac.at
%

function [xRealRes, isIncreasing] = resample_trajectory(xReal, xDemo)
    N = length(xDemo);
    
    % Interpolate points inbetween
    O(:,1)=interp(xReal(1,:)',1);
    O(:,2)=interp(xReal(2,:)',1);

    % Calculate distance between points
    dis=[0;cumsum(sqrt(sum((O(2:end,:)-O(1:end-1,:)).^2,2)))];
    
    % Avoid error in interp1 due to non-increasing distances
    isIncreasing = all(diff(dis));

    if(~isIncreasing)
        xRealRes = 1:N;
    else
        % Resample to make uniform points
        K(:,1) = interp1(dis,O(:,1),linspace(0,dis(end),N));
        K(:,2) = interp1(dis,O(:,2),linspace(0,dis(end),N));

        % Return aligned trajectory
        xRealRes = K';
    end
end