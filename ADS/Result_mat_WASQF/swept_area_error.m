% Compute the Swept Area Error between 2 position trajectories.
% 
% Author: Matteo Saveriano
%
% Copyright (c) 2021 Matteo Saveriano, Dept. of Computer Science and 
% Digital Science Center, University of Innsbruck, 6020, Innsbruck,
% Austria, https://iis.uibk.ac.at
%

function sArea = swept_area_error(xReal, xDemo)
    %% Resample real trajectory
    [xR, isIncreasing] = resample_trajectory(xReal, xDemo);
    
    if(~isIncreasing)
        sArea = 1e6;
        return;
    end        
    
    %% Compute Swept Area using Gauss formula (Shoelace theorem)
    N = length(xR);
    sArea = 0;
    for i = 1:N-1        
        vX = [xR(1,i) xR(1,i+1) xDemo(1,i) xDemo(1,i+1)]';
        vY = [xR(2,i) xR(2,i+1) xDemo(2,i) xDemo(2,i+1)]'; 
          
        % Compute convex-hull to make area independent on the point order
        K = convhull(vX,vY);
        
        sArea = sArea + polyarea(vX(K),vY(K));       
    end    
end