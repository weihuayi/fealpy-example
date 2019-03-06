load amg.mat

theta = 0.025
disp(size(A))
disp(size(b))

[Ac,Pro,Res] = coarsenAMGrs(A,theta)
