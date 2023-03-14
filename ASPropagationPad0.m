function U2  = ASPropagationPad0(U1,z,ps,lambda,bandlimit)
% angluar spectrum propagation using Fresnel transfer function
% digitally prop complex field U1 to U2 at z far
%'Inputs': 'U1',input complex field, |Phasor|
%          'z',Propagation distance |meter| 
%          'ps', pixelsize |meter| 
%          'lambda',Wavelength |meter| 
% 'Outputs': 'U1',Complex field after propagation; |Phasor|
% note that different units are OK if they are the same.
% Ref: https://github.com/flyingwolfz/angular-spectrum-method
[N,M] = size(U1);
Lx =2 * ps * M;
Ly =2 * ps * N;
k = 2*pi/lambda;
fx = -1/(2*ps) : 1/Lx : (1/(2*ps) - 1/Lx);
fy = -1/(2*ps) : 1/Ly : (1/(2*ps) - 1/Ly);
[Fx,Fy] = meshgrid(fx,fy);
H_AS = exp(1i*k*z*sqrt(1-(lambda*Fx).^2-(lambda*Fy).^2));
% SincF = 1*lambda/0.01;
if(strcmp(bandlimit,'limit'))  %%% better 2z >> ps * N
    fxmax = 1/(1*102*ps);
    fymax = 1/(1*102*ps);
%     fxmax = 1/Lx;
%     fymax = 1/Ly;
    fxlimit=1/sqrt((2*fxmax*z)^2+1)/lambda;
    fylimit=1/sqrt((2*fymax*z)^2+1)/lambda;
    H_AS(abs(Fx)>fxlimit)=0;
    H_AS(abs(Fy)>fylimit)=0;
end
% H_AS = H_AS.*sinc(2*Fx).^2.*sinc(2*Fy).^2;
H_AS = fftshift(H_AS);

U1 = padarray(U1,[M/2 N/2],'both');

U1F = fft2(fftshift(U1));
U2F = U1F .* H_AS;

U2 = ifftshift(ifft2(U2F));
U2 = U2(M/2+1:M/2+M, N/2+1:N/2+N);

