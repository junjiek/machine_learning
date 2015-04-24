%NONLINEAR_SVM Summary of this function goes here
%   Detailed explanation goes here

%generate data1
r=sqrt(rand(100,1));%generate 100 random radius
t=2*pi*rand(100,1);%generate 100 random angles, in range [0,2*pi]
data1=[r.*cos(t),r.*sin(t)];%points

%generate data2
r2=sqrt(3*rand(100,1)+1);%generate 100 random radius
t2=2*pi*rand(100,1);%generate 100 random angles, in range [0,2*pi]
data2=[r2.*cos(t2),r2.*sin(t2)];%points

%plot datas
 plot(data1(:,1),data1(:,2),'r.')
 hold on
plot(data2(:,1),data2(:,2),'b.')
% ezpolar(@(x)1);%在极坐标下画ρ=1，θ∈[0,2π]的图像，即x^2+y^2=1
% ezpolar(@(x)2);
axis equal %make x and y axis with equal scalar
hold off

%build a vector for classification
data=[data1;data2];     %merge the two dataset into one
datalabel=[ones(100,1); zeros(100, 1)];  %label for the data
datalabel(1:100)=-1;

%train with Non-linear SVM classifier use Gaussian Kernel

model=svmtrain(datalabel,data,'-c 100 -g 4 -b 1'); 
visualizeBoundary(data, datalabel, model);