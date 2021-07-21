## Peak detection using MATLAB

A peak is the highest point betweem "valleys". What makes a peak is the fact that there are lower points around it. This strategy is adopted by "peakdet": Look for the highest point, around which there are points lower by X on both sides.

First, let's create the graph shown in the figure above:

t=0:0.001:10;
x=0.3*sin(t) + sin(1.3*t) + 0.9*sin(4.2*t) + 0.02*randn(1, 10001);
figure; plot(x);

Now we'll find the peaks and valleys: (you'll need to copy the "peakdet" function from the bottom of this page and put it in your working directory or a directory in the MATLAB search path):

[maxtab, mintab] = peakdet(x, 0.5);
hold on; plot(mintab(:,1), mintab(:,2), 'g*');
plot(maxtab(:,1), maxtab(:,2), 'r*');
