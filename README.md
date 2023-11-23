# cs462 Programing Assignment 5
Vincent Broda

For this assignment, I created mean-variance.cu, which is a c program that used Nvidia's Cuda to calculate the mean and variance of an array of numbers.
In the program's current state, this array is simply all of the integers between 1 and 1,000,000 (inclusive), however it could, and was for testing, reworked to incorporate random numbers, as well as diffrent sized arrays.
The diffrent sized arrays can still be easily adjusted by altering N.

When making the code, I tested out to make sure I was getting the correct awnser two different ways. At first, I used simple online variance/mean calculator (https://www.calculatorsoup.com/calculators/statistics/variance-calculator.php) where I would just check my output with the sites. This was do-able when I wouold set N to be a smaller number, so I did this for 2-10 to make sure their where no inconsistencies at first. After this, I wanted to check that my output was correct for the million sized array. I was confident that the mean was correct, since that follows an easy to tell pattern, being about half of the max value when we do 1-N, but I was less sure in the variance that I was getting. To check this, I wrote a simple python script that also calculated the variance of the same data set. 
This is that script:
import numpy as np
data = np.arange(1, 1000001)
mean_value = np.mean(data)
variance_value = np.var(data, ddof=1)
print("Mean:", mean_value)
print("Variance:", variance_value)

Doing this, I got an output of Mean: 500000.5, Variance: 83333416666.66667, which was consistent on multiple runs.
This was a great sign, as my program outputs something like:
Mean: 499939.937500
Variance: 83301498880.000000
I say something like because when the data set gets this large, the output becomes a little indeterministic. I belive that this is from floating point percision, and more specifically from using single precision. However, from both the smaller sample tests, which had deterministic and correct output, and how close this is to my python output, I am confident in the basic logic of the program with doing these clalculations. As well, this is one of the further off results that I have gotten, and now It seems that I get much more accurate results.

Another thing to add is that I did basically all of my testing on my desktop, as I happen to have an Nvidia graphics card (GTX1070) on it. The set up was not to bad after looking it up, and I ended up using visual studio to run and debug my program. This was really cool I thought, as now I feel like I have the power to do insaine calculations on my computer now. It is worth noting that the only time I also had any preformance drops was when I was trying to use print statements to debug. Because of this, how I compiled was a bit different, but I belive that basically everything else was about the same.

To run it on our nodes, here are just some of the commands that I did, mostly here for my own reference.
ssh vbroda@acf-login.acf.tennessee.edu
salloc -A isaac-utk0252 -q campus-gpu -p campus-beacon-gpu -t 1:00:00 -N 1 --gpus=1
w
ssh acf-bkxxx
module load cuda
nvcc mean-variance.cu -o mean-variance
./mean-variance

I had trouble with this the first time I tried it, for some reason I didn't get a gpu, but I tried again and It worked fine for some reason. Perhaps this was becasuse when I decided to test it I think everyone else in the class was. However, when I did get it to work I got results that matched up to what I was expecting.

When making the actual code, it was quite simple. We allocated out host memory, with malloc, and out device memory, with cudaMalloc. We send over the data to initialize it or when the work is done. We do our work in two functions, both with a similar idea. We will be the summation parts of these calculations with the gpu, meaning we will add all of the numbers up in mean, and in the varince function, we will get the summation of the difference between the data point and the mean, then square it. In both cases, we will do our division after these summations are done, which is acceptable by the math and more efficient to way to do these calculations, since divsion can be expensive. It is worth noting that I belive doing division within the functions would also lead to a more indeterminate output, even more so that we allready are experiencing, beccause of floating point opperations not being associative.
I also implimented the code using sample variance, meaning we do the division with N - 1 instead of  just N like how the mean is calculated.

I also did not make the code customizable with any command line arguments, since I thought it was much easier to make a change in the code and recompile for somereason when I made this. This helps with simplicity becasuse to run we should just have to do make and then ./mean-variance to get the output. 
