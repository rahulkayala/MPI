// Distributed two-dimensional Discrete FFT transform
// RAHUL KAYALA, 903038578
// ECE8893 Project 1


#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <signal.h>
#include <math.h>
#include <mpi.h>

#include "Complex.h"
#include "InputImage.h"


using namespace std;

int imageWidth, imageHeight, nCPU, rank, nRow, rc;

Complex W(double N,  int n, int k, bool inverse)
{
	if(inverse)
	{
		return Complex((1/N)*cos(2*M_PI*n*k/N),(1/N)*sin(2*M_PI*n*k/N));
	}
	else
	{
		return Complex(cos(2*M_PI*n*k/N),-sin(2*M_PI*n*k/N));
	}

}

void Transform1D(Complex* h, int w, Complex* H, bool inverse)
{

	Complex* tempIP = new Complex[w];
	Complex* tempOP = new Complex[w];  

	for(int i=0; i < nRow; i++ )
	{
		for(int j=0; j < w; j++)
		{
			tempIP[j]=h[j+i*w];
		}
	
		for(int n=0; n<w; n++)
		{
			tempOP[n] = Complex(0,0);
			
			for(int k=0;k<w; k++)
			{
				tempOP[n] = tempOP[n] + W((double)w, n, k, inverse)*tempIP[k];	
			}
		}
		
		for (int j=0; j<w; j++)
		{
			H[j+i*w] = tempOP[j]; 
		}
	}
}

void Transform2D(const char* inputFN, const char* outputFN, bool inverse) 
{
	//Opening the Image  
	InputImage image(inputFN);
	Complex* inputImageData = image.GetImageData();
	imageWidth = image.GetWidth();
	imageHeight = image.GetHeight();

	//Rows per Processor
 	nRow = imageHeight/nCPU;

	//Getting 1D Transform Row by Row	
	Complex* before1D = new Complex[nRow*imageWidth];
	Complex* after1D = new Complex[nRow*imageWidth];
	

	for(int j=0; j < nRow*imageWidth; j++)
	{
		before1D[j]=inputImageData[rank*nRow*imageWidth + j];
	}


	Transform1D(before1D, imageWidth, after1D, inverse);
	MPI_Barrier(MPI_COMM_WORLD);

	//Preparing for AllGather
	Complex *gatherBuffer1D = new Complex[imageWidth*imageHeight];
	MPI_Allgather(after1D, nRow*imageWidth, MPI_LONG_DOUBLE, gatherBuffer1D, nRow*imageWidth, MPI_LONG_DOUBLE, MPI_COMM_WORLD);

	//Getting 2D Transform Column by Column
	Complex* transform2D = new Complex[imageHeight*imageWidth];
	
	//Transforming into Columnwise
	for(int i=0; i < imageWidth; i++)
	{
		for(int j=0; j < imageHeight; j++)
			transform2D[j+i*imageWidth] = gatherBuffer1D[j*imageHeight + i];
	}

	
	Complex* before2D = new Complex[nRow*imageWidth];
	Complex* after2D = new Complex[nRow*imageWidth];
	for(int j=0; j < nRow*imageWidth; j++)
	{
		before2D[j]=transform2D[rank*nRow*imageWidth + j];
	}

	Transform1D(before2D, imageHeight, after2D, inverse);
	MPI_Barrier(MPI_COMM_WORLD);
	
	//Setting up for AllGather
	Complex* gatherBuffer2D = new Complex[imageHeight*imageWidth];
	MPI_Allgather(after2D, nRow*imageWidth, MPI_LONG_DOUBLE, gatherBuffer2D, nRow*imageWidth, MPI_LONG_DOUBLE, MPI_COMM_WORLD);

	for(int i=0; i < imageHeight; i++)
	{
		for(int j=0; j < imageWidth; j++)
			transform2D[j+i*imageHeight] = gatherBuffer2D[j*imageWidth + i];
	}
	
	//Writing to Output File
	if(rank == 0)
	image.SaveImageData(outputFN.c_str(),transform2D,imageWidth,imageHeight); 
}

 


int main(int argc, char** argv)
{
  string fn("Tower.txt"); // default file name
  if (argc > 1) fn = string(argv[1]);  // if name specified on cmd line

  // MPI initialization here
  rc=MPI_Init(&argc,&argv);
  if(rc!=MPI_SUCCESS)
  {
	cout<<"Unexpected Error Occured";
	MPI_Abort(MPI_COMM_WORLD, rc);
  }
	
  MPI_Comm_size(MPI_COMM_WORLD,&nCPU);
  MPI_Comm_rank(MPI_COMM_WORLD,&rank);

  //Start The 2D Forward DFT 
  Transform2D(fn.c_str(),"MyAfter2d.txt",false); 
  
  //Start the 2D IDFT
  Transform2D("MyAfter2d.txt","MyAfterInverse.txt",true);
  
  // Finalize MPI here
  MPI_Finalize();
}  
  

  
