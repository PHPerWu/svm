#include <iostream>
#include <sstream>
#include <string>
#include <fstream>
#include <time.h>
#include <Windows.h>
#include <iomanip>
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/ml/ml.hpp>
using namespace std;
using namespace cv;
using namespace ml;
uchar *lbpData;
uchar *lbpStr;
void ShowTime()
{
	time_t tt = time(NULL);//这句返回的只是一个时间cuo
	tm* t = localtime(&tt);
	printf("%d-%02d-%02d %02d:%02d:%02d\n",
		t->tm_year + 1900,
		t->tm_mon + 1,
		t->tm_mday,
		t->tm_hour,
		t->tm_min,
		t->tm_sec);
}
void inputArray(float labels[],int dim,float data[])
{
	for (int i = 0; i < dim; i++)
	{
		labels[i] = data[i];
	}
}
void CheckImage(string name,Mat img,int time=0)
{
	imshow(name, img);
	uchar key = waitKey(time);
	switch (key)
	{
	case 's':
		cout << "please print save name" << endl;
		cin >> name;
		imwrite("..\\image\\" + name + ".png", img);
		break;
	case 'S':
		cout << "please print save name" << endl;
		cin >> name;
		imwrite("..\\image\\" + name + ".png", img);
		break;
	}

}
//求三个数的中位数  
int mid(int a, int b, int c)
{
	int max = a;
	int min = b;
	if ((a <= b&&b <= c) || (c <= b&&b <= a))
	{
		return b;
	}
	else if ((b <= a&&a <= c) || (c <= a&&a <= b))
	{
		return a;
	}
	else
	{
		return c;
	}
}
// 计算跳变次数  
int getHopCount(int i)
{
	int a[8] = { 0 };
	int cnt = 0;
	int k = 7;
	while (i)
	{
		a[k] = i & 1;
		i = i >> 1;
		--k;
	}
	for (k = 0; k < 7; k++)
	{
		if (a[k] != a[k + 1])
		{
			++cnt;
		}
	}
	if (a[0] != a[7])
	{
		++cnt;
	}
	return cnt;
}

// 降维数组 由256->59  
void lbp59table(uchar *table)
{
	memset(table, 0, 256);
	uchar temp = 1;
	for (int i = 0; i < 256; i++)
	{
		if (getHopCount(i) <= 2)    // 跳变次数<=2 的为非0值  
		{
			table[i] = temp;
			temp++;
		}
	}
}


void convert59(Mat &image, Mat &result, uchar *table)
{
	int height = image.rows;
	int width = image.cols;

	for (int y = 0; y < height; y++)
	{
		for (int x = 0; x < width; x++)
		{
			int temp = image.at<uchar>(y, x);
			//cout << temp << " ";
			temp = table[image.at<uchar>(y, x)];
			result.at<uchar>(y, x) = table[image.at<uchar>(y, x)];   //  降为59维空间  
			//cout << setw(3) << (int)table[image.at<uchar>(y, x)] << " ";
		}
		//cout << endl;
	}
}
void uniformLBP(Mat &image, Mat &result)
{
	int height = image.rows;
	int width = image.cols;
	for (int y = 0; y < height; y++)
	{
		for (int x = 0; x < width; x++)
		{
			uchar neighbor[8] = { 0 };
			neighbor[0] = image.at<uchar>(mid(y + 1, 0, height - 1), mid(x + 1, 0, width - 1));
			neighbor[1] = image.at<uchar>(mid(y + 1, 0, height - 1), mid(x, 0, width - 1));
			neighbor[2] = image.at<uchar>(mid(y + 1, 0, height - 1), mid(x - 1, 0, width - 1));
			neighbor[3] = image.at<uchar>(mid(y, 0, height - 1), mid(x + 1, 0, width - 1));
			neighbor[4] = image.at<uchar>(mid(y, 0, height - 1), mid(x - 1, 0, width - 1));
			neighbor[5] = image.at<uchar>(mid(y - 1, 0, height - 1), mid(x + 1, 0, width - 1));
			neighbor[6] = image.at<uchar>(mid(y - 1, 0, height - 1), mid(x, 0, width - 1));
			neighbor[7] = image.at<uchar>(mid(y - 1, 0, height - 1), mid(x - 1, 0, width - 1));
			uchar center = image.at<uchar>(mid(y, 0, height - 1), mid(x, 0, width - 1));
			uchar temp = 0;
			for (int k = 0; k < 8; k++)
			{
				temp += (neighbor[k] >= center)* (1 << k);  // 计算LBP的值  
			}
			//cout << setw(3) << (int)temp << " ";
			result.at<uchar>(y, x) = temp;

		}
		//cout << endl;
	}
}



int main()
{
	int endInput;
	uchar table[256];
	lbp59table(table);
	stringstream tempPath;
	lbpData = new uchar[5900];
	Mat image;
	Ptr<SVM> svm = SVM::create();
	svm->setType(SVM::C_SVC);
	svm->setKernel(SVM::LINEAR);
	svm->setTermCriteria(TermCriteria(CV_TERMCRIT_ITER, 1000, 1e-6));
	//CheckImage("trainingDataMat", trainingDataMat);
	//CheckImage("labelsMat", labelsMat);
	int picNum = 10000;
	int *labels = new int[picNum];
	float **trainingData = new float*[picNum];
	string rPath = "E:\\age\\IMDB\\imdb_crop\\";
	int dim = 64;
	int dimdim = dim*dim;
	
	ifstream info("E:\\age\\IMDB\\infoFileW.txt");
	if (!info.is_open())
	{
		cout << "can't open the info file!" << endl;
		cin >> endInput;
		return 0;
	}
	
	for (int train_count = 0; train_count < 2; train_count++)
	{
		
		
		string path,eof;
		int gender;
		info >> path;
		info >> gender;
		info >> eof;
		eof=info.peek();
		while ("\n" != eof)
		{
			info >> eof;
			eof = info.peek();
		}
		
		Mat Oimage = imread(rPath+ path, 0);
		resize(Oimage, image, Size(dim, dim));

		//Mat image;
		if (!image.data)
		{
			cout << "Fail to load image" << endl;
			return 0;
		}
		//resize(OM, image, Size(10, 10));
		Mat result, result59;
		result.create(Size(dim,dim), image.type());
		result59.create(Size(dim,dim), image.type());

		uniformLBP(image, result);
		convert59(result, result59, table);
		float *feature = new float[dimdim];

		labels[train_count] = gender;
		trainingData[train_count] = new float[dimdim];
		for (int y = 0, j = 0; y < result59.rows; y++)
		{
			for (int x = 0; x < result59.cols; x++,j++)
			{
				trainingData[train_count][j] = (float)result59.at<uchar>(y, x);
			}
		}
	}
	Mat labelsMat(picNum, 1, CV_32SC1, labels);
	Mat trainingDataMat(picNum,dimdim, CV_32FC1, trainingData);
	stringstream saveNameStream;
	int i = 0;
	while (true)
	{
		string savePath;
		saveNameStream << i;
		saveNameStream >> savePath;
		saveNameStream.clear();
		if (svm->train(trainingDataMat, ROW_SAMPLE, labelsMat))
		{
			savePath = "..\\model\\svmResult_" + savePath + ".xml";
			ShowTime();
			cout << "Train " << i << endl;
			svm->save(savePath);
			i++;
		}
		
	}
	
	Mat Oimage = imread(rPath+"02\\nm0000002_rm289065984_1924-9-16_1974.jpg",0);

	Mat sampleMat;
	resize(Oimage, sampleMat, Size(dim, dim));
	Mat result, result59;
	result.create(Size(dim, dim), image.type());
	result59.create(Size(dim, dim), image.type());

	uniformLBP(sampleMat, result);
	convert59(result, result59, table);
	float **testArray = new float*[1];
	testArray[0] = new float[dimdim];
	for (int y = 0, j = 0; y < result59.rows; y++)
	{
		for (int x = 0; x < result59.cols; x++, j++)
		{
			testArray[0][j] = (float)result59.at<uchar>(y, x);
		}
	}
	Mat test(1, dimdim, CV_32FC1, testArray);
	//resize(result59, test, Size(dimdim, 1));
	int response =svm->predict(test);
	cout << "response:" << response << endl;
	int a;
	cin >> a;
	return 0;
}
