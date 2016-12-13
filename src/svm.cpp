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
#define EachTrainNum 10000
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



int train()
{
	int endInput;
	uchar table[256];
	lbp59table(table);
	stringstream tempPath;
	lbpData = new uchar[5900];
	Mat image;
	string savePath = "..\\model\\svmResult_1200.xml";
	Ptr<SVM> svm = StatModel::load<SVM>(savePath);
	/*
	Ptr<SVM> svm = SVM::create();
	*/
	svm->setType(SVM::C_SVC);
	svm->setKernel(SVM::LINEAR);
	svm->setTermCriteria(TermCriteria(CV_TERMCRIT_ITER, 10000, 1e-12));
	

	int picNum = 10000;
	int testNum=200;
	string rPath = "E:\\AdienceBenchmarkOfUnfilteredFacesForGenderAndAgeClassification\\feature\\";
	string rTestPath = "E:\\AdienceBenchmarkOfUnfilteredFacesForGenderAndAgeClassification\\feature\\";
	int dim = 64;
	int dimdim = dim*dim;
	
	ifstream info("E:\\AdienceBenchmarkOfUnfilteredFacesForGenderAndAgeClassification\\gender.txt");
	ofstream accurateFile("..\\model\\accurate.txt");
	if (!info.is_open())
	{
		cout << "can't open the info file!" << endl;
		cin >> endInput;
		return 0;
	}
	//string eof;
	string *path = new string[picNum];
	int *gender = new int[picNum];
	cout << "reading info..." << endl;
	for (int i = 0; i < picNum; i++)
	{
		
		info >> path[i];
		if (0 == path[i].length())
		{
			cout << path[i-1] << endl;
			cout << "reading info " << i << endl;
		}
		
		info >> gender[i];
		//cout<< gender[i] << endl;
		/*
		info >> eof;
		eof = info.peek();
		while ("\n" != eof)
		{
			info >> eof;
			eof = info.peek();
		}
		*/
		//cout << "reading info "<<i << endl;
	}
	ifstream testInfo("E:\\AdienceBenchmarkOfUnfilteredFacesForGenderAndAgeClassification\\test.txt");
	if (!testInfo.is_open())
	{
		cout << "can't open the info file!" << endl;
		cin >> endInput;
		return 0;
	}
	
	string *testPath = new string[testNum];
	int *testGender = new int[testNum];
	cout << "reading info..." << endl;
	for (int i = 0; i < testNum; i++)
	{
		
		testInfo >> testPath[i];
		testInfo >> testGender[i];
		if (0 == testPath[i].length())
		{
			cout << "error test " << testPath[i-1] << endl;;
			cout << "reading info " << i << endl;
		}
		/*
		testInfo >> eof;
		eof = testInfo.peek();
		while ("\n" != eof)
		{
			testInfo >> eof;
			eof = testInfo.peek();
		}
		*/
		//cout << "reading info "<<i << endl;
	}
	cout << "test info read!" << endl;
	
	/*int *labels = new int[EachTrainNum];
	float **trainingData = new float*[EachTrainNum];
	for (int i = 0; i < EachTrainNum; i++)
	{
		trainingData[i] = new float[dimdim];
	}*/
	float trainingData[EachTrainNum][4096];
	int labels [EachTrainNum];
	srand((unsigned)time(NULL));
	int  train_count= 1200;
	//cout << path[30985] << endl;
	while (true)
	{
		//cout << "select data" << endl;
		for (int pic_count = 0; pic_count < EachTrainNum; pic_count++)
		{
			int randNum = rand() % picNum;
			string imgPath =  path[randNum];
			//cout << path[1000] << endl;
			Mat Oimage = imread(rPath+imgPath, 0);
			while (Oimage.empty())
			{
				randNum = rand() % picNum;
				imgPath = path[randNum];
				//cout << path[1000] << endl;
				Oimage = imread(rPath + imgPath, 0);
			}
			resize(Oimage, image, Size(dim, dim));

			//Mat image;
			if (!image.data)
			{
				cout << "Fail to load image" << endl;
				return 0;
			}
			//resize(OM, image, Size(10, 10));
			Mat result, result59;
			result.create(Size(dim, dim), image.type());
			result59.create(Size(dim, dim), image.type());

			uniformLBP(image, result);
			convert59(result, result59, table);


			labels[pic_count] = gender[randNum];

			for (int y = 0, j = 0; y < result59.rows; y++)
			{
				for (int x = 0; x < result59.cols; x++, j++)
				{
					trainingData[pic_count][j] = (float)result59.at<uchar>(y, x);
				}
			}
		}
		Mat labelsMat(EachTrainNum, 1, CV_32SC1, labels);
		Mat trainingDataMat(EachTrainNum, dimdim, CV_32FC1, trainingData);
		stringstream saveNameStream;

		string savePath;
		saveNameStream << train_count;
		saveNameStream >> savePath;
		saveNameStream.clear();
		//cout << "trainning..." << endl;
		if (svm->train(trainingDataMat, ROW_SAMPLE, labelsMat) && 0 == (train_count) % 20)
		{
			savePath = "..\\model\\svmResult_" + savePath + ".xml";
			ShowTime();
			cout << "saving " << train_count << endl;
			svm->save(savePath);

			//test
			{
				int rightNum = 0;
				int realTestNum = testNum;
				for (int i = 0; i < testNum; i++)
				{
					Mat Oimage = imread(rTestPath + testPath[i], 0);
					while (Oimage.empty())
					{
						//cout << path[1000] << endl;
						i++;
						realTestNum--;
						Oimage = imread(rTestPath + testPath[i], 0);
					}

					//cout << "test " << i << endl;
					Mat sampleMat;
					resize(Oimage, sampleMat, Size(dim, dim));
					Mat result, result59;
					result.create(Size(dim, dim), image.type());
					result59.create(Size(dim, dim), image.type());
					uniformLBP(sampleMat, result);
					convert59(result, result59, table);
					float testArray[1][4096];
					for (int y = 0, j = 0; y < result59.rows; y++)
					{
						for (int x = 0; x < result59.cols; x++, j++)
						{
							testArray[0][j] = (float)result59.at<uchar>(y, x);
						}
					}
					Mat test(1, dimdim, CV_32FC1, testArray);
					//resize(result59, test, Size(dimdim, 1));

					float response = svm->predict(test);
					/*
					if (0==response)
					{
						putText(Oimage, "nv",Point(50,50), CV_FONT_HERSHEY_COMPLEX,2,Scalar(0,0,255));
					}
					else if(1 == response)
					{
						putText(Oimage, "nan", Point(50, 50), CV_FONT_HERSHEY_COMPLEX, 2, Scalar(0, 0, 255));
					}
					if (0 == testGender[i])
					{
						putText(Oimage, "nv", Point(104, 124), CV_FONT_HERSHEY_COMPLEX, 2, Scalar(0, 0, 255));
					}
					else if (1 == testGender[i])
					{
						putText(Oimage, "nan", Point(104, 124), CV_FONT_HERSHEY_COMPLEX, 2, Scalar(0, 0, 255));
					}
					CheckImage("Oimage", Oimage);
					*/
					//cout << testPath[i]<<" "<<testGender[i] << " "<<response<<endl;
					//CheckImage("Oimage", Oimage);
					if (response == testGender[i])
					{
						rightNum++;
					}
				}
				cout << "accurate:" << 1.0*rightNum * 100 / realTestNum << "%" << endl;
				accurateFile << train_count << ": " << 1.0*rightNum * 100 / realTestNum << "%" << endl;
			}
			//test train
			{
				int rightNum = 0;
				int realTestNum = testNum;
				for (int i = 0; i < testNum; i++)
				{
					Mat Oimage = imread(rPath + path[i], 0);
					while (Oimage.empty())
					{
						//cout << path[1000] << endl;
						i++;
						realTestNum--;
						Oimage = imread(rPath + path[i], 0);
					}

					//cout << "test " << i << endl;
					Mat sampleMat;
					resize(Oimage, sampleMat, Size(dim, dim));
					Mat result, result59;
					result.create(Size(dim, dim), image.type());
					result59.create(Size(dim, dim), image.type());
					uniformLBP(sampleMat, result);
					convert59(result, result59, table);
					float testArray[1][4096];
					for (int y = 0, j = 0; y < result59.rows; y++)
					{
						for (int x = 0; x < result59.cols; x++, j++)
						{
							testArray[0][j] = (float)result59.at<uchar>(y, x);
						}
					}
					Mat test(1, dimdim, CV_32FC1, testArray);
					//resize(result59, test, Size(dimdim, 1));

					float response = svm->predict(test);
					if (response == gender[i])
					{
						rightNum++;
					}
				}
				cout << "accurate:" << 1.0*rightNum * 100 / realTestNum << "%" << endl;
			}
		}
		train_count++;
	}
	
	
	
	int a;
	cin >> a;
	return 0;
}
int test()
{
	
	string savePath = "..\\model\\svmResult_70000.xml";
	Ptr<SVM> svm = StatModel::load<SVM>(savePath);
	uchar table[256];
	lbp59table(table);
	uchar key= 1;
	while (key)
	{
		string name;
		cout << "print name:" << endl;
		cin >> name;
		Mat Oimage = imread("E:\\" + name,0);
		while (Oimage.empty())
		{
			cout << "error print name:" << endl;
			cin >> name;
			Oimage = imread("E:\\" + name, 0);
		}
		int dim = 64;
		int dimdim = dim*dim;
		Mat sampleMat;
		resize(Oimage, sampleMat, Size(dim, dim));
		Mat result, result59;
		result.create(Size(dim, dim), Oimage.type());
		result59.create(Size(dim, dim), Oimage.type());
		uniformLBP(sampleMat, result);
		convert59(result, result59, table);
		float testArray[1][4096];
		for (int y = 0, j = 0; y < result59.rows; y++)
		{
			for (int x = 0; x < result59.cols; x++, j++)
			{
				testArray[0][j] = (float)result59.at<uchar>(y, x);
			}
		}
		Mat test(1, dimdim, CV_32FC1, testArray);
		//resize(result59, test, Size(dimdim, 1));

		float response = svm->predict(test);
		cout << response << endl;
		imshow("Oimage",Oimage);
		key = waitKey(0);
	}
	return 0;
}
int main()
{
	train();
	return 0;
}