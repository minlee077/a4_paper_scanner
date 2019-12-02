#include <opencv2/opencv.hpp>

#include <iostream>
#include <cmath>
#include <vector>
#include <string>

#define RGBPIXEL2GREYSCALE(B,G,R) 0.3*R + 0.59*G + 0.11*B
#define BINARIZATION(P) ((P>150)?255:0)


#define SCALE 2
#define TARGET_WIDTH SCALE * 224
#define TARGET_HEIGHT SCALE * 297

using namespace cv;
using namespace std;

const int dx[9] = { -1,0,1,1,1,0,-1,-1,0 };
const int dy[9] = { 1,1,1,0,-1,-1,-1,0,0 };

int contourTrial =0;

Mat convertRGB2Greyscale(Mat image);
Mat GaussianFiltering(Mat image);
Mat convertBinaryImage(Mat image);

void errorRoutine(Mat binaryImage, Mat image);

void setContourStartPoint(Point& p, Mat image);
vector<Point> searchContour(Mat image);
Mat perspectiveTransform(Mat image, Mat transformMatrix, Size size);
Mat getPerspectiveMatrix(vector<Point2f> src, vector<Point2f> dst);

Mat plotContours(vector<Point> ps, Mat image);
Mat plotVerties(vector<Point> ps, Mat image);

vector<Point> rdp(vector <Point>v, int epsilon);
double findPerpendicularDistance(Point p, Point p1, Point p2);
vector<Point> findVertex(vector<Point> v);

vector<Point> linePointDist(vector<Point>line, vector<Point> v);
vector<Point> lastPointExtraction(vector<Point>vertex, vector<Point>approx);

vector<Point> vertexAlign(vector<Point> vertex);

Mat matrix_inv(Mat m);



int main()
{
	Mat image, greyScaleImage, binaryImage,transformMatrix,gauimage,csimage;
	vector<Point> contours, approx, vertex;

	vector<Point2f> fvertex,a4Standard;

	int width, height;

	
	String imageName = "(3)";
	String imagePath = imageName+".jpg";
	image = imread(imagePath);   // Read the file
	if (!image.data)                              // Check for invalid input	
	{
		std::cout << "Could not open or find the image" << std::endl;
		return -1;
	}

	height = image.size().height;
	width = image.size().width;

	
	cout << "height:"<<height<<"\nwidth:" <<width<< endl;

	greyScaleImage = convertRGB2Greyscale(image);
	gauimage=GaussianFiltering(greyScaleImage);

	binaryImage = convertBinaryImage(gauimage); 
	contours = searchContour(binaryImage); // 컨투어 탐색	
	//image = plotContours(contours, image);

	//cv::namedWindow("Binalized Image", WINDOW_AUTOSIZE);// Create a window for display.
	//cv::imshow("Binalized Image", image);                   // Show our image inside it.

	//waitKey(0);
	if (contours.empty())
	{
		errorRoutine(binaryImage,image);
		return -1;
	}
	cout << "contour detected" << endl;
	approx = rdp(contours, 0.01 * contours.size()); //Ramer–Douglas–Peucker_algorithm for simplifying the contours ( extract corner points )
	if (approx.size() < 4)
	{
		errorRoutine(binaryImage, image);
		return -2;
	}
	//image = plotContours(approx, image);

	vertex = findVertex(approx); // 컨투어의 특정 점과 가장 먼 다른 점을 찾기 ( 대각점 )
	vertex = linePointDist(vertex, approx); // 점과 직선사이 거리를 이용하여 직선과 가장 먼 한점을 찾아냄 (나머지 한 꼭짓점)
	vertex = lastPointExtraction(vertex, approx);
	
	//image =plotVerties(vertex, image);
	vertex = vertexAlign(vertex); // 시계방향 정렬
	cout << "Cornner detected" << endl;

	a4Standard.push_back(Point2f(0, 0));
	a4Standard.push_back(Point2f(TARGET_WIDTH, 0));
	a4Standard.push_back(Point2f(TARGET_WIDTH, TARGET_HEIGHT));
	a4Standard.push_back(Point2f(0, TARGET_HEIGHT));//
	fvertex.push_back(vertex[0]);
	fvertex.push_back(vertex[1]);
	fvertex.push_back(vertex[2]);
	fvertex.push_back(vertex[3]);

	transformMatrix=getPerspectiveMatrix(fvertex, a4Standard); 
	image = perspectiveTransform(image, transformMatrix, Size(TARGET_WIDTH, TARGET_HEIGHT));

	cout<<"----------------------------------------------"<<endl;
	cout << "Aligned corner points : " << vertex[0] << ' ' << ' ' << vertex[1] << ' ' << vertex[2] << ' ' << vertex[3] << endl;
	cout << "----------------------------------------------" << endl;
	cv::namedWindow("Binalized Image", WINDOW_AUTOSIZE);// Create a window for display.
	cv::imshow("Binalized Image", binaryImage);                   // Show our image inside it.
	cv::namedWindow("Detected Document", WINDOW_AUTOSIZE);// Create a window for display.
	cv::imshow("Detected Document", image);                   // Show our image inside it.

	imwrite("scanned_images/"+imageName + "_Scanned.jpg", image);
	cv::waitKey(0);                                          // Wait for a keystroke in the window

	return 0;
}

Mat convertRGB2Greyscale(Mat image)
{
	Mat dst = Mat::zeros(image.size(), CV_8UC1);

	int height = image.size().height;
	int width = image.size().width;

	for (int y = 0; y < height; y++)
		for (int x = 0; x < width; x++)
			dst.at<uchar>(y, x) = RGBPIXEL2GREYSCALE(image.at<Vec3b>(y, x).val[0], image.at<Vec3b>(y, x).val[1], image.at<Vec3b>(y, x).val[2]);
	return dst;
}


Mat convertBinaryImage(Mat image)
{
	Mat dst = Mat::zeros(image.size(), CV_8UC1);

	int height = image.size().height;
	int width = image.size().width;

	for (int y = 0; y < height; y++)
		for (int x = 0; x < width; x++)
		{
			dst.at<uchar>(y, x) = BINARIZATION(image.at<uchar>(y, x));

		}

	return dst;
}

/*
Algorithm : for detecting contour 150~210

## square tracing algorithm ##
Input: A square tessellation, T, containing a connected component P of black cells.
Output: A sequence B (b1, b2 ,..., bk) of boundary pixels i.e. the contour.

Begin

Set B to be empty.
From bottom to top and left to right scan the cells of T until a black pixel, s, of P is found.
Insert s in B.
Set the current pixel, p, to be the starting pixel, s.
Turn left i.e. visit the left adjacent pixel of p.
Update p i.e. set it to be the current pixel.

While p not equal to s do
   If the current pixel p is black
		insert p in B and turn left (visit the left adjacent pixel of p).
		Update p i.e. set it to be the current pixel.

   else
		turn right (visit the right adjacent pixel of p).
		Update p i.e. set it to be the current pixel.
end While

End

*/

void setContourStartPoint(Point& p, Mat image)
{
	int height = image.size().height;
	int width = image.size().width;

	for (int row = 0; row < height; row++)
		for (int col = 0; col < width; col++)
		{
			if (image.at<uchar>(row, col) == 255)
			{
				p.x = col;
				p.y = row;
				return;
			}
		}
}

Point GoLeft(Point p) { return Point(p.y, -p.x); }
Point GoRight(Point p) { return Point(-p.y, p.x); }
vector<Point> searchContour(Mat image)
{
	vector<Point> contourPoints;
	vector<Point> emptyPoints;
	// contour image (http://www.imageprocessingplace.com/downloads_V3/root_downloads/tutorials/contour_tracing_Abeer_George_Ghuneim/square.html)
	int height = image.size().height;
	int width = image.size().width;

	Point startPoint(-1, -1);
	setContourStartPoint(startPoint, image);

	if (startPoint.x == -1 && startPoint.y == -1)
	{
		//fail to detect any points	
		return contourPoints;
	}
	bool outPos=false;
	contourPoints.push_back(startPoint);
	Point nextStep = GoLeft(Point(1, 0));
	Point next = startPoint + nextStep;
	while (next != startPoint) {
		if (next.y <= 0 || next.y >= height|| next.x <= 0 || next.x >= width)
		{
			cout << "invalid position" << endl;
			outPos = true;
			break;

		}

		if (image.at<uchar>(next.y, next.x) == 0) {
			nextStep = GoRight(nextStep);
			next = next + nextStep;
		}
		else {
			contourPoints.push_back(next);
			nextStep = GoLeft(nextStep);
			next = next + nextStep;
		}
	}


	if (contourPoints.size() < 40 || outPos) // Perhaps 8-connected but not 4-connected cause problem http://www.imageprocessingplace.com/downloads_V3/root_downloads/tutorials/contour_tracing_Abeer_George_Ghuneim/8con.html
	{
		contourTrial++;
		for (Point p : contourPoints) {
			image.at<uchar>(p.y, p.x) = 0;
		}
		
		if (contourTrial > 30)
			return emptyPoints;
		contourPoints = searchContour(image);

		if (contourPoints.size() == 0)
		{
			return emptyPoints;
		}
	}

	return contourPoints;

}

Mat plotContours(vector<Point> ps, Mat image) {

	int height = image.size().height;
	int width = image.size().width;

	for (Point p : ps) {
		for (int i = 0; i < 9; i++)
		{
			image.at<Vec3b>(p.y + dy[i], p.x + dx[i]).val[0] = 0;
			image.at<Vec3b>(p.y + dy[i], p.x + dx[i]).val[1] = 255;
			image.at<Vec3b>(p.y + dy[i], p.x + dx[i]).val[2] = 0;
		}
	}
	return image;
}

double findPerpendicularDistance(Point p,Point p1, Point p2){
	double result;
	double slope;
	double intercept;
	if (p1.x == p2.x) {
		result = fabs(p.x - p1.x);
	}
	else {
		slope = (double)(p2.y - p1.y) / (double)(p2.x - p1.x);
		intercept = (double)p1.y - (slope * p1.x);
		result = fabs(slope * p.x - (double)p.y + intercept) / sqrt(pow(slope, 2) + 1.0);
	}
	return result;
}

vector<Point> findVertex (vector<Point> v)
{
	Point firstPoint = v[0];
	Point lastPoint = v[v.size() - 1];

	if (v.size() < 3) {
		return v;
	}
	int index = -1;
	double maxDist = 0;

	for (int i = 1; i < v.size() - 1; i++) {
		double cDist = findPerpendicularDistance(v[i], firstPoint, lastPoint);
		if (cDist > maxDist) {
			index = i;
			maxDist = cDist;
		}
	}

	vector<Point> ret;

	ret.push_back(firstPoint);
	ret.push_back(v[index]);

	return ret;
}


Mat plotVerties(vector<Point> ps, Mat image)
{

	int height = image.size().height;
	int width = image.size().width;

	for (Point p : ps) {
		for (int i = 0; i < 9; i++)
		{
			image.at<Vec3b>(p.y + dy[i], p.x + dx[i]).val[0] = 0;
			image.at<Vec3b>(p.y + dy[i], p.x + dx[i]).val[1] = 0;
			image.at<Vec3b>(p.y+dy[i], p.x+dx[i]).val[2] = 255;
		}
	}
	return image;

}

vector<Point> linePointDist(vector<Point>line, vector<Point> v)
{
	double a, b, c, maxD, currentD;
	maxD = 0;
	currentD = 0;
	if (line[0].x == line[1].x)
	{
		// ax+by+c=0
		// 1*x=k
		a = 1;
		b = 0;
		c = line[0].x;
	}
	else
	{
		//y-mx+line[0].y+mline[0].x=0
		double m = ((double)line[0].y - line[1].y) / ((double)line[0].x-line[1].x);
		a = -m;
		b = 1;
		c = line[0].x * m - line[0].y;
	}
	
	double denominator =sqrt( a*a+b*b);
	int idx = 0;
	int maxIdx = -1;
	for (Point cur : v)
	{
		currentD=fabs(a* cur.x + b * cur.y + c)/denominator;
		if (currentD > maxD)
		{
			maxD = currentD;
			maxIdx = idx;
		}
		idx++;
	}
	
	line.push_back(v[maxIdx]);
	return line;
}



vector<Point> rdp(vector <Point>v, int epsilon) {

	Point firstPoint = v[0];
	Point lastPoint = v[v.size() - 1];

	if (v.size() < 3) {
		return v;
	}
	int index = -1;
	double maxDist = 0;

	for (int i = 1; i < v.size() - 1; i++) {
		double cDist = findPerpendicularDistance(v[i], firstPoint, lastPoint);
		if (cDist > maxDist) {
			index = i;
			maxDist = cDist;
		}
	}
	if (maxDist > epsilon) {
		vector<Point> l1 = vector<Point>(v.begin(), v.begin() + index);
		vector<Point> l2 = vector<Point>(v.begin() + index, v.end());
		vector<Point> r1 = rdp(l1, epsilon);
		vector<Point> r2 = rdp(l2, epsilon);
		vector<Point> rs = r1;
		rs.insert(rs.end(), r2.begin(), r2.end());
		return rs;
	}
	else {
		vector<Point>a{ firstPoint, lastPoint };
		return a;
	}
	return v;
}

vector<Point> lastPointExtraction(vector<Point>vertex, vector<Point>approx)
{
	double maxArea=0;
	double area = 0;

	int x1, x2, x3, y1, y2, y3, x, y,cornerX,cornerY;
	x1 = vertex[0].x; x2 = vertex[1].x; x3 = vertex[2].x;
	y1 = vertex[0].y; y2 = vertex[1].y; y3 = vertex[2].y;

	for (Point p : approx)
	{
		x = p.x;
		y = p.y;
		area=abs((x1 * y2 + x2 * y + x * y1) - (x2 * y1 + x * y2 + x1 * y))
			+ abs((x1 * y + x * y3 + x3 * y1) - (x * y1 + x3 * y + x1 * y3))
			+ abs((x * y2 + x2 * y3 + x3 * y) - (x2 * y + x3 * y2 + x * y3));

		if (area > maxArea)
		{
			maxArea = area;
			cornerX = x;
			cornerY = y;

		}
	}
	vertex.push_back(Point(cornerX,cornerY));
	return vertex;
}

vector<Point> vertexAlign(vector<Point> vertex)
{
	//reverse clock-wise sort
	vector<Point>TopPoints;
	vector<Point>BottomPoints;
	vector<Point>ret(4);
	for (Point v : vertex)
	{
		int cnt = 0;
		for (int i = 0; i < 4; i++)
		{
			if (v == vertex[i])
				continue;
			if (v.y < vertex[i].y)//if vertex[i] is positioned below the v
				cnt++;

		}
		if (cnt >= 2)
			TopPoints.push_back(v);
		else
			BottomPoints.push_back(v);
	}
	if (TopPoints[0].x < TopPoints[1].x)
	{
		ret[0] = TopPoints[0];
		ret[1] = TopPoints[1];
	}
	else
	{
		ret[0] = TopPoints[1];
		ret[1] = TopPoints[0];
	}
	if (BottomPoints[0].x < BottomPoints[1].x)
	{
		ret[2] = BottomPoints[1];
		ret[3] = BottomPoints[0];
	}
	else
	{
		ret[2] = BottomPoints[0];
		ret[3] = BottomPoints[1];
	}


	return ret;
}

void errorRoutine(Mat binaryImage,Mat image)
{

	cout << "fail to detect contours " << endl;
	cv::namedWindow("1.jpg Image", WINDOW_AUTOSIZE);// Create a window for display.
	cv::imshow("1.jpg Image", binaryImage);                   // Show our image inside it.
	cv::namedWindow("2.jpg Image", WINDOW_AUTOSIZE);// Create a window for display.
	cv::imshow("2.jpg Image", image);                   // Show our image inside it.

	cv::waitKey(0);   // Wait for a keystroke in the window

	return ;
}

Mat GaussianFiltering(Mat image)
{
	//3x3, sigma =1.0

	double gaussianKernel[3][3] = { {0,1.0 / 8,0},
	{1.0 / 8,1.0 / 2,1.0 / 8},
	{0,1.0 / 8,0}
	};

	int height = image.size().height;
	int width = image.size().width;
	Mat outputImage = image;

	for (int x = 0; x < height; x++) {
		for (int y = 0; y < width; y++) {
			double summation = 0;
			if (x == 0 || x == height - 1 || y == 0 || y == width - 1)
				continue;
			else
			{
				for (int i = -1; i <= 1; i++)
				{
					for (int j = -1; j <= 1; j++)
					{
						outputImage.at<uchar>(x, y) += 1.0 / 9.0 * image.at<uchar>(x + i, y + j) * gaussianKernel[i + 1][j + 1];
					}
				}
			}
		}
	}
	return outputImage;
}

/*
calculate Perespective Transformation Matrix
*/

Mat getPerspectiveMatrix(vector<Point2f> src, vector<Point2f> dst)
{
	double a[8][8], b[8];
	Mat M(3, 3, CV_64F), X(8, 1, CV_64F, M.ptr());
	Mat A(8, 8, CV_64F, a),B(8, 1, CV_64F, b);
	for (int i = 0; i < 4; ++i)
	{
		a[i][0] = a[i + 4][3] = src[i].x;
		a[i][1] = a[i + 4][4] = src[i].y;
		a[i][2] = a[i + 4][5] = 1;
		a[i][3] = a[i][4] = a[i][5] = a[i + 4][0] = a[i + 4][1] = a[i + 4][2] = 0;
		a[i][6] = -src[i].x * dst[i].x;
		a[i][7] = -src[i].y * dst[i].x;
		a[i + 4][6] = -src[i].x * dst[i].y;
		a[i + 4][7] = -src[i].y * dst[i].y;
		b[i] = dst[i].x;
		b[i + 4] = dst[i].y;
	}
	Mat AInv = matrix_inv(A);
	Mat C = AInv  * B;
	for (int i = 0; i < 8; i++)
		M.ptr<double>()[i] = C.ptr<double>()[i];
	M.ptr<double>()[8] = 1.;
	return M;
}


/*
calc 4x4 inverse matrix
using Gaussian elimination
*/

Mat matrix_inv(Mat m) {

	double a[8][8], b[8][16];
	Mat inv(8, 8, CV_64F, Scalar(0));
	inv = 0;
	Mat n(8, 16, CV_64F, b);
	n = 0;
	int iter, i, j, k;

	double v;

	double tmp;
	int    max_key;
	iter = m.rows;

	for (j = 0; j < iter; j++)
		for (i = 0; i < iter; i++)
			b[j][i] = m.at<double>(j, i);

	for (i = 0; i < iter; i++)
		b[i][i + iter] = 1.0;


	for (i = 0; i < iter; i++) {
		max_key = i;
		for (j = i + 1; j < iter; j++)
			if (b[j][i] > b[max_key][i])
				max_key = j;
		if (max_key != i) {
			for (j = 0; j < iter * 2; j++) {
				tmp = b[i][j];
				b[i][j] = b[max_key][j];
				b[max_key][j] = tmp;
			}
		}
		v = b[i][i];
		for (j = i + 1; j < iter * 2; j++)
			b[i][j] /= v;

		for (j = i + 1; j < iter; j++) {
			v = b[j][i];
			b[j][i] = 0.0;
			for (k = i + 1; k < iter * 2; k++) {
				b[j][k] -= b[i][k] * v;
			}
		}
	}
	for (i = iter - 2; i >= 0; i--) {

		for (j = i; j >= 0; j--) {
			v = b[j][i + 1];
			for (k = 0; k < iter * 2; k++) {
				b[j][k] -= b[i + 1][k] * v;
			}
		}
	}
	for (j = 0; j < iter; j++)
		for (i = 0; i < iter; i++)
			inv.at<double>(j, i) = b[j][i + iter];
	return inv;
}


/*
Apply perspective transform
*/
Mat perspectiveTransform(Mat image, Mat transformMatrix, Size size)
{
	int height = image.size().height;
	int width = image.size().width;
	Mat outputImage(size,image.type());
	int xp;
	int yp;
	for (int y = 0; y < height; y++)
		for (int x = 0; x < width; x++) {
			xp = (int)abs(((transformMatrix.at<double>(0, 0) * x + transformMatrix.at<double>(0, 1) * y + transformMatrix.at<double>(0, 2)) 
				/ (transformMatrix.at<double>(2, 0) * x + transformMatrix.at<double>(2, 1) * y + 1))); 
			yp = (int)abs(((transformMatrix.at<double>(1, 0) * x + transformMatrix.at<double>(1, 1) * y + transformMatrix.at<double>(1, 2)) 
				/ (transformMatrix.at<double>(2, 0) * x + transformMatrix.at<double>(2, 1) * y + 1)));
			if (yp >= 0 && yp < TARGET_HEIGHT && xp >= 0 && xp < TARGET_WIDTH)
			{
				outputImage.at<Vec3b>(yp, xp).val[0] = image.at<Vec3b>(y, x).val[0];
				outputImage.at<Vec3b>(yp, xp).val[1]=image.at<Vec3b>(y, x).val[1];
				outputImage.at<Vec3b>(yp, xp).val[2]=image.at<Vec3b>(y, x).val[2] ;
			}
		}
	return outputImage;
}
