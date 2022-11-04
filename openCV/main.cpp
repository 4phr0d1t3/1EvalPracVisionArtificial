#define _USE_MATH_DEFINES // para el valor de pi que se usa en la generacion del kernel
#include <cmath>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>

using namespace std;
using namespace cv;

// Validacion para que el tamano del kernel sea par
int valN() {
	int n = 0;
	cout << "Introduce solo numeros impares" << std::endl;

	while (true) {
		cout << "Tamano del kernel: ";
		cin >> n;
		if (n % 2 == 0) cout << "No puedes escoger un numero par, intenta con un impar" << endl;
		else return n;
	}
	return 0;
}

// Creacion del kernel gaussiano
Mat createGaussKernel(int n, double sigma) {
	double r, s = 2.0 * sigma * sigma;
	Mat kernel(n, n, CV_64FC1);

	double sum = 0.0;
	int offset = int(n / 2);
	for (int x = offset * -1; x <= offset; ++x)
		for (int y = offset * -1; y <= offset; ++y) {
			r = sqrt(x*x + y*y);
			kernel.at<double>(x + offset, y + offset) = (exp(-(r*r) / s)) / (M_PI * s);
			sum += kernel.at<double>(x + offset, y + offset);
		}

	// Normalizamos el Kernel
	cout << "Kernel Gaussiano:" << endl;
	for (int i = 0; i < n; ++i) {
		for (int j = 0; j < n; ++j) {
			kernel.at<double>(i, j) /= sum;
			cout << '\t' << kernel.at<double>(i, j);
		}
		cout << endl;
	}

	return kernel;
}

// Creacion del relleno de los bordes
Mat padding(Mat img, int width, int height) {
	Mat tmp;
	img.convertTo(tmp, CV_64FC1);
	int padRows, padCols;

	padRows = (height - 1) / 2;
	padCols = (width - 1) / 2;

	Mat padded_image(Size(tmp.cols + 2*padCols, tmp.rows + 2*padRows), CV_64FC1, Scalar(0));
	img.copyTo(padded_image(Rect(padCols, padRows, img.cols, img.rows)));

	return padded_image;
}

// Aplicacion del kernel gaussiano a la imagen
void gauss(Mat& image, Mat& imgSmooth, int n, double sigma) {
	Mat kernel;

	kernel = createGaussKernel(n, sigma);
	imgSmooth = padding(imgSmooth, n, n);

	cout << "\nImagen Suavizada:" << endl << "\tfilas: " << imgSmooth.rows << "\tcolumnas : " << imgSmooth.cols << endl;

	Mat temp = Mat::zeros(image.size(), CV_64FC1);

	for (int i = 0; i < image.rows; i++)
		for (int j = 0; j < image.cols; j++)
			temp.at<double>(i, j) = sum(kernel.mul(imgSmooth(Rect(j, i, n, n)))).val[0];

	temp.convertTo(imgSmooth, CV_8UC1);
	
	// Mostrado e imprenta del tamano de la imagen suavizada
	namedWindow("Imagen Suavizada", WINDOW_AUTOSIZE);
	imshow("Imagen Suavizada", imgSmooth);
	cout << "\nImagen Suavizada:" << endl << "\tfilas: " << imgSmooth.rows << "\tcolumnas : " << imgSmooth.cols << endl;
}

// Iniciacion de la suavizacion
void promptGaussian(Mat image, Mat imgGrey) {
	double sigma = 0.0;
	int n = valN();

	// Obtencion del sigma que se usara para el gausseano
	cout << "\nSigma: ";
	cin >> sigma;

	// funcion del gaussiano
	gauss(image, imgGrey, n, sigma);
}

int main(int argc, const char* argv[]) {
	/*--------------------Obtencion de la imagen lena.png y sus atributos--------------------*/
	char NombreImagen[] = "lena.png"; // path
	Mat image;
	image = imread(NombreImagen);

	if (!image.data) { // por si no se encuentra la imagen
		cout << "Error al cargar la imagen: " << NombreImagen << endl;
		exit(1);
	}
	// Mostrado e imprenta del tamano de la imagen original
	namedWindow("Original", WINDOW_AUTOSIZE);
	imshow("Original", image);
	cout << "Imagen original:" << endl << "\tfilas: " << image.rows << "\tcolumnas : " << image.cols << endl;
	/*--------------------------Fin de la imagen lena.png y sus atributos--------------------*/


	/*--------------------Obtencion de la escala de grises por promedio--------------------*/
	Mat channels[3];
	cv::split(image, channels);

	vector<Mat> tempNTSC;
	tempNTSC.push_back((0.299) * (channels[0]) + (0.587) * (channels[1]) + (0.587) * (channels[2]));

	Mat imgNTSC;
	cv::merge(tempNTSC, imgNTSC);
	// Mostrado e imprenta del tamano de la imagen en escala de grises
	namedWindow("Escala de grises", WINDOW_AUTOSIZE);
	imshow("Escala de grises", imgNTSC);
	cout << "Imagen escala de grises:" << endl << "\tfilas: " << imgNTSC.rows << "\tcolumnas : " << imgNTSC.cols << endl;
	/*--------------------------Fin de la escala de grises por promedio--------------------*/


	// Llamada a la funcion para el proceso de suavizar la imagen
	promptGaussian(image, imgNTSC);

	// Imagen ecuializada
	Mat imgEqu;
	cv::equalizeHist(imgNTSC, imgEqu);
	// Mostrado e imprenta del tamano de la imagen ecualizada
	namedWindow("Imagen ecualizada", WINDOW_AUTOSIZE);
	imshow("Imagen ecualizada", imgEqu);
	cout << "Imagen ecualizada:" << endl << "\tfilas: " << imgEqu.rows << "\tcolumnas : " << imgEqu.cols << endl;

	waitKey(0);
	return 0;
}
