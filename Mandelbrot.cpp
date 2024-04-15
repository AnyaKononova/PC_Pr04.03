#include <iostream>
#include <opencv2/opencv.hpp>
#include <mpi.h>

using namespace std;

const int IMAGE_WIDTH = 700;
const int IMAGE_HEIGHT = 700;
const int MAX_ITERATIONS = 1000;

int computeMandelbrot(double cr, double ci) {
    double zr = 0, zi = 0;
    int iterations = 0;
    while (zr * zr + zi * zi < 4 && iterations < MAX_ITERATIONS) {
        double zr_new = zr * zr - zi * zi + cr;
        double zi_new = 2 * zr * zi + ci;
        zr = zr_new;
        zi = zi_new;
        iterations++;
    }
    return iterations;
}

cv::Vec3b getColor(int iterations) {
    if (iterations >= 255) {
        return cv::Vec3b(0, 0, 0);
    }
    else {
        return cv::Vec3b((iterations % 8 + 1) / 2.0 * 255, (iterations % 27 + 1) / 3.0 * 255, (iterations % 97 + 1) / 5.5 * 255);
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, numProcesses;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numProcesses);

    int rowsPerProcess = IMAGE_HEIGHT / numProcesses;            
    int startRow = rank * rowsPerProcess;
    int endRow = (rank + 1) * rowsPerProcess;

    cv::Mat fractalImage(IMAGE_HEIGHT, IMAGE_WIDTH, CV_8UC3);

    for (int y = startRow; y < endRow; y++) {
        for (int x = 0; x < IMAGE_WIDTH; x++) {
            double cr = -2.0 + (1.0 + 2.0) * x / (IMAGE_WIDTH - 1);
            double ci = -1.5 + (1.5 + 1.5) * y / (IMAGE_HEIGHT - 1);

            int iterations = computeMandelbrot(cr, ci);
            fractalImage.at<cv::Vec3b>(y, x) = getColor(iterations);
        }
    }

    cv::Mat resultImage;
    if (rank == 0) {
        resultImage = cv::Mat(IMAGE_HEIGHT, IMAGE_WIDTH, CV_8UC3);
    }
    MPI_Gather(fractalImage.data, IMAGE_WIDTH * rowsPerProcess * 3, MPI_BYTE,
        resultImage.data, IMAGE_WIDTH * rowsPerProcess * 3, MPI_BYTE, 0, MPI_COMM_WORLD);

    MPI_Finalize();

    if (rank == 0) {
        cv::imshow("Finish", resultImage);
        cv::imwrite("finish4.jpg", resultImage);
        cv::waitKey(0);
    }

    return 0;
}

