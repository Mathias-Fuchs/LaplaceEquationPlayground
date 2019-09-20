#include "abps2.h"
#include <Eigen/Sparse>
#include <vector>
#include <iostream>
#define _USE_MATH_DEFINES
#include <math.h>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

typedef Eigen::SparseMatrix<double> SpMat; // declares a column-major sparse matrix type of double
typedef Eigen::Triplet<double> T;

void buildProblem(std::vector<T>& coefficients, Eigen::VectorXd& b, int n, const Eigen::VectorXd& boundary);
void saveAsBitmap(const Eigen::VectorXd& x, int n, const char* filename);

int main()
{
	constexpr int n = 10;  // size of the image
	constexpr int m = n * n;  // number of unknows (=number of pixels)
	
					// Assembly:
	std::vector<T> coefficients;            // list of non-zeros coefficients
	Eigen::VectorXd b(m);                   // the right hand side-vector resulting from the constraints
	Eigen::ArrayXd boundary = Eigen::ArrayXd::LinSpaced(n, 0, 3 * M_PI).sin().pow(2);
	buildProblem(coefficients, b, n, boundary);
	SpMat A(m, m);
	A.setFromTriplets(coefficients.begin(), coefficients.end());
	// Solving:
	Eigen::SimplicialCholesky<SpMat> chol(A);  // performs a Cholesky factorization of A
	Eigen::VectorXd x = chol.solve(b);         // use the factorization to solve for the given right hand side
	// Export the result to a file:
	saveAsBitmap(x, n, "x.bmp");
	return 0;
}

void insertCoefficient(int id, int i, int j, double w, std::vector<T>& coeffs,
	Eigen::VectorXd& b, const Eigen::VectorXd& boundary)
{
	int n = int(boundary.size());
	int id1 = i + j * n;
	if (i == -1 || i == n) b(id) -= w * boundary(j); // constrained coefficient
	else  if (j == -1 || j == n) b(id) -= w * boundary(i); // constrained coefficient
	else  coeffs.push_back(T(id, id1, w));              // unknown coefficient
}

void buildProblem(std::vector<T>& coefficients, Eigen::VectorXd& b, int n, const Eigen::ArrayXd& boundary)
{
	b.setZero();
	for (int j = 0; j < n; ++j)
	{
		for (int i = 0; i < n; ++i)
		{
			int id = i + j * n;
			insertCoefficient(id, i - 1, j, -1, coefficients, b, boundary);
			insertCoefficient(id, i + 1, j, -1, coefficients, b, boundary);
			insertCoefficient(id, i, j - 1, -1, coefficients, b, boundary);
			insertCoefficient(id, i, j + 1, -1, coefficients, b, boundary);
			insertCoefficient(id, i, j, 4, coefficients, b, boundary);
		}
	}
}

void saveAsBitmap(const Eigen::VectorXd& x, int n, const char* filename)
{
	double min = x.minCoeff();
	double max = x.maxCoeff();
	auto data = std::vector<unsigned char>(4 * n * n);
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			double value = (x(i * n + j) - min) / (max-min);
			double r = abs(sin(value));
			double g = abs(cos(value)) / 20;
			double b = abs(cos(value) * sin(value));
			data[3 * (i * n + j)] = (unsigned int)(r * 254);
			data[3 * (i * n + j) + 1] = (unsigned int)(g * 254);
			data[3 * (i * n + j) + 2] = (unsigned int)(b * 254);
		}
	}
	stbi_write_bmp(filename, n, n, 3, data.data());
}