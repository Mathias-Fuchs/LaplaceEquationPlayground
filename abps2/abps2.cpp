#include "abps2.h"
#include <Eigen/Sparse>
#include <vector>
#include <iostream>
#define _USE_MATH_DEFINES
#include <math.h>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include <algorithm>

typedef Eigen::SparseMatrix<double> SpMat; // declares a column-major sparse matrix type of double
typedef Eigen::Triplet<double> T;
#define INF 10000 

struct Point
{
	int x;
	int y;
	Point(int x, int y) :x(x), y(y) {}
};

void buildProblem(std::vector<T>& coefficients, Eigen::VectorXd& b, int n, const Eigen::VectorXd& boundary);
static inline void insertCoefficient(int id, int i, int j, double w, std::vector<T>& coeffs, Eigen::VectorXd& b, const Eigen::VectorXd& boundary);
void saveAsBitmap(const Eigen::VectorXd& x, int n, const char* filename);
static inline bool isInside(const std::vector<Point>& polygon, const Point& p);
const std::vector<double> complicatedPolygon = 
{
0.33333, 0.333333,0,
0.666666, 0.333333, 0,
0.5, 0.6666666, 0
};

int main()
{
	constexpr int n = 600;  // size of the image
	constexpr int m = n * n;  // number of unknows (=number of pixels)

					// Assembly:
	std::vector<T> coefficients;            // list of non-zeros coefficients
	Eigen::VectorXd b(m);                   // the right hand side-vector resulting from the constraints
	Eigen::VectorXd boundary(n); boundary.setZero(); //  = Eigen::ArrayXd::LinSpaced(n, 0, 2 * M_PI).sin().pow(2);
	std::vector<Point> polygon;
	for (auto d = complicatedPolygon.begin(); d != complicatedPolygon.end();) {
		double x = *d++;
		double y = *d++;
		double z = *d++;
		polygon.push_back(Point((int)(n * x), (int)(n * y)));
	}

	buildProblem(coefficients, b, n, boundary);
	SpMat A(m, m);
	A.setFromTriplets(coefficients.begin(), coefficients.end());

	// uncomment this for true Laplace solver proper
	 
	for (int j = 0; j < n; ++j)
	{
		for (int i = 0; i < n; ++i)
		{
			if (isInside(polygon, Point(i, j))) {
				b(i + j * n) = 1.0;
			}

		}
	}
	
	
	SpMat B(m, m);
	B.setIdentity();
	Eigen::VectorXd c(m); 
	for (int j = 0; j < n; ++j)
	{
		for (int i = 0; i < n; ++i)
		{
			if (isInside(polygon, Point(i, j))) {
				c(i + j * n) = 1.0;
				(B.diagonal())(i + j * n) = 1.0;
			}
			else {
				(B.diagonal())(i + j * n) = 0;
			}
		}
	}
	
	auto C = A * A + B * A + A * B + B;
	auto d = A * b + c;
	
	// Solving:
	Eigen::SimplicialCholesky<SpMat> chol(C);  // performs a Cholesky factorization of A
	Eigen::VectorXd x = chol.solve(d);         // use the factorization to solve for the given right hand side
	// Export the result to a file:
	saveAsBitmap(x, n, "x.bmp");
	return 0;
}

void buildProblem(std::vector<T>& coefficients, Eigen::VectorXd& b, int n, const Eigen::VectorXd& boundary)
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


static inline void insertCoefficient(int id, int i, int j, double w, std::vector<T>& coeffs,
	Eigen::VectorXd& b, const Eigen::VectorXd& boundary)
{
	int n = int(boundary.size());
	int id1 = i + j * n;
	if (i == -1 || i == n) b(id) -= w * boundary(j); // constrained coefficient
	else  if (j == -1 || j == n) b(id) -= w * boundary(i); // constrained coefficient
	else  coeffs.push_back(T(id, id1, w));              // unknown coefficient

}

void saveAsBitmap(const Eigen::VectorXd& x, int n, const char* filename)
{
	double min = x.minCoeff();
	double max = x.maxCoeff();
	std::vector<unsigned char> data(3 * n * n);
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			double value = (x(i * n + j) - min) / (max - min);
			double r = (double)(((int)(13 * value)) & 1);
			double g = abs(cos(value)) / 20;
			double b = abs(cos(value) * sin(value));
			data[3 * (i * n + j)] = (unsigned int)(r * 254);
			data[3 * (i * n + j) + 1] = (unsigned int)(g * 254);
			data[3 * (i * n + j) + 2] = (unsigned int)(b * 254);
		}
	}
	stbi_write_bmp(filename, n, n, 3, data.data());
}

// from  https://www.geeksforgeeks.org/check-if-two-given-line-segments-intersect/ 

static inline bool onSegment(const Point& p, const Point& q, const Point& r)
{
	if (q.x <= std::max(p.x, r.x) && q.x >= std::min(p.x, r.x) &&
		q.y <= std::max(p.y, r.y) && q.y >= std::min(p.y, r.y))
		return true;
	return false;
}

static inline int orientation(const Point& p, const Point& q, const Point& r)
{
	int val = (q.y - p.y) * (r.x - q.x) -
		(q.x - p.x) * (r.y - q.y);

	if (val == 0) return 0;  // colinear 
	return (val > 0) ? 1 : 2; // clock or counterclock wise 
}

static inline bool doIntersect(const Point& p1, const Point& q1, const Point& p2, const Point& q2)
{
	// Find the four orientations needed for general and 
	// special cases 
	int o1 = orientation(p1, q1, p2);
	int o2 = orientation(p1, q1, q2);
	int o3 = orientation(p2, q2, p1);
	int o4 = orientation(p2, q2, q1);

	// General case 
	if (o1 != o2 && o3 != o4)
		return true;

	// Special Cases 
	// p1, q1 and p2 are colinear and p2 lies on segment p1q1 
	if (o1 == 0 && onSegment(p1, p2, q1)) return true;

	// p1, q1 and p2 are colinear and q2 lies on segment p1q1 
	if (o2 == 0 && onSegment(p1, q2, q1)) return true;

	// p2, q2 and p1 are colinear and p1 lies on segment p2q2 
	if (o3 == 0 && onSegment(p2, p1, q2)) return true;

	// p2, q2 and q1 are colinear and q1 lies on segment p2q2 
	if (o4 == 0 && onSegment(p2, q1, q2)) return true;

	return false; // Doesn't fall in any of the above cases 
}

static inline bool isInside(const std::vector<Point>& polygon, const Point& p)
{
	int n = polygon.size();
	// There must be at least 3 vertices in polygon[] 
	if (n < 3)  return false;

	// Create a point for line segment from p to infinite 
	Point extreme = { INF, p.y };

	// Count intersections of the above line with sides of polygon 
	int count = 0, i = 0;
	do
	{
		int next = (i + 1) % n;

		// Check if the line segment from 'p' to 'extreme' intersects 
		// with the line segment from 'polygon[i]' to 'polygon[next]' 
		if (doIntersect(polygon[i], polygon[next], p, extreme))
		{
			// If the point 'p' is colinear with line segment 'i-next', 
			// then check if it lies on segment. If it lies, return true, 
			// otherwise false 
			if (orientation(polygon[i], p, polygon[next]) == 0)
				return onSegment(polygon[i], p, polygon[next]);

			count++;
		}
		i = next;
	} while (i != 0);

	// Return true if count is odd, false otherwise 
	return count & 1;  // Same as (count%2 == 1) 
}
