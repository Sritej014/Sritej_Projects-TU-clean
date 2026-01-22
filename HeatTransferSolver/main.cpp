#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <fstream>
#include <string>

#include <vtkSmartPointer.h>

#include <vtkXMLImageDataReader.h>
#include <vtkXMLImageDataWriter.h>
#include <vtkImageData.h>
#include <vtkDoubleArray.h>
#include <vtkPointData.h>
#include <vtkStructuredGrid.h>
#include <vtkPoints.h>
#include <vtkDoubleArray.h>
#include <vtkPointData.h>
#include <vtkXMLStructuredGridWriter.h>

#include <DenseBase.h>
#include <Matrix.h>
#include <VectorBlock.h>
#include "C:\Users\srite\Desktop\VCPKG\vcpkg\packages\eigen3_x64-windows\include\eigen3\Eigen\Dense"
#include "C:\Users\srite\Desktop\VCPKG\vcpkg\packages\eigen3_x64-windows\include\eigen3\Eigen\Sparse"
#include <Eigen/LU>
using namespace std;
using namespace Eigen;



const double PI = 3.14159265358979323846;

// Function declarations
struct HeatEquationParams{
    int N;
    int nsteps;
    int outputInterval;
    double alpha;
    double beta;
    double gamma;
    double dt;
    double dx;
};

void initializeTemperature(vector<vector<double>>& u, int N);
double calculateSourceTerm(double x, double y, double u, double t, double alpha, double beta, double gamma);
void applyDirichletBC(vector<vector<double>>& u);
MatrixXd createMatrixA(int n, double dt, double alpha);
void solveLinearSystem(VectorXd& u, const MatrixXd& A, int N, double dt, double alpha, double beta, double gamma, int timeStep);
void solveNonLinearSystem(VectorXd& u, const VectorXd& u_old, int N, double dt, double alpha, double beta, double gamma, int timeStep);
VectorXd flattenGrid(const vector<vector<double>>& u);
void unflattenGrid(const VectorXd& u_flat, vector<vector<double>>& u);
MatrixXd calculateJacobian(const VectorXd& u, int n, double dt, double alpha, double beta, double gamma);
VectorXd calculateResidual(const VectorXd& u, const VectorXd& u_old, int n, double dt, double alpha, double beta, double gamma, int timeStep);
double calculateResidualNorm(const VectorXd& delta_u);
//void writeVTKFile(const vector<vector<double>>& u, int n, double dx, int timestep);
double calculateL2Error(const vector<vector<double>>& u, int n, double t, double gamma);


int main(int argc, char* argv[]) {

    if (argc != 6) {
        cout << "Usage:" << argv[0] << "<Number of grid points> <Number of time steps> <Output interval> <Beta> <Gamma>" << endl;
        return 1;
    }


    int n = stoi(argv[1]);
    int nsteps = stoi(argv[2]);
    int outputInterval = stoi(argv[3]);
    double beta = stod(argv[4]);
    double gamma = stod(argv[5]);

    // Parameters
    double alpha = 1.0;
    double T = 1.0;
    double dt = T / nsteps;
    double dx = 1.0 / (n - 1);

    HeatEquationParams params{ n, nsteps, outputInterval, alpha, beta, gamma, dt, dx };

    // Grid intialziation
    int N = n * n;
    vector<vector<double>> u_grid(n, vector<double> (n, 0.0)); // Declare Temperature variable
    vector<vector<double>> u_old_grid(n, vector<double>(n, 0.0)); // Declare my old temperature
    initializeTemperature(u_grid, n ); // Intialize u(x, y, t=0)
   

    VectorXd u = flattenGrid(u_grid);
    VectorXd u_old = flattenGrid(u_old_grid);

    u_old = u; // Intialize u_old with intial temperature

 

    // Time integration

    for (int k = 0; k < nsteps; ++k) {
        if (beta != 0) {
            solveNonLinearSystem(u, u_old, N, dt, alpha, beta, gamma, k);
        }
        else {
            MatrixXd A = createMatrixA(N, dt, alpha);
            solveLinearSystem(u, A, N, dt, alpha, beta, gamma, k);
        }

        unflattenGrid(u, u_grid);
        applyDirichletBC(u_grid);

        // Update flattened vector for the next time step
        u = flattenGrid(u_grid);
        u_old = u;

        u_old = u; // update the old time step
        if (k % outputInterval == 0) {
            writeVTKFile(u, N, dx, k);
            double L2Error = calculateL2Error(u, N, k * dt, gamma);
            cout << "Time : " << k * dt << ", Newton Iterations: " << k << ", L2 Error: " << L2Error << endl;

            // writing in VTK file

        }
    }

    return 0;
}

void initializeTemperature(vector<vector<double>>& u, int N) {
    double dx = 1.0 / (N - 1);
    for (int i = 0; i < N; ++i) {
        double x = i * dx;
        for (int j = 0; j < N; ++j) {
            double y = j * dx;
            u[i][j] = sin(PI * x) * sin(PI * x) * sin(PI * y) * sin(PI * y);
        }
    }
}
double calculateSourceTerm(double x, double y, double u, double t, double alpha, double beta, double gamma) {
    return -2 * PI * sin(PI * x) * sin(PI * x) * sin(PI * y) * sin(PI * y) * sin(PI * gamma * t) * cos(PI * gamma * t)
        - 2 * alpha * PI * (cos(PI * x) * cos(PI * x) - sin(PI * x) * sin(PI * x)) * sin(PI * y) * sin(PI * y) * cos(PI * gamma * t) * cos(PI * gamma * t)
        - 2 * alpha * PI * (cos(PI * y) * cos(PI * y) - sin(PI * y) * sin(PI * y)) * sin(PI * x) * sin(PI * x) * cos(PI * gamma * t) * cos(PI * gamma * t)
        - beta * pow(u, 4);
}

void applyDirichletBC(vector<vector<double>>& u) {
    int n = sqrt(u.size());
    int N = u.size();
    for (int i = 0; i < N; ++i) {
        u[0][i * n] = 0;
        u[i * n + n - 1][i] = 0;
        u[i * n][0] = 0;
        u[i][(n - 1) * n + i] = 0;

    }
}

MatrixXd createMatrixA(int n, double dt, double alpha) {
    int N = n * n;
    MatrixXd A(N,N);
    A.setZero();  // Initializes all elements to zero

    double dx = 1.0 / (n - 1);
    double C = alpha * dt / (dx * dx);

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            int idx = i * n + j;
            A(idx, idx) = 1 + 4 * C;

            if (i - 1 >= 0) {
                A(idx, (i - 1) * n + j) = -C;
            }

            if (i + 1 < n) {
                A(idx, (i + 1) * n + j) = -C;
            }

            if (j - 1 >= 0) {
                A(idx, i * n + (j - 1)) = -C;
            }

            if (j + 1 < n) {
                A(idx, i * n + (j + 1)) = -C;
            }
        }
    }

    return A;
}
void solveLinearSystem(VectorXd& u, const MatrixXd& A, int N, double dt, double alpha, double beta, double gamma, int timeStep)
{
    VectorXd b = u;

    // Add the source term to b

    int n = sqrt(N);
    double dx = 1.0 / (n - 1);
    for (int i = 0; i < n; ++i) {
        double x = i * dx;
        for (int j = 0; j < n; ++j) {
            double y = j * dx;
            int idx = i * n + j;
            b(idx) += dt * calculateSourceTerm(x, y, u(idx), timeStep * dt, alpha, beta, gamma);
        }
        //solving using LU decompostion in Eigen library technique refered Numerical Reciepes in C++ by William H Press Chapt 2 pg 43 for Theory 
        VectorXd u_new = A.lu().solve(b);

        // Update u
        u = u_new;
}
}

void solveNonLinearSystem(VectorXd & u, const VectorXd & u_old, int N, double dt, double alpha, double beta, double gamma, int timeStep) {
        VectorXd delta_u;
        double tolerance = 1e-2;
        int maxIterations = 100;
        for (int iter = 0; iter < maxIterations; ++iter) {
            MatrixXd J = calculateJacobian(u, sqrt(N), dt, alpha, beta, gamma);
            VectorXd g = calculateResidual(u, u_old, sqrt(N), dt, alpha, beta, gamma, timeStep);

            // Solve the linearized system J* delta_u = -g

            delta_u = J.lu().solve(-g);

            // Update u
            u += delta_u;

            // Check for convergence

            if (calculateResidualNorm(delta_u) < tolerance) {
                break;
            }

        }
    }
VectorXd flattenGrid(const vector<vector<double>>&u) {
        int n = sqrt(u.size());
        VectorXd u_flat(n * n);
            for (int i = 0; i < n; ++i) {
                for (int j = 0; j < n; ++j) {
                    u_flat[i * n + j] = u[i][j];
                }
            }
        return u_flat;
    }
void unflattenGrid(const VectorXd & u_flat, vector<vector<double>>&u) {
        int n = sqrt(u_flat.size());
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                u[i][j] = u_flat[i * n + j];
            }
        }

    }
MatrixXd calculateJacobian(const VectorXd& u, int n, double dt, double alpha, double beta, double gamma)
{
    int N = n * n;
    MatrixXd J(N,N);
    J.setZero();
    double dx = 1.0 / (n - 1);
    double C = alpha * dt / (dx * dx);
    //double beta;
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            int idx = i * n + j;
            //J(idx, idx) = 1 + 4 * C - 4 * beta * dt * u[idx][idx] * u[idx][idx] * u[idx][idx];
            J(idx, idx) = 1 + 4 * C - 4 * beta * dt * u[idx] * u[idx] * u[idx];
            if (i - 1 >= 0) {
                J(idx, (i - 1) * n + j) = -C;
            }
            if (i + 1 < n) {
                J(idx, (i + 1) * n + j) = -C;

            }
            if (j - 1 >= 0) {
                J(idx, i * n + (j - 1)) = -C;
            }

            if (j + 1 < n) {
                J(idx, i * n + (j + 1)) = -C;
            }
        }
    }
    return J;
}


VectorXd calculateResidual(const VectorXd & u, const VectorXd & u_old, int n, double dt, double alpha, double beta, double gamma, int timeStep) {

        int N = n * n;
        VectorXd g(N);
        double dx = 1.0 / (n - 1);
        double Cx = alpha * dt / (dx * dx);
        double Cy = alpha * dt / (dx * dx);

        for (int i = 0; i < n; ++i) {
            double x = i * dx;
            for (int j = 0; j < n; ++j) {
                double y = j * dx;
                int idx = i * n + j;

                g(idx) = u[idx] - u_old[idx] - dt * (
                    Cx * (u[(i > 0 ? idx - n : idx)] - 2 * u[idx] + (i < n - 1 ? u[idx + n] : 0.0)) +
                    Cy * (u[(j > 0 ? idx - 1 : idx)] - 2 * u[idx] + (j < n - 1 ? u[idx + 1] : 0.0))
                    ) - dt * calculateSourceTerm(x, y, u[idx], timeStep * dt, alpha, beta, gamma);
                // Adjust for boundary condtions

                if (i == 0 || i == n - 1 || j == 0 || j == n - 1) {
                    g(idx) = 0.0;
                }
            }

        }

        return g;
    }

    double calculateResidualNorm(const VectorXd & delta_u) {
        return delta_u.norm();
    }
    
    void writeVTKFile(const vector<vector<double>>&u, int n, double dx, int timestep) {
        vtkSmartPointer<vtkImageData> imageData = vtkSmartPointer<vtkImageData>::New();
        imageData->SetDimensions(n, n, 1);
        imageData->AllocateScalars(VTK_DOUBLE, 1);

        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                double* pixel = static_cast<double*>(imageData->GetScalarPointer(i, j, 0));
                pixel[0] = u[i][j];
            }
        }

        string filename = "output_" + to_string(timestep) + ".vti";
        vtkSmartPointer<vtkXMLImageDataWriter> writer = vtkSmartPointer<vtkXMLImageDataWriter>::New();
        writer->SetFileName(filename.c_str());
        writer->SetInputData(imageData);
        writer->Write();
    }

    
    double calculateL2Error(const vector<vector<double>>&u, int n, double t, double gamma) {
        double error = 0.0;
        double dx = 1.0 / (n - 1);
        for (int i = 0; i < n; ++i) {
            double x = i * dx;
            for (int j = 0; j < n; ++j) {
                double y = j * dx;
                double exact = sin(PI * x) * sin(PI * x) * sin(PI * y) * sin(PI * y) * cos(PI * gamma * t);
                error += (u[i][j] - exact) * (u[i][j] - exact);
            }
        }
        return sqrt(error);
    }
    