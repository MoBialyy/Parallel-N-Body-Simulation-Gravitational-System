#include <mpi.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>
 
using namespace std;
 
const double G = 6.67430e-3;
const double EPS = 1e-5;
const double dt = 0.01;
 
struct Particle {
    double x, y;
    double vx, vy;
    double mass;
};
 
int main(int argc, char** argv) {
 
    MPI_Init(&argc, &argv);
 
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
 
    int N, iterations;
 
    // input only from root
    if (rank == 0) {
        cout << "Enter number of particles: ";
        cin >> N;
 
        cout << "Enter iterations: ";
        cin >> iterations;
    }
 
    MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&iterations, 1, MPI_INT, 0, MPI_COMM_WORLD);
 
    int localN = N / size;
 
    vector<Particle> particles(N);
    vector<Particle> localParticles(localN);
    vector<Particle> globalParticles(N);
 
    // initialize in root
    if (rank == 0) {
        srand(time(0));
 
        for (int i = 0; i < N; i++) {
            particles[i].x = rand() % 100;
            particles[i].y = rand() % 100;
            particles[i].vx = 0;
            particles[i].vy = 0;
            particles[i].mass = (rand() % 100) + 1;
        }
    }
 
    MPI_Scatter(particles.data(), localN * sizeof(Particle), MPI_BYTE,
                localParticles.data(), localN * sizeof(Particle), MPI_BYTE,
                0, MPI_COMM_WORLD);
 
    // START TIMER
    double start_time = MPI_Wtime();
 
    for (int iter = 0; iter < iterations; iter++) {
 
        MPI_Allgather(localParticles.data(), localN * sizeof(Particle), MPI_BYTE,
                      globalParticles.data(), localN * sizeof(Particle), MPI_BYTE,
                      MPI_COMM_WORLD);
 
        vector<double> fx(localN, 0.0);
        vector<double> fy(localN, 0.0);
 
        for (int i = 0; i < localN; i++) {
 
            int gi = rank * localN + i;
 
            double fx_i = 0.0;
            double fy_i = 0.0;
 
            for (int j = 0; j < N; j++) {
 
                if (j == gi) continue;
 
                double dx = globalParticles[j].x - globalParticles[gi].x;
                double dy = globalParticles[j].y - globalParticles[gi].y;
 
                double dist = sqrt(dx * dx + dy * dy + EPS);
 
                double force =
                    (G * globalParticles[gi].mass * globalParticles[j].mass)
                    / (dist * dist);
 
                fx_i += force * dx / dist;
                fy_i += force * dy / dist;
            }
 
            fx[i] = fx_i;
            fy[i] = fy_i;
        }
 
        for (int i = 0; i < localN; i++) {
 
            double ax = fx[i] / localParticles[i].mass;
            double ay = fy[i] / localParticles[i].mass;
 
            localParticles[i].vx += ax * dt;
            localParticles[i].vy += ay * dt;
 
            localParticles[i].x += localParticles[i].vx * dt;
            localParticles[i].y += localParticles[i].vy * dt;
        }
 
        MPI_Barrier(MPI_COMM_WORLD);
    }
 
    // END TIMER
    double end_time = MPI_Wtime();
 
    MPI_Gather(localParticles.data(), localN * sizeof(Particle), MPI_BYTE,
               particles.data(), localN * sizeof(Particle), MPI_BYTE,
               0, MPI_COMM_WORLD);
 
    if (rank == 0) {
 
        //cout << "\nFinal Positions:\n";
 
        // for (int i = 0; i < N; i++) {
        //     cout << "Particle " << i << " ("
        //          << particles[i].x << ", "
        //          << particles[i].y << ")\n";
        // }
 
        cout << "\nExecution Time: "
             << (end_time - start_time)
             << " seconds\n";
    }
 
    MPI_Finalize();
    return 0;
}