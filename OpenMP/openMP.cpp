#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <chrono>
#include <omp.h>

using namespace std;

const double G = 6.67430e-3; // scaled gravitational constant
const double EPS = 1e-5; 

// each particle has position (x, y), velocity (vx, vy), and mass
struct Particle {
    double x, y;
    double vx, vy;
    double mass;
};

int main(int argc, char* argv[]) {

    int N; // number of particles
    int iterations; // number of iterations for the simulation
    double dt; // time step
    int num_of_threads; // number of threads to use in parallel regions
    int available_threads = omp_get_max_threads(); // get the maximum number of threads available on the system

    cout << "**************************************************\n";
    cout << "*     N-Body Simulation Using OpenMP             *\n";
    cout << "**************************************************\n\n";

    // read number of particles 
    cout << "Enter number of particles \n";
    cin >> N;

    // read the number of iterations 
    cout << "Enter number of iterations \n";
    cin >> iterations;

    // read the time step 
    cout << "Enter time step \n";
    cin >> dt;

    // read number of threads and print the options for the user to choose from, we want to make sure that the 
    // number of threads chosen is a factor of N to ensure better load balancing among threads
    cout << "Enter number of threads so that it is in range of ";
    for (int i = 1; i <= available_threads; i++) {
        if (N % i == 0) {
            cout << i << " ";
        }
    }
    cout << "\n";
    cin >> num_of_threads;
    // validate the number of threads entered by the user, it should be a positive integer, less than or equal to 
    // the available threads, and a factor of N
    while ( num_of_threads < 1 || num_of_threads > available_threads || N % num_of_threads != 0 ) {
        if(num_of_threads < 1) {
            cout << "Number of threads must be a positive integer. Enter again: ";
        } else if (num_of_threads > available_threads) {
            cout << "Number of threads cannot exceed the maximum available threads (" << available_threads << "). Enter again: ";
        } else if (N % num_of_threads != 0) {
            cout << "Number of threads must be a factor of the number of particles (" << N << "). Enter again: ";
        }
        cin >> num_of_threads;
    }


    // Initialize particles
    vector<Particle> particles(N);

    // Force vectors
    vector<double> fx(N, 0.0);
    vector<double> fy(N, 0.0);

    srand(time(0));

    // Randomly initialize positions, velocities, and masses
    for (int i = 0; i < N; i++) {
        particles[i].x = rand() % 100;
        particles[i].y = rand() % 100;

        particles[i].vx = 0.0;
        particles[i].vy = 0.0;

        // +1 so that mass is never zero KG
        particles[i].mass = (rand() % 100) + 1;
    }

    cout << "\n";
    cout << " -------------------------\n";
    cout << "|      INITIALIZING       |\n";
    cout << " -------------------------\n";

    // Only print initial positions and mass if the number of particles is 15 or less 
    if(N <= 15) { 
        cout << "Initial positions and mass of particles:\n\n";
        for (int i = 0; i < N; i++) {
            cout << "Particle " << i
                << " Position: ("
                << particles[i].x << ", "
                << particles[i].y
                << "), Mass: " << particles[i].mass << " Kg\n";
        }
    }

    cout << "\n";
    cout << "=====================================\n";
    cout << "      SIMULATION CONFIGURATION       \n";
    cout << "=====================================\n";

    cout << "Particles           : " << N << "\n";
    cout << "Iterations          : " << iterations << "\n";
    cout << "Time Step (dt)      : " << dt << "\n";
    cout << "Threads Used        : " << num_of_threads << "\n";
    cout << "Max Available       : " << available_threads << "\n";

    cout << "=====================================\n\n";

    cout << "\n";
    cout << " -------------------------\n";
    cout << "|   STARTING SIMULATION   |\n";
    cout << " -------------------------\n";

    // Start Timer
    auto start = chrono::high_resolution_clock::now();

    // Simulation loop
    for (int iter = 0; iter < iterations; iter++) {
            
        // Reset forces to zeroes for each iteration
        fill(fx.begin(), fx.end(), 0.0);
        fill(fy.begin(), fy.end(), 0.0);

        // Compute pairwise forces between particles
        // parallelize the outer loop with guided scheduling to balance load with 10 min chunk size
        #pragma omp parallel for schedule(dynamic,10) num_threads(num_of_threads)  // we can add num_threads(5) to specify the number of threads but it's optional
        for (int i = 0; i < N; i++) {

            // local Variables, instead of accessing the shared fx and fy arrays in the inner loop, 
            // we can use local variables to accumulate the forces for each particle and then update 
            // the shared arrays after the inner loop
            double local_fx = 0.0;
            double local_fy = 0.0;

            for (int j = 0; j < N; j++) {

                // Skip self-interaction
                if (i == j)
                    continue;

                // Compute distance components that will be used to calculate the force
                double dx = particles[j].x - particles[i].x; // distance in x
                double dy = particles[j].y - particles[i].y; // distance in y

                // scalar distance between the two particles
                // EPS is added to prevent division by zero in case particles are very close
                double dist = sqrt(dx * dx + dy * dy + EPS); 

                // force as in Newton's law of universal gravitation, G is constant and is defined above
                double force =
                    (G * particles[i].mass * particles[j].mass)
                    / (dist * dist);

                // since we have the magnitude of the force, which is scalar,
                // we need to find the components of the force in x and y directions
                local_fx += force * dx / dist; // force component in x direction
                local_fy += force * dy / dist; // force component in y direction
            }

            // Store results in the shared array
            fx[i] = local_fx;
            fy[i] = local_fy;
        }

        // Update velocity and position
        #pragma omp parallel for schedule(dynamic,10) num_threads(num_of_threads) // parallelize the update loop also
        for (int i = 0; i < N; i++) {

            // calculate acceleration in x and y directions
            // acceleration = force / mass
            double ax = fx[i] / particles[i].mass; // acceleration in x direction
            double ay = fy[i] / particles[i].mass; // acceleration in y direction

            // calculate velocity in x and y directions
            // velocity = velocity + acceleration * time step, our time step is defined above as dt
            particles[i].vx += ax * dt; // velocity in x direction
            particles[i].vy += ay * dt; // velocity in y direction

            // finally we calculate the new x and y positions
            // position = position + velocity * time step
            particles[i].x += particles[i].vx * dt; // position in x direction
            particles[i].y += particles[i].vy * dt; // position in y direction
        }
    }

    // End Timer
    auto end = chrono::high_resolution_clock::now();

    // Calculate duration
    auto duration = chrono::duration_cast<chrono::milliseconds>(end - start);
    
    cout << "\n";
    cout << " -------------------------\n";
    cout << "|    SIMULATION ENDED     |\n";
    cout << " -------------------------\n\n";
    cout << "Simulation completed in " << duration.count() << " milliseconds.\n\n";

    // Print final positions also only if the number of particles is 15 or less
    if(N <= 15) { 
        cout << "Particles' final positions after " << iterations << " iterations:\n\n";
            for (int i = 0; i < N; i++) {
                cout << "Particle " << i
                    << " Position: ("
                    << particles[i].x << ", "
                    << particles[i].y
                    << "), Mass: " << particles[i].mass << " Kg\n";
            }
    }

    return 0;
}