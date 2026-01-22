#include <iostream>
#include <fstream>

#include "integrator.hpp"
#include "brusselator.hpp"
using namespace std;
double t = 30;
double Ca = 1.0;
double Cb = 1.7;
double dt = 0.15;
double current_time = 0;
typedef Eigen::Vector2d state;

Eigen::Vector2d C({1.0 , 1.0});





// construct RHS with chosen parameters
state f(const state& C) {
    return brusselator::odeSystem(C, Ca, Cb);
}

int main() {
    
    ofstream outFile;
    outFile.open("output.csv");
    char choice;
    char ask;
    cout << "Time for simulation requirement"  
        << t << endl;
    
    cout << "Please enter The step size " 
        << dt <<endl;
    if (!outFile.is_open)
    {
        cout << "Could not open the file\n";
        exit(EXIT_FAILURE);
    }
    
vienna : 
    cout << "Which time scheme do you want" << 
        << "a) EULER                        << b)RK4\n "  ;
    cin.get(choice).get();
    switch (choice)
    {
    case 'a' :
        for (current_time = 0; current_time <= t; current_time += dt)
        {
            outFile << "Writing the file values" << endl;
            C = integrate::Euler(f, C, dt);




            outFile << "Simulated timesteps" << current_time + dt << endl;
            outFile << "Concentration : " << C[0] << endl;
            outFile << "Concentration : " << C[1] << endl;

            cout << "The Output Value: C1" << C[0] << endl;
            cout << "THe Output Vlaue C2 " << C[1] << endl;

        }

        outFile.close();
    case 'b' :
        for (current_time = 0; current_time <= t; current_time += dt)
        {
            outFile << "Writing the file values" << endl;
            C = integrate::rk4(f, C, dt);




            outFile << "Simulated timesteps" << current_time + dt << endl;
            outFile << "Concentration : " << C[0] << endl;
            outFile << "Concentration : " << C[1] << endl;

            cout << "The Output Value: C1" << C[0] << endl;
            cout << "THe Output Vlaue C2 " << C[1] << endl;

        }

    default :
        cout << "Do you want to quit or continue\n" << 
            << "Enter q or c"endl;
        cin.get(ask).get();
        if ((ask == 'q') || (ask == 'Q'))
            break;
        else if ((ask == 'c') || (ask == 'C'))
            goto vienna;
        else
            cout << "We See each other next time yes" << endl;
            exit(EXIT_FAILURE);
    }
    
    




    

    return 0;


}
