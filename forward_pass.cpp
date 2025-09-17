#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include "../Core-Foundations/TensorClass/neuroBlock.h"
#include "../Core-Foundations/Activation_Functions/AcFn.h"
#include <cmath>
using namespace std;
class ANN{

	public :
	string ac_fn;


//############################### INITIALIZATION METHODS



//########################### XAVIER : for tanh and sigmoid

	float xavier_init(int n_in,int n_out){
	
//seeding for random gen
static bool seeded = false;         
    if (!seeded) {
        std::srand(std::time(0));   
        seeded = true;
    }
	
    float p_limit=sqrt(6.0f/(n_in+n_out));
    float n_limit=-(p_limit);
    float weight=n_limit+ static_cast<float>(rand()) / static_cast<float>(RAND_MAX )* (p_limit - n_limit);
	
return weight;
	}

//########################## HE : for relu


	float he_init(int n_in){
	
//seeding for random gen
static bool seeded = false;         
    if (!seeded) {
        std::srand(std::time(0));   
        seeded = true;
    }
	
    float p_limit=sqrt(6.0f/(n_in));
    float n_limit=-(p_limit);
    float weight=n_limit+ static_cast<float>(rand()) / static_cast<float>(RAND_MAX )* (p_limit - n_limit);
	
return weight;
	}





//########################### FORWARD PASS	
	//ANN(int neurons, int h_layers,string ac_fn,weight_initialization_method){
		//this->ac_fn=ac_fn;


	void forward_pass( const vector<float> inputs){
		int n_out=1;
		vector<float> weights ={};

		for (int i=0;i<inputs.size();i++){
			weights.push_back(he_init(inputs.size()));
		}
		cout<<"Weights: ";
		for (auto w :weights)cout<<w<<" ";
		cout<<endl;
	}


//################3

	};

int main(){

vector<float> inputs ={1,2,3};


//xavier init
//ANN ann;
//ann.forward_pass(inputs);

//he init
ANN ann1;
ann1.forward_pass(inputs);


return 0;
}





