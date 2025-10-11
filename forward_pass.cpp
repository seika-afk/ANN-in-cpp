#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include "../Core-Foundations/TensorClass/neuroBlock.h"
#include "../Core-Foundations/Activation_Functions/AcFn.h"
#include "loss_fns/Losses.h"
#include "gradients/Gradients.h"
#include <cmath>

using namespace std;
class ANN{

	public :
	string ac_fn;
	vector<float> inputs;
	AcFn af;
	Losses lf;
	Gradients gd;


//############################### INITIALIZATION METHODS



//########################### XAVIER : for tanh and sigmoid

	float xavier(int n_in,int n_out){
	
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


	float he(int n_in){
	
//seeding for random gen
static bool seeded = false;         
    if (!seeded) {
        std::srand(std::time(0));   
        seeded = true;
    }
	
    float p_limit=sqrt(6.0f/(n_in));
    float n_limit=-(p_limit);
    float weight=n_limit+ static_cast<float>(rand()) / static_cast<float>(RAND_MAX )* (p_limit - n_limit);
	//cout<<"Activated";
return weight;
	}





//########################### FORWARD PASS	
	//ANN(int neurons, int h_layers,string ac_fn,weight_initialization_method){
		//this->ac_fn=ac_fn;

// We are expecting input like ,single row, need to iterate of whole dataset themself
	void input_layer( const vector<float> input){
	this->inputs=input;
	}



	void neuron(int input_shape,float yi,int output_shape,string weight_init_method,string ac_fn,string loss_fn,float lr)
{

	int n_out=output_shape;

		vector<float> weights ={};

		for (int i=0;i<inputs.size();i++){

			if (weight_init_method=="xavier"){
			weights.push_back(xavier(inputs.size(),n_out));
			}
			else{
	weights.push_back(he(inputs.size()));


			}


			}

		//checking weights
		cout<<"Weights: ";
		for (auto w :weights)cout<<w<<" ";
		cout<<endl;

		//next steps :
		//applying z: weighted sum
		//applying activation fn  
		float bias=0.0;
		float z=0.0;

		for (int i =0; i<inputs.size();i++){
  			z+=inputs[i]* weights[i];

		} 
		z=z+bias;
		cout<<"Weighted Sum is : "<<z<<endl;
		float activation;
		// Applying activation fn 
		if (ac_fn =="relu"){
		activation=af.ReLU(z);}
		else{
activation=af.sigmoid(z);

		}
		cout<<"FInal Activation"<<activation<<endl;
//#### LOSS CALC
		cout<<"Loss : "<<this->calcLoss(loss_fn,yi,activation)<<endl;

		cout<<"Performing Weight Updation"<<endl;
// for now just taking for 1 ijnput: Trial
		float gradient=grad(loss_fn,ac_fn,yi,activation,inputs[0]);
		
		//using formula => wnew=wold-n*gradient
	
		float updated_weight= weights[0]-((lr)*(gradient));

		cout<<"UPdating Weight from : "<<weights[0]<<"to "<<updated_weight <<endl;





}

	float calcLoss(string loss_fn,float yi,float xi){
		if (loss_fn=="mae"){
   			return lf.mae(yi,xi);
		}
		else if (loss_fn=="bce") {
				return lf.bce(yi,xi);
		}
		else{
cout<<"Unknown Loss.";
return 0.0;
		}
		

		
	}	

//######################### CALCULATE GRADIENT

	float grad(string loss, string ac_fn,float y_loss,float a,float xi ){
	float loss_gradient=0;
	float ac_gradient=0;
	float final_gradient=0;
	
//calculating grad of loss
	if (loss=="mae"){
		loss_gradient=gd.mae_grad(y_loss,a);

	}
	if (loss=="bce"){

		loss_gradient=gd.bce_grad(y_loss,a);
	}

//GRadient for activation
	if (ac_fn=="relu"){
ac_gradient=gd.relu_grad(a);
		
	}
	if(ac_fn=="sigmoid"){
ac_gradient=gd.sigmoid_grad(a);
	}
	final_gradient=loss_gradient*ac_gradient*xi;

	return final_gradient;
	
	
	}


// ################### END OF CLASS
};

int main(){

vector<float> inputs ={1,2,3};


//xavier init
//ANN ann;
//ann.forward_pass(inputs);

//he init
//ANN ann1;
//ann1.forward_pass(inputs);



// demoing Z
ANN ann;
ann.input_layer(inputs);
ann.neuron(3,0.7,1,"he","relu","mae",0.01);


return 0;
}




