#include <iostream>
#include <variant>
#include <vector>
#include <cstdlib>
#include <ctime>
#include "../Core-Foundations/TensorClass/neuroBlock.h"
#include "../Core-Foundations/Activation_Functions/AcFn.h"
#include "loss_fns/Losses.h"
#include "gradients/Gradients.h"
#include <cmath>
#include <map>
using namespace std;
//
class ANN{

	public :
	string ac_fn;
	vector<float> inputs;
	AcFn af;
	Losses lf;
	Gradients gd;
	map<string,variant<string,int,float>> layer_configs;
	vector<map<string,variant<string,int,float>>> h_layers;

// ### for storing weights for each neuron in each hidden layer  and its output 
	vector <vector<float>> activations_;
	vector <float> biases;

	vector <vector<float>> weights;




//############################### INITIALIZATION METHODS
	void add_input(vector<float> single_line_input){
	inputs=single_line_input;
	}


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



//########################### FN TO ADD LAYER : fn to ann layers and fn to add final _layer


	//### FN TO ADD ANN LAYER

	void add_layer(int input_shape,int num_neurons,string weight_init_method,string ac_fn ){
		// setting configs
		this->layer_configs= {{"input_shape",input_shape},{"num_neurons",num_neurons},{"weight_init_method",weight_init_method},{"ac_fn",ac_fn}};
		this->h_layers.push_back(layer_configs);
		//cout<<"New Hidden Layer Added !"<<endl;
		
	

		}

//FN TO ADD ANN_LAST_LAYER
	void add_layer(int input_shape,int num_neurons,string weight_init_method,string ac_fn,string loss_fn,float lr){
		// setting configs
		this->layer_configs= {{"input_shape",input_shape},{"num_neurons",num_neurons},{"weight_init_method",weight_init_method},{"ac_fn",ac_fn},{"loss_fn",loss_fn},{"lr",lr}};
		this->h_layers.push_back(layer_configs);
		cout<<"Final  Hidden Layer Added !"<<endl;



		}



//################# WEIGHT INIT METHOD

	void set_weights(map<string,variant<string,int,float>> layer_configs,int max_neuron){

		// run a loop to run till max_neuron times ,and if it reaches more than current layers number :
		// make it initialize weights  : => rows=>hidden layer , cols= neurons
	
	vector<float> to_be_stored;
	//params for layer_confi
	int num_neurons=get<int>(layer_configs["num_neurons"]);
	int n_in= get<int>(layer_configs["input_shape"]);
	int n_out= get<int>(layer_configs["num_neurons"]);
	int input_shape=get<int>(layer_configs["input_shape"]);
	string method=get<string>(layer_configs["weight_init_method"]);
	for (int i=0 ;i<num_neurons*input_shape; i++){
		
		to_be_stored.push_back(give_weight(method,n_in,n_out));
	
	}
	//for (int j =num_neurons;j<max_neuron;j++){
	//to_be_stored.push_back(NAN);

	//}
	//appended bias at ends of weights
	this->weights.push_back(to_be_stored);






	}
	float give_weight(string weight_init_method,int n_in,int n_out){

			if (weight_init_method=="xavier"){
			return(xavier(n_in,n_out));
			}
			else{
		return(he(n_in));


			}



	}

	void run_layer(string loss_fn,float yi){
	//int i =0;
	//for (auto h :this->h_layers){
		//cout <<"Hidden Layer "<<i<<" : "<<endl;
		//cout<<"---------------------"<<endl;
		//	for (auto & [key,val] : h){
		//		cout<<key<<" : ";
		//		std::visit([](auto&&arg){cout<<arg;},val);
		//		cout<<endl;
		//	}
		//	i=i+1;
		//}
	
	//set weights
	//function to set weight for each neuron in each layer
	
	int max_neurons=4;
	for (auto h: this->h_layers){
	
		set_weights(h,max_neurons) ;//pass h_layers from here , all of them in loop
		
	}
	for(auto h : this->h_layers){
this->biases.push_back(0);
	}

	// run each layer with input in a loop
	//run_hidden_layer();
	
	
//printing all weights:
	for (int i=0;i<this->weights.size();i++){
	cout<<"----------------"<<endl;
	cout<<"Layer : "<<i<<endl;
for(int j=0;j<this->weights[i].size();j++){

	cout<<weights[i][j]<<"  ";

}

//calculating loss


}


//this is mandatory utilizing the first layer
run_hidden_layer(this->h_layers[0],0);
	//here 0 is index of layer , which layer is working to use its weights


//NOw we need for all layers ,other than last , for output, where loss will be calc


run_hidden_layer(this-> h_layers[1],1);
run_hidden_layer(this-> h_layers[2],2);


for(auto a:activations_){
	cout<<"New layer"<<endl;
	for(auto aa:a){
cout<<aa<<endl;

	}
}
//cout << activations_[0].back()<<endl;



	float loss = calcLoss(loss_fn,yi,activations_.back().back());
	cout<<"Loss :  "<<loss<<endl;




//end of fn
}
	void run_hidden_layer(map<string,variant<string,int,float>> h_layer,int ind_layer){
// todo
// in this layer
// - run each neuron to number of neurons, take previous all input and do the calc part in it . 
// - produce output
// - store that output


//imagining -> [1,2,3,4] -> [1,2],[3,4]-> we need a fn to divide a array into parts acc bby dividence
//running all neutrons from the current layer

	
	//this part will be dynamic ,that is weights
	
	//vector<float> weight_1neuron(this->weights[0].begin(), this->weights[0].begin() + inputs.size());
//cout<<"performing partition of the wieghts and showing :"<<endl;
//	vector<vector<float>> res= part_weights(weights[1],2);
//	for(int i =0; i<res.size();i++){
//		for (auto h: res[i]){
//cout<<h<<endl;
//		}
//		cout<<"for another neutron :-----------"<<endl;
		

//	}
	int neurons=std::get<int>(h_layer["num_neurons"]);
	string ac=get<string>(h_layer["ac_fn"]);


	int input_shape=get<int>(h_layer["input_shape"]);

vector<vector<float>> neurons_weights = part_weights(weights[ind_layer], neurons, input_shape);


vector<float> activations_layer;
for (int i =0;i<neurons_weights.size();i++){
float a;
if (ind_layer==0){
	a=run_neuron(this->inputs,ac,neurons_weights[i],this->biases[ind_layer]);
//cout<<a<<endl;
}
else{
a = run_neuron(activations_.back(),ac,neurons_weights[i],this->biases[ind_layer]);
//cout<<"heh";
}
activations_layer.push_back(a);

}

activations_.push_back(activations_layer);

//endoffunction
	}
vector<vector<float>> part_weights(vector<float> weight_part, int num_neurons, int input_size) {
    vector<vector<float>> res;
    int k = 0;
    for (int i = 0; i < num_neurons; i++) {
        vector<float> neuron_weights;
        for (int j = 0; j < input_size; j++) {
            neuron_weights.push_back(weight_part[k++]);
        }
        res.push_back(neuron_weights);
    }
    return res;
}

	float run_neuron(vector<float> inputs,string ac_fn,vector<float> weight_,float bias){
	
	float z=0.0;

	//calculating weighted sumz
	for (int i =0; i<inputs.size();i++){
  			z+=inputs[i]* weight_[i];

		} 
		z=z+bias;
		//cout<<"Weighted Sum is : "<<z<<endl;
	//calculating activation fn

	float activation;
		// Applying activation fn 
		if (ac_fn =="relu"){
		activation=af.ReLU(z);}
		else{
activation=af.sigmoid(z);

		}
		return activation;


	}

//############################## SAMPLE NEURON
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
		// yi-> true value, a-> predicted value
		//inputs[0]-> input of that particular neuron of that layer
		//
		//
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

//vector<float> inputs ={1,2,3};


//xavier init
//ANN ann;
//ann.forward_pass(inputs);

//he init
//ANN ann1;
//ann1.forward_pass(inputs);



// demoing Z
//ANN ann;
//ann.input_layer(inputs);
//ann.neuron(3,0.7,1,"he","relu","mae",0.01);

// ##### HOW I WANT IT TO LOOK LIKE:

ANN ann;
//ann.input_layer(inputs); // inputs being an array of features of first row
//ann.hidden_layer(input_shape,number_of_neurons,weight_init,activation_fn)
//ann.hidden_layer(input_shape_of_Previous,number of neurons ,weight init method, activation fn)
//ann.final_node(input_shape,number of possibility , weght init , activation fn,loss_fn,lr)


// for normal adding layer: input_shape(basically if input ,just input size[num of features], else the num of neurons in prev hidden layer)
// input_shape | number of neurons | weight_init | ac_fn

// if at last neuron : input_shape | number of neurons | weight_init | ac_fn | loss_fn | learning rate



//for now input :

vector<float> inputs ={0.2,0.9};

ann.add_input(inputs);
ann.add_layer(2,2,"he","relu");
ann.add_layer(2,3,"he","relu");
ann.add_layer(3,1,"he","relu");
//tobe used for back propogation ann.add_layer(3,1,"he","relu","mae",0.1);

ann.run_layer("mae",0.1);

return 0;
}




