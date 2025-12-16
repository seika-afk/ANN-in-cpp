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

class ANN{

	public :
	string ac_fn;
	vector<float> inputs;
	vector<vector<float>> zs_;
	

	AcFn af;
	Losses lf;
	Gradients gd;
	map<string,variant<string,int,float>> layer_configs;
	vector<map<string,variant<string,int,float>>> h_layers;
	
// ### for storing weights for each neuron in each hidden layer  and its output 
	vector <vector<float>> activations_;
	vector <vector<float>> biases;

	vector <vector<float>> weights;

	vector<vector<float>> deltas_;



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



//######################### BUILD
void build(){
	weights.clear();
	biases.clear();

	for (auto &layer : h_layers){

	int input_shape=get<int>(layer["input_shape"]);
	int num_neurons=get<int>(layer["num_neurons"]);
	string method= get<string>(layer["weight_init_method"]);

		vector<float> layer_weights;
			for (int i=0; i <num_neurons*input_shape;i++){

				layer_weights.push_back(

						give_weight(method,input_shape,num_neurons)

						);
			}
		weights.push_back(layer_weights);
//bias
		vector<float> layer_biases(num_neurons,0.0f);
		biases.push_back(layer_biases);



			}
		



			



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
	//	cout<<"Final  Hidden Layer Added !"<<endl;



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
		activations_.clear();
		zs_.clear();
		deltas_.clear();
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
	

		//this was for weight generation ,but now its moved to fn build
	//int max_neurons=4;
	//for (auto h: this->h_layers){
	
	//	set_weights(h,max_neurons) ;//pass h_layers from here , all of them in loop
		
//	}
//	for(auto h : this->h_layers){
//this->biases.push_back(0);
//	}

	// run each layer with input in a loop
	//run_hidden_layer();
	
	
//printing all weights:
	for (int i=0;i<this->weights.size();i++){
	cout<<"----------------"<<endl;
	cout<<"Layer : "<<i<<endl;
for(int j=0;j<this->weights[i].size();j++){

	cout<<weights[i][j]<<"  ";

}
cout<<endl;}

//calculating loss





//this is mandatory utilizing the first layer
//run_hidden_layer(this->h_layers[0],0);
	//here 0 is index of layer , which layer is working to use its weights


//NOw we need for all layers ,other than last , for output, where loss will be calc


//run_hidden_layer(this-> h_layers[1],1);
//run_hidden_layer(this-> h_layers[2],2);


for(int i=0;i<h_layers.size();i++){

run_hidden_layer(h_layers[i],i);
}

for(auto a:activations_){
	cout<<"New layer"<<endl;
	for(auto aa:a){
cout<<aa<<endl;

	}
}
//cout << activations_[0].back()<<endl;


	float loss = calcLoss(loss_fn,yi,activations_.back().back());
	cout<<"Loss :  "<<loss<<endl;



	//RUNNING BACKPROPOGATION
	string layer_loss_fn= get<string>(h_layers.back()["loss_fn"]);
	float layer_lr= get<float>(h_layers.back()["lr"]);
	//cout<<layer_loss_fn<<endl;
	//cout<<layer_lr<<endl;
	
int L= activations_.size()-1;

vector<float> delta_L;
	float a = activations_[L][0];
	float z= zs_[L][0];
	
// dL/da
		float dL_da;
	if (layer_loss_fn == "mae"){
    		dL_da = gd.mae_grad(yi, a);}
		else{
    			dL_da = gd.bce_grad(yi, a);}

// da/dz
	string ac = get<string>(h_layers[L]["ac_fn"]);
	float da_dz;
	if (ac == "relu")
    		{da_dz = gd.relu_grad(a);}
else
{ da_dz = gd.sigmoid_grad(a);
}
// delta = dL/dz
		float delta = dL_da * da_dz;
	delta_L.push_back(delta);

	deltas_.push_back(delta_L);


for (int l=L-1;l>=0;l--){
vector<float> delta_layer;
for(int i=0;i<activations_[l].size();i++){
float sum=0.0f;

for (int j=0;j<deltas_[0].size();j++){
   sum += deltas_[0][j] * weights[l+1][j * activations_[l].size() + i];
     

}
        string ac = get<string>(h_layers[l]["ac_fn"]);
        float da_dz = (ac == "relu") ? gd.relu_grad(activations_[l][i])
                                     : gd.sigmoid_grad(activations_[l][i]);


delta_layer.push_back(sum * da_dz);
}

 deltas_.insert(deltas_.begin(), delta_layer); 
}

//########## weight updation 
for (int l = 0; l < weights.size(); l++) {
    for (int i = 0; i < activations_[l].size(); i++) {
for (int j = 0; j < (l == 0 ? inputs.size() : activations_[l-1].size()); j++) {
            float a_j = (l == 0) ? inputs[j] : activations_[l-1][j];
        	    weights[l][i * (l == 0 ? inputs.size() : activations_[l-1].size()) + j] -= 
            	    layer_lr * deltas_[l][i] * a_j;
        }
        
	biases[l][i] -= layer_lr * deltas_[l][i];
    }
}




for (const auto& vec : activations_) {
    for (float v : vec) {
        cout << v << " ";
    }
    cout << endl;
}

		//float gradient=grad(loss_fn,ac_fn,yi,activations_[0],inputs[0]);
		// yi-> true value, a-> predicted value
		//inputs[0]-> input of that particular neuron of that layer
		//
		//
		//using formula => wnew=wold-n*gradient
	
		//float updated_weight= weights[0][0]-((layer_lr)*(gradient));

		//cout<<"UPdating Weight from : "<<weights[0][0]<<"to "<<updated_weight <<endl;




//end of fn
}
	void run_hidden_layer(map<string,variant<string,int,float>> h_layer,int ind_layer){
		zs_.push_back({});

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
	a=run_neuron(this->inputs,ac,neurons_weights[i],this->biases[ind_layer][i]);
//cout<<a<<endl;
}
else{
a = run_neuron(activations_.back(),ac,neurons_weights[i],this->biases[ind_layer][i]);
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
		cout<<"z="<<z<<endl;
		//cout<<"Weighted Sum is : "<<z<<endl;
	//calculating activation fn
	zs_.back().push_back(z);
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


void train(vector<vector<float>> X , vector<float>Y ,int epochs){
// run through each epoch ->
// add input
// run layer
// calc loss 
for (int epoch=0;epoch<epochs;epoch++){
float total_loss=0.0;

for(int i =0;i<X.size();i++){
add_input(X[i]);
	run_layer(get<string>(h_layers.back()["loss_fn"]),Y[i]);
total_loss+=calcLoss(get<string>(h_layers.back()["loss_fn"]),Y[i],activations_.back().back());

}
cout<<"For epoch : "<<epoch+1<< "/" << "- Loss :"<<total_loss/X.size()<<endl;
}


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

//vector<float> inputs ={0.2,0.9};
    vector<vector<float>> X = { {0.2, 0.9}, {0.5, 0.1}, {0.8, 0.7}, {0.3, 0.4} };
    vector<float> Y = { 0.5, 0.2, 0.8, 0.4 };

//ann.add_input(inputs);
ann.add_layer(2,2,"he","relu");
ann.add_layer(2,3,"he","relu");
ann.add_layer(3,1,"he","sigmoid","mae",0.1);
//tobe used for back propogation ann.add_layer(3,1,"he","relu","mae",0.1);
ann.build();

//ann.run_layer("mae",0.1);
ann.train(X,Y,100);
return 0;
}




